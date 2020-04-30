import collections
import contextlib
import faulthandler
import itertools
import math
import os
import pickle
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import sacred
import scipy.linalg
import scipy.optimize
import torch
from torch.utils.data import DataLoader, Subset

import cnn_limits
import cnn_limits.sacred_utils as SU
from cnn_gp import create_h5py_dataset

faulthandler.enable()

experiment = sacred.Experiment("predict_cv_acc", [SU.ingredient])
if __name__ == '__main__':
    SU.add_file_observer(experiment)


def dataset_targets(dset):
    _, y = next(iter(DataLoader(dset, batch_size=len(dset))))
    return np.asarray(y.numpy())


def centered_one_hot(y, N_classes=10):
    oh = y[:, None] == np.arange(N_classes)
    return (oh.astype(np.float64)*N_classes - 1) / (N_classes-1)


@experiment.config
def config():
    kernel_matrix_path = "/scratch/ag919/logs/save_new/166"

    N_classes = 10
    layer_range = (2, 999, 3)
    n_grid_opt_points = 1000

    N_files = 9999
    eig_engine = "scipy"
    n_splits = 4



EigenOut = collections.namedtuple("EigenOut", ("vals", "vecs"))
@experiment.capture
def eigdecompose(Kxx, eig_engine, lower=True):
    if isinstance(Kxx, EigenOut):
        return Kxx
    if eig_engine == "jax":
        Kxx = jnp.asarray(Kxx, dtype=jnp.float64)
        return EigenOut(*jax.scipy.linalg.eigh(
            Kxx, lower=lower, eigvals_only=False, check_finite=False))
    if eig_engine == "magma":
        raise NotImplementedError
    if eig_engine == "scipy":
        return EigenOut(*scipy.linalg.eigh(
            Kxx.astype(np.float64, copy=False), lower=lower,
            eigvals_only=False, check_finite=False))
    raise ValueError(f"eig_engine={eig_engine}")


@experiment.capture
def accuracy_eig(Kxx, Kxt, oh_train_Y, test_Y, sigy_grid, _log, FY=None,
                 lower=True):
    """Evaluates accuracy for a path using the eigendecomposition.
    Returns the accuracy at each position
    """
    N, D = oh_train_Y.shape
    Kxt = jnp.asarray(Kxt)
    oh_train_Y = jnp.asarray(oh_train_Y)
    test_Y = jnp.asarray(test_Y)
    if FY is not None:
        FY = jnp.asarray(FY)

    eig = EigenOut(*map(jnp.asarray, eigdecompose(Kxx)))

    _log.debug("Calculating alpha and beta")
    if FY is None:
        FY = oh_train_Y
    alpha = eig.vecs.T @ FY
    gamma = Kxt.T @ eig.vecs

    def jax_acc(sigy):
        pred_F = (gamma / (eig.vals + sigy)) @ alpha
        pred_Y = jnp.argmax(pred_F, axis=-1)
        return (pred_Y == test_Y).mean(-1)

    accuracies = [float(jax_acc(sigy)) for sigy in sigy_grid]
    return sigy_grid, np.asarray(accuracies)


def fold_test_idx(n_samples, n_splits):
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int)
    fold_sizes[:n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        yield start, stop
        current = stop


def fold_eigdecompose(Kxx, start, stop, lower=True):
    m = len(Kxx) - (stop - start)
    train = np.zeros((m, m), dtype=Kxx.dtype)
    train[:start, :start] = Kxx[:start, :start]
    train[:start, start:] = Kxx[:start, stop:]
    train[start:, :start] = Kxx[stop:, :start]
    train[start:, start:] = Kxx[stop:, stop:]
    return eigdecompose(train, lower=lower)


@experiment.capture
def do_one_N(Kxx, Kxt, oh_train_Y, test_Y, n_grid_opt_points, n_splits,
             FY=None, lower=True):
    fold_eig = [fold_eigdecompose(Kxx, start, stop)
                for start, stop in fold_test_idx(len(Kxx), n_splits)]
    min_eigval = min(*[e.vals.min() for e in fold_eig])
    eigval_floor = (0. if min_eigval < 0 else
                    np.nextafter(np.abs(min_eigval), np.inf))
    # grid = np.exp(np.linspace(
    #     max(np.log(eigval_floor), -36), 5, n_grid_opt_points))

    folds = []
    # for (start, stop), eig in zip(fold_test_idx(len(Kxx), n_splits),
    #                             fold_eig):
    #     _Kxt = np.concatenate([Kxx[start:stop, :start].T,  # Use lower triangle
    #                         Kxx[stop:, start:stop]], 0)
    #     _oh_train_Y = np.concatenate(
    #         [oh_train_Y[:start], oh_train_Y[stop:]], 0)
    #     _test_Y = oh_train_Y[start:stop].argmax(-1)
    #     print(eig.vals, _Kxt)
    #     fold = accuracy_eig(eig, _Kxt, _oh_train_Y, _test_Y, grid)
    #     # fold = accuracy_eig(eigdecompose(Kxx), Kxt, oh_train_Y, test_Y, grid)
    #     folds.append(fold)

    # grid_acc = np.sum(folds, axis=0)
    # print("Grid_acc:", grid_acc)
    # sigy = np.expand_dims(grid[grid_acc.argmax()], 0)

    return folds, accuracy_eig(
        eigdecompose(Kxx, lower=lower), Kxt, oh_train_Y, test_Y, np.array([eigval_floor]), FY=FY,
        lower=lower)


@experiment.automain
def main_no_eig(kernel_matrix_path):
    train_set, test_set = SU.load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    oh_train_Y = centered_one_hot(train_Y)
    test_Y = dataset_targets(test_set)
    kernel_matrix_path = Path(kernel_matrix_path)

    with h5py.File(kernel_matrix_path/"kernels.h5", "r") as f:
        N_layers, _, _ = f['Kxx'].shape

        all_N = [2**i * 10 for i in range(8)]  # up to 1280
        data = pd.DataFrame(index=range(N_layers), columns=all_N)
        accuracy = pd.DataFrame(index=range(N_layers), columns=all_N)

        for layer in reversed(data.index):
            Kxx = f['Kxx'][layer]
            Kxt = f['Kxt'][layer]
            for N in data.columns:
                data.loc[layer, N], accuracy.loc[layer, N] = do_one_N(
                    Kxx[:N, :N], Kxt[:N], oh_train_Y[:N], test_Y, FY=None,
                    lower=True)
                pd.to_pickle(data, SU.base_dir()/"grid_acc.pkl.gz")
                pd.to_pickle(accuracy, SU.base_dir()/"accuracy.pkl.gz")
