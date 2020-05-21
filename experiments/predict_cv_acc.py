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
from cnn_limits.layers import proj_relu_kernel

faulthandler.enable()

experiment = sacred.Experiment("predict_cv_acc", [SU.ingredient])
if __name__ == '__main__':
    SU.add_file_observer(experiment)


def dataset_targets(dset):
    _, y = next(iter(DataLoader(dset, batch_size=len(dset))))
    return np.asarray(y.numpy())


def centered_one_hot(y, N_classes=10):
    return np.eye(N_classes)[y]
    # oh = y[:, None] == np.arange(N_classes)
    # return (oh*2-1).astype(np.float64)
    # return (oh.astype(np.float64)*N_classes - 1) / (N_classes-1)


@experiment.config
def config():
    kernel_matrix_path = "/scratch/ag919/logs/save_new/166"

    N_classes = 10
    n_grid_opt_points = 1000

    N_files = 9999
    eig_engine = "scipy"
    n_splits = -1
    multiply_var = False
    apply_relu = False


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
    if eig_engine == "torch":
        Kxx = torch.from_numpy(Kxx).to(dtype=torch.float64, device='cuda')
        out = torch.symeig(Kxx, eigenvectors=True, upper=not lower)
        return EigenOut(*(a.cpu().numpy() for a in (out.eigenvalues,
                                                    out.eigenvectors)))
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


def fold_idx(n_samples, n_splits):
    for start in range(n_splits):
        test_idx = slice(start, n_samples, n_splits)
        idx = np.arange(n_samples)
        train_idx = np.delete(idx, idx[test_idx])
        yield train_idx, test_idx


@experiment.capture
def do_one_N(Kxx, Kxt, oh_train_Y, test_Y, n_grid_opt_points, n_splits,
             FY=None, lower=True):
    if n_splits == -1:
        return do_one_N_loo(Kxx, Kxt, oh_train_Y, test_Y, n_grid_opt_points,
                            FY=FY, lower=lower)

    fold_eig = [eigdecompose(Kxx[train_idx, :][:, train_idx], lower=lower)
                for train_idx, _ in fold_idx(len(Kxx), n_splits)]

    min_eigval = min(*[e.vals.min() for e in fold_eig])

    eigval_floor = max(0., -min_eigval)
    while min_eigval + eigval_floor <= 0.0:
        eigval_floor = np.nextafter(eigval_floor, np.inf)

    sigy_grid = np.exp(np.linspace(
        max(np.log(eigval_floor), -36), 5, n_grid_opt_points))
    # Make sure that none is lower than eigval_floor
    sigy_grid = np.clip(sigy_grid, eigval_floor, np.inf)

    folds = []
    for (train_idx, test_idx), eig in zip(fold_idx(len(Kxx), n_splits),
                                          fold_eig):
        _Kxt = Kxx[train_idx, :][:, test_idx]
        _oh_train_Y = oh_train_Y[train_idx, :]
        _test_Y = oh_train_Y[test_idx, :].argmax(-1)
        _, fold = accuracy_eig(eig, _Kxt, _oh_train_Y, _test_Y, sigy_grid)
        folds.append(fold)

    grid_acc = np.mean(folds, axis=0)
    sigy = np.expand_dims(sigy_grid[grid_acc.argmax()], 0)

    return (sigy_grid, grid_acc), accuracy_eig(
        eigdecompose(Kxx, lower=lower), Kxt, oh_train_Y, test_Y, sigy, FY=FY,
        lower=lower)

@experiment.capture
def do_one_N_loo(Kxx, Kxt, oh_train_Y, test_Y, n_grid_opt_points,
                 _log, FY=None, lower=True):
    eig = eigdecompose(Kxx, lower=lower)
    min_eigval = eig.vals.min()
    eigval_floor = max(0., -min_eigval)
    while min_eigval + eigval_floor <= 0.0:
        eigval_floor = np.nextafter(eigval_floor, np.inf)

    sigy_grid = np.exp(np.linspace(
        max(np.log(eigval_floor), -36), 5, n_grid_opt_points))
    # Make sure that none is lower than eigval_floor
    sigy_grid = np.clip(sigy_grid, eigval_floor, np.inf)

    assert np.all(np.isfinite(sigy_grid))

    eig = EigenOut(*map(jnp.asarray, eig))
    Kxt = jnp.asarray(Kxt)
    oh_train_Y = jnp.asarray(oh_train_Y)
    test_Y = jnp.asarray(test_Y)
    if FY is not None:
        FY = jnp.asarray(FY)

    if FY is None:
        FY = oh_train_Y
    alpha = eig.vecs.T @ FY
    # K^-1 = (eig.vecs / (eig.vals + sigy) @ eig.vecs.T
    # diag(K^-1) = (eig.vecs**2 / (eig.vals + sigy)).sum(1)
    eigvecs_sq = eig.vecs**2
    train_Y = jnp.argmax(oh_train_Y, -1)

    def loo_cv_acc(sigy):
        new_eigvals = eig.vals + sigy
        diag_K = (eigvecs_sq / new_eigvals).sum(1, keepdims=True)
        Ky = (eig.vecs / new_eigvals) @ alpha
        pred_F = oh_train_Y - Ky/diag_K
        pred_Y = jnp.argmax(pred_F, axis=-1)
        return (pred_Y == train_Y).mean(-1)

    loo_accuracies = [float(loo_cv_acc(sigy)) for sigy in sigy_grid]
    assert np.all(np.isfinite(loo_accuracies))
    sigy = np.expand_dims(sigy_grid[np.argmax(loo_accuracies)], 0)
    return (sigy_grid, loo_accuracies), accuracy_eig(
        eig, Kxt, oh_train_Y, test_Y, sigy, FY=FY, lower=lower)


@experiment.automain
def main_no_eig(kernel_matrix_path, multiply_var, _log, apply_relu, n_splits):
    kernel_matrix_path = Path(kernel_matrix_path)
    train_set, test_set = SU.load_sorted_dataset(
        dataset_treatment="load_train_idx",
        train_idx_path=kernel_matrix_path)

    train_Y = dataset_targets(train_set)
    oh_train_Y = centered_one_hot(train_Y).astype(np.float64)
    test_Y = dataset_targets(test_set)

    with h5py.File(kernel_matrix_path/"kernels.h5", "r") as f:
        N_layers, N_total, _ = f['Kxx'].shape

        all_N = list(itertools.takewhile(
            lambda a: a <= N_total,
            (2**i * 10 for i in itertools.count(0))))
        data = pd.DataFrame(index=range(N_layers), columns=all_N)
        accuracy = pd.DataFrame(index=range(N_layers), columns=all_N)

        new_base_dir = SU.base_dir()/f"n_splits_{n_splits}"
        os.makedirs(new_base_dir, exist_ok=True)

        for layer in reversed(data.index):
            Kxx = f['Kxx'][layer].astype(np.float64)
            mask = np.triu(np.ones(Kxx.shape, dtype=np.bool), k=1)
            Kxx[mask] = Kxx.T[mask]
            assert np.array_equal(Kxx, Kxx.T)
            assert np.all(np.isfinite(Kxx))

            Kxt = f['Kxt'][layer].astype(np.float64)
            try:
                Kx_diag = f['Kx_diag'][layer].astype(np.float64)
                Kt_diag = f['Kt_diag'][layer].astype(np.float64)
            except KeyError:
                Kx_diag = np.diag(Kxx)

            if multiply_var:
                assert np.allclose(np.diag(Kxx), 1.)
                Kxx *= np.sqrt(Kx_diag[:, None]*Kx_diag)
                Kxt *= np.sqrt(Kx_diag[:, None]*Kt_diag)
            else:
                assert not np.allclose(np.diag(Kxx), 1.)

            if apply_relu:
                prod12 = Kx_diag[:, None] * Kx_diag
                Kxx = np.asarray(proj_relu_kernel(Kxx, prod12, False))
                Kxx = Kxx * prod12    # Use covariance, not correlation matrix
                prod12_t = Kx_diag[:, None] * Kt_diag
                Kxt = np.asarray(proj_relu_kernel(Kxt, prod12_t, False))
                Kxt = Kxt * prod12_t

            for N in reversed(data.columns):
                # Made a mistake, the same label are all contiguous in the training set.
                train_idx = slice(0, N_total, N_total//N)
                data.loc[layer, N], accuracy.loc[layer, N] = do_one_N(
                    Kxx[train_idx, train_idx], Kxt[train_idx], oh_train_Y[train_idx], test_Y, n_splits=n_splits,
                    FY=None, lower=True)
                (sigy, acc) = map(np.squeeze, accuracy.loc[layer, N])
                _log.info(f"For layer={layer}, N={N}, sigy={sigy}; accuracy={acc}")
                pd.to_pickle(data, new_base_dir/"grid_acc.pkl.gz")
                pd.to_pickle(accuracy, new_base_dir/"accuracy.pkl.gz")
