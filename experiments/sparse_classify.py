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
import jax.scipy.linalg
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
from experiments.predict_cv_acc import (dataset_targets, centered_one_hot,
                                        EigenOut, eigdecompose, accuracy_eig, fold_idx)
from experiments.predict_cv_acc import experiment as predict_cv_acc_experiment

faulthandler.enable()

experiment = sacred.Experiment("sparse_classify", [SU.ingredient, predict_cv_acc_experiment])
if __name__ == '__main__':
    SU.add_file_observer(experiment)


@experiment.capture
def do_one_N_chol(Kuu, Kux, Kut, oh_train_Y, test_Y, n_splits, predict_cv_acc, _log):
    n_grid_opt_points = predict_cv_acc['n_grid_opt_points']
    if n_splits == -1:
        raise NotImplementedError("leave-one-out")

    try:
        Luu = np.linalg.cholesky(Kuu)
    except np.linalg.LinAlgError:
        for i in range(-6, 6, 2):
            _log.warning(f"Adding 10^{i} to Kuu")
            K_ = Kuu + (10**i) * np.eye(Kuu.shape[0])
            try:
                Luu = np.linalg.cholesky(K_)
                break
            except np.linalg.LinAlgError:
                pass
    Luu_Kux = scipy.linalg.solve_triangular(Luu, Kux, lower=True)

    fold_eig = []
    for train_idx, test_idx in fold_idx(Kux.shape[1], n_splits):
        A = Luu_Kux[:, train_idx] @ oh_train_Y[train_idx, :]
        B = Luu_Kux[:, test_idx]
        # # Luu_ = np.linalg.cholesky(Kuu[train_idx, :][:, train_idx] + 9.737369588754647e-06 * np.eye(Kuu[train_idx].shape[0]))
        # # A = scipy.linalg.solve_triangular(Luu_, oh_train_Y[train_idx, :])
        # # B = scipy.linalg.solve_triangular(Luu_, Kux[train_idx, :][:, test_idx])
        # # eig = eigdecompose(np.eye(Kuu[train_idx].shape[0])) #A @ A.T)
        # A = oh_train_Y[train_idx, :]
        # B = Kux[train_idx, :][:, test_idx]
        # eig = eigdecompose(Kuu[train_idx, :][:, train_idx]
        #                    + 9.737369588754637e-06 * np.eye(Kuu[train_idx].shape[0]))
        eig = eigdecompose(Luu_Kux @ Luu_Kux.T)
        fold_eig.append((eig, A, B))

    min_eigval = min(*[e.vals.min() for e, _, _ in fold_eig])
    eigval_floor = max(0., -min_eigval)
    while min_eigval + eigval_floor <= 0.0:
        eigval_floor = np.nextafter(eigval_floor, np.inf)

    sigy_grid = np.exp(np.linspace(
        max(np.log(eigval_floor), -36), 5, n_grid_opt_points))
    # Make sure that none is lower than eigval_floor
    sigy_grid = np.clip(sigy_grid, eigval_floor, np.inf)

    folds = []
    for (train_idx, test_idx), (eig, A, B) in zip(fold_idx(Kux.shape[1], n_splits),
                                                  fold_eig):
        _oh_train_Y = oh_train_Y[train_idx, :]
        _test_Y = oh_train_Y[test_idx, :].argmax(-1)
        _, fold = accuracy_eig(eig, Kxt=B, oh_train_Y=_oh_train_Y, test_Y=_test_Y,
                               sigy_grid=sigy_grid, FY=A)
        folds.append(fold)

    grid_acc = np.mean(folds, axis=0)
    sigy = np.expand_dims(sigy_grid[grid_acc.argmax()], 0)
    Luu_Kut = scipy.linalg.solve_triangular(Luu, Kut, lower=True)

    return (sigy_grid, grid_acc), accuracy_eig(
        # Kuu, Kut,
        eigdecompose(Luu_Kux @ Luu_Kux.T, lower=True), Luu_Kut,
        oh_train_Y, test_Y, sigy, FY=Luu_Kux @ oh_train_Y)


def make_sigy_grid(min_eigval, n_grid_opt_points):
    eigval_floor = max(0., -min_eigval)
    while min_eigval + eigval_floor <= 0.0:
        eigval_floor = np.nextafter(eigval_floor, np.inf)

    sigy_grid = np.exp(np.linspace(
        max(np.log(eigval_floor), -36), 5, n_grid_opt_points))
    # Make sure that none is lower than eigval_floor
    sigy_grid = np.clip(sigy_grid, eigval_floor, np.inf)
    return sigy_grid

@experiment.capture
def find_cv_opt(Kuu_eig, Kux, Kut, oh_train_Y, n_splits, predict_cv_acc):
    Luu_Kux = (Kuu_eig.vecs * Kuu_eig.vals**-.5)  @ Kux

    fold_eig = []
    for train_idx, test_idx in fold_idx(Kux.shape[1], n_splits):
        A = Luu_Kux[:, train_idx] @ oh_train_Y[train_idx, :]
        B = Luu_Kux[:, test_idx]
        eig = eigdecompose(A @ A.T)
        fold_eig.append((eig, A, B))

    min_eigval = min(*[e.vals.min() for e, _, _ in fold_eig])
    sigy_grid = make_sigy_grid(min_eigval, predict_cv_acc['n_grid_opt_points'])

    folds = []
    for (train_idx, test_idx), (eig, A, B) in zip(fold_idx(Kux.shape[1], n_splits),
                                                  fold_eig):
        _oh_train_Y = oh_train_Y[train_idx, :]
        _test_Y = oh_train_Y[test_idx, :].argmax(-1)
        _, fold = accuracy_eig(eig, Kxt=B, oh_train_Y=_oh_train_Y, test_Y=_test_Y,
                               sigy_grid=sigy_grid, FY=A)
        folds.append(fold)
    grid_acc = np.mean(folds, axis=0)
    return sigy_grid, grid_acc


@experiment.capture
def do_one_N(Kuu, Kux, Kut, oh_train_Y, test_Y, n_splits, predict_cv_acc, _log):
    if n_splits == -1:
        raise NotImplementedError("leave-one-out")

    Kuu_eig = eigdecompose(Kuu)
    outer_sigy_grid = make_sigy_grid(Kuu_eig.vals.min(), 100)
    max_acc = -np.inf
    for outer_sigy in outer_sigy_grid:
        sigy_grid, grid_acc = find_cv_opt(EigenOut(vals=Kuu_eig.vals + outer_sigy, vecs=Kuu_eig.vecs),
                                          Kux, Kut, oh_train_Y, n_splits)
        sigy = np.expand_dims(sigy_grid[grid_acc.argmax()], 0)
        max_grid_acc = grid_acc.max()
        _log.info(f"For outer_sigy={outer_sigy}: sigy={sigy}, acc={max_grid_acc}, max_acc={max_acc}")
        if max_grid_acc > max_acc:
            max_acc = max_grid_acc
            max_outer_sigy = outer_sigy
    _log.info(f"Max acc: {max_acc}, outer_sigy={max_outer_sigy}")

    Kuu_eig = EigenOut(vals=Kuu_eig.vals + max_outer_sigy, vecs=Kuu_eig.vecs)
    Luu_Kux = (Kuu_eig.vecs * Kuu_eig.vals**-.5)  @ Kux
    Luu_Kut = (Kuu_eig.vecs * Kuu_eig.vals**-.5)  @ Kut

    return (sigy_grid, grid_acc), accuracy_eig(
        # Kuu, Kut,
        eigdecompose(Luu_Kux @ Luu_Kux.T, lower=True), Luu_Kut,
        oh_train_Y, test_Y, sigy, FY=Luu_Kux @ oh_train_Y)


@experiment.command
def test_rbf(_log, i_SU):
    import gpytorch
    torch.set_default_dtype(torch.float64)

    kern = gpytorch.kernels.RBFKernel()
    kern.lengthscale = 3.
    X = torch.linspace(0, 99, i_SU['N_train'])[:, None][torch.randperm(i_SU['N_train'])]
    Xt = torch.linspace(0.01, 29.01, i_SU['N_test'])[:, None]
    Z = Xt + 0.02

    K = kern(torch.cat((X, Xt), dim=0)).evaluate()
    Y = gpytorch.lazy.NonLazyTensor(K).zero_mean_mvn_samples(10).t().detach().numpy()

    with torch.no_grad():
        Kxx = K[:X.size(0), :X.size(0)].detach().numpy()
        Kxt = K[:X.size(0), X.size(0):].detach().numpy()
        oh_train_Y = Y[:X.size(0)]
        test_Y = Y[X.size(0):].argmax(1)
        Kzx = kern(Z, X).evaluate().detach().numpy()
        Kzt = kern(Z, Xt).evaluate().detach().numpy()
        Kzz = kern(Z).evaluate().detach().numpy()

    from experiments.predict_cv_acc import do_one_N as orig_do_one_N

    # data, accuracy = orig_do_one_N(Kxx, Kxt, oh_train_Y, test_Y, n_splits=4)
    data, accuracy = do_one_N_chol(Kzz, Kzx, Kzt, oh_train_Y, test_Y, n_splits=4)

    (sigy, acc) = map(np.squeeze, accuracy)
    _log.info(f"For RBF kernel, N={X.size(0)}, sigy={sigy}; accuracy={acc}, cv_accuracy={np.max(data[1])}")




@experiment.automain
def main(predict_cv_acc, _log):
    kernel_matrix_path = predict_cv_acc['kernel_matrix_path']
    multiply_var = predict_cv_acc['multiply_var']
    apply_relu = predict_cv_acc['apply_relu']
    n_splits = predict_cv_acc['n_splits']

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

            for N in [1280]: #reversed(data.columns):
                # Made a mistake, the same label are all contiguous in the training set.
                train_idx = slice(0, N_total, N_total//N)
                data.loc[layer, N], accuracy.loc[layer, N] = do_one_N_chol(
                    Kxx[train_idx, train_idx], Kxx[train_idx, :],
                    Kxt[train_idx, :], oh_train_Y, test_Y,
                    n_splits=n_splits)
                (sigy, acc) = map(np.squeeze, accuracy.loc[layer, N])
                _log.info(f"For layer={layer}, N={N}, sigy={sigy}; accuracy={acc}, cv_accuracy={np.max(data.loc[layer, N][1])}")
                pd.to_pickle(data, new_base_dir/"grid_acc.pkl.gz")
                pd.to_pickle(accuracy, new_base_dir/"accuracy.pkl.gz")
