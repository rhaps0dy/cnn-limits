import collections
import contextlib
import faulthandler
import itertools
import math
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sacred
import scipy.linalg
import scipy.optimize
import torch
from torch.utils.data import DataLoader, Subset

import cnn_limits.sacred_utils as SU

from experiments.rbfmyrtle_v3 import experiment as rbfmyrtle_v3_experiment
from experiments.rbfmyrtle_v3 import (dataset_targets, centered_one_hot,
                                      fold_idx, accuracy_eig, accuracy_chol, eigdecompose)

experiment = sacred.Experiment("classify_inducing_points", [SU.ingredient, rbfmyrtle_v3_experiment])
if __name__ == '__main__':
    SU.add_file_observer(experiment)

@experiment.capture
def do_one_N_noopt(Kuu, Kux, Kut, oh_train_Y, test_Y, n_splits, rbfmyrtle_v3, _log):
    n_grid_opt_points = rbfmyrtle_v3['n_grid_opt_points']
    device = rbfmyrtle_v3['device']
    assert n_splits > 1
    torch.cuda.empty_cache()
    Kuu = torch.from_numpy(Kuu).to(device=device)
    Kux = torch.from_numpy(Kux)
    Kut = torch.from_numpy(Kut)
    oh_train_Y = torch.from_numpy(oh_train_Y).to(device=device)
    test_Y = torch.from_numpy(test_Y).to(device=device)
    assert Kuu.dtype == torch.get_default_dtype()
    N, D = oh_train_Y.shape

    try:
        Luu = torch.cholesky(Kuu)
    except RuntimeError:
        for i in range(-6, 6, 2):
            _log.warning(f"Adding 10^{i} to Kuu")
            K_ = Kuu + (10**i) * torch.eye(Kuu.shape[-1], device=Kuu.device, dtype=Kuu.dtype)
            try:
                Luu = torch.cholesky(K_)
                break
            except RuntimeError:
                pass
            del K_
    del Kuu
    Luu = Luu.cpu()
    torch.cuda.empty_cache()
    Luu_Kux = torch.triangular_solve(Kux, Luu, upper=False).solution
    del Kux
    torch.cuda.empty_cache()

    Luu_Kux = Luu_Kux.cuda()
    M = Luu_Kux @ Luu_Kux.transpose(-1, -2)
    FY = Luu_Kux @ oh_train_Y
    del Luu_Kux
    torch.cuda.empty_cache()
    Luu_Kut = torch.triangular_solve(Kut, Luu, upper=False).solution.cuda()
    del Kut
    del Luu
    torch.cuda.empty_cache()
    input("Check how much memory CUDA has")
    return ([123], [123]), accuracy_chol(
        # Kuu, Kut,
        M[None, ...], Luu_Kut[None, ...],
        FY[None, ...], test_Y, torch.tensor([[1e-6]]).cuda())



@experiment.capture
def do_one_N_chol(Kuu, Kux, Kut, oh_train_Y, test_Y, n_splits, rbfmyrtle_v3, _log):
    n_grid_opt_points = rbfmyrtle_v3['n_grid_opt_points']
    device = rbfmyrtle_v3['device']
    assert n_splits > 1
    torch.cuda.empty_cache()
    Kuu = torch.from_numpy(Kuu).to(device=device)
    Kux = torch.from_numpy(Kut)
    Kut = torch.from_numpy(Kut)
    oh_train_Y = torch.from_numpy(oh_train_Y)
    test_Y = torch.from_numpy(test_Y).to(device=device)
    assert Kuu.dtype == torch.get_default_dtype()
    B, N, D = oh_train_Y.shape

    try:
        Luu = torch.cholesky(Kuu)
    except RuntimeError:
        for i in range(-6, 6, 2):
            _log.warning(f"Adding 10^{i} to Kuu")
            K_ = Kuu + (10**i) * torch.eye(Kuu.shape[-1], device=Kuu.device, dtype=Kuu.dtype)
            try:
                Luu = torch.cholesky(K_)
                break
            except RuntimeError:
                pass
            del K_
    del Kuu
    Luu = Luu.cpu()
    torch.cuda.empty_cache()
    Luu_Kux = torch.triangular_solve(Kux, Luu, upper=False).solution
    del Kux
    torch.cuda.empty_cache()
    Luu_Kut = torch.triangular_solve(Kut, Luu, upper=False).solution
    del Kut
    del Luu
    torch.cuda.empty_cache()
    input("Check how much memory CUDA has")

    fold_eig = []
    for train_idx, test_idx in fold_idx(N, n_splits):
        Luu_Kux_ = Luu_Kux[:, :, train_idx]
        A = Luu_Kux_ @ oh_train_Y[:, train_idx, :]
        B = Luu_Kux[:, :, test_idx]
        M = Luu_Kux_ @ Luu_Kux_.transpose(-1, -2)
        fold_eig.append((M.cpu(), A.cpu(), B.cpu()))
    del A
    del B
    del M

    Luu_Kux = Luu_Kux.cpu()
    Luu_Kut = Luu_Kut.cpu()
    del Luu_Kut
    del Luu_Kux
    torch.cuda.empty_cache()

    for i, (M, _, _) in fold_eig:
        fold_eig[i] = eigdecompose(M.cuda()).cpu()
        torch.cuda.empty_cache()
    del M

    all_eigvals = torch.stack([e.eigenvalues for e in fold_eig], dim=1)
    min_eigvals = all_eigvals.min(2).values.min(1).values

    all_sigy_grid = torch.zeros((B, n_grid_opt_points), dtype=torch.get_default_dtype())
    for i, min_eigval in enumerate(min_eigvals):
        min_eigval = min_eigval.item()
        eigval_floor = max(0., -min_eigval)
        while min_eigval + eigval_floor <= 0.0:
            eigval_floor = np.nextafter(eigval_floor, np.inf)

        sigy_grid = np.exp(np.linspace(
            max(np.log(eigval_floor), -36), 5, n_grid_opt_points))
        # Make sure that none is lower than eigval_floor
        sigy_grid = np.clip(sigy_grid, eigval_floor, np.inf)
        all_sigy_grid[i] = torch.from_numpy(sigy_grid)

    if N <= 10:
        # Skip cross_validation
        all_sigy_grid[:, :] = all_sigy_grid[:, 0, None]

    folds = None
    for (train_idx, test_idx), (eig, A, B) in zip(fold_idx(N, n_splits),
                                                  fold_eig):
        _oh_train_Y = oh_train_Y[:, train_idx, :]
        _test_Y = oh_train_Y[:, test_idx, :].argmax(-1)
        torch.cuda.empty_cache()
        _, fold = accuracy_eig(eig.cuda(), B.cuda(), A.cuda(), _test_Y.cuda(), all_sigy_grid.to(device=device))
        if folds is None:
            folds = fold
        else:
            folds += fold
    folds /= n_splits
    grid_acc = folds

    sigy = torch.zeros([B, 1], dtype=torch.get_default_dtype())
    assert len(grid_acc.shape) == 2
    for i in range(len(grid_acc)):
        sigy[i] = all_sigy_grid[i][grid_acc[i].argmax()].item()

    return (sigy_grid, grid_acc), accuracy_chol(
        # Kuu, Kut,
        Luu_Kux @ Luu_Kux.transpose(-1, -2), Luu_Kut,
        Luu_Kux @ oh_train_Y, test_Y, sigy)


@experiment.automain
def classify(kernel_matrix_path, _log, i_SU):
    N_inducing = 15000

    assert i_SU["dataset_treatment"] == "no_treatment"
    train_set, test_set = SU.load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    oh_train_Y = centered_one_hot(train_Y).astype(np.float64)
    test_Y = dataset_targets(test_set)
    print(oh_train_Y.shape, test_Y.shape)

    with h5py.File(kernel_matrix_path, 'r') as f:
        print(f["Kuu"].shape, f["Kux"].shape, f["Kut"].shape)
        mask_Kux = ~(np.isnan(f["Kux"][0, ...]).any(-1))
        mask_Kut = ~(np.isnan(f["Kut"][0, ...]).any(-1))
        mask = mask_Kux & mask_Kut

        for i in range(f["Kuu"].shape[0]):
            Kuu = f["Kuu"][i, :, :][mask, :][:, mask].astype(np.float64)
            Kuu_mask = np.triu(np.ones(Kuu.shape, dtype=bool))
            Kuu[Kuu_mask] = Kuu.T[Kuu_mask]
            Kux = f["Kux"][i, :, :].astype(np.float64)[mask, :]
            Kut = f["Kut"][i, :, :].astype(np.float64)[mask, :]

            data, accuracy = do_one_N_noopt(Kuu[:N_inducing, :N_inducing], Kux[:N_inducing, :],
                                            Kut[:N_inducing, :], oh_train_Y, test_Y, n_splits=4)
            (sigy, acc) = map(np.squeeze, accuracy)
            _log.info(
                f"For i={i}, N_inducing={Kuu.shape[0]}, sigy={sigy}; accuracy={acc}, cv_accuracy={np.max(data[1])}")
