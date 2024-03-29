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
faulthandler.enable()

experiment = sacred.Experiment("rbfmyrtle_v3", [SU.ingredient])
if __name__ == '__main__':
    SU.add_file_observer(experiment)


def dataset_targets(dset):
    _, y = next(iter(DataLoader(dset, batch_size=len(dset))))
    return np.asarray(y.numpy())


def centered_one_hot(y, N_classes=10):
    return np.eye(N_classes)[y] - 1/N_classes


@experiment.config
def config():
    kernel_matrix_path = "/scratch/ag919/logs/save_new/166"

    N_classes = 10
    n_grid_opt_points = 1000
    n_splits = 4
    device = 'cuda'


def eigdecompose(Kxx, lower=True):
    return torch.symeig(Kxx, eigenvectors=True, upper=not lower)


def accuracy_chol(Kxx, Kxt, oh_train_Y, test_Y, sigy, FY=None, lower=True):
    assert FY is None
    B, N, D = oh_train_Y.shape
    T = Kxt.shape[-1]
    assert sigy.shape == (B, 1)

    if B == 1:
        Kxx = Kxx.squeeze(0)
        sigy_ = sigy.item()
    else:
        sigy_ = sigy.view((-1, 1, 1))
    L = torch.cholesky(Kxx + sigy_*torch.eye(N, dtype=Kxx.dtype, device=Kxx.device), upper=False)
    if B == 1:
        L = L.unsqueeze(0)
    assert L.shape == (B, N, N)

    LY = torch.triangular_solve(oh_train_Y, L, upper=False).solution
    assert LY.shape == (B, N, D)
    Lxt = torch.triangular_solve(Kxt, L, upper=False).solution
    assert Lxt.shape == (B, N, T)
    pred_F = Lxt.transpose(-2, -1) @ LY
    pred_Y = pred_F.argmax(-1)
    assert pred_Y.shape == (B, T)
    accuracies = torch.zeros((B, 1))
    accuracies[:, 0] = (pred_Y == test_Y).to(dtype=torch.get_default_dtype()).mean(-1)
    return sigy, accuracies


@experiment.capture
def accuracy_eig(Kxx, Kxt, oh_train_Y, test_Y, sigy_grid, _log, FY=None,
                 lower=True):
    """Evaluates accuracy for a path using the eigendecomposition.
    Returns the accuracy at each position
    """
    B, N, D = oh_train_Y.shape
    T = Kxt.shape[-1]
    if FY is not None:
        # FY = torch.from_numpy(FY).cuda()
        pass
    else:
        FY = oh_train_Y
    if isinstance(Kxx, torch.Tensor):
        eig = eigdecompose(Kxx, lower=lower)
    else:
        eig = Kxx
    del Kxx
    for a in (eig.eigenvalues, eig.eigenvectors, Kxt, oh_train_Y, sigy_grid):
        assert a.dtype == torch.get_default_dtype()

    assert eig.eigenvalues.shape == (B, N)
    assert eig.eigenvectors.shape == (B, N, N)
    assert Kxt.shape == (B, N, T)
    assert test_Y.shape[-1] == T and len(test_Y.shape) <= 2
    assert oh_train_Y.shape == (B, N, D)
    assert sigy_grid.shape[0] == B and len(sigy_grid.shape) == 2

    _log.debug("Calculating alpha and beta")
    alpha = eig.eigenvectors.transpose(-2, -1) @ FY
    try:
        gamma = Kxt.transpose(-2, -1) @ eig.eigenvectors
        big_matrices = False
    except RuntimeError:
        _log.warn(f"Not enough memory in GPU to multiply Kxt @ eigenvectors ({Kxt.transpose(-2, -1).shape, eig.eigenvectors.shape}). Calculating on CPU instead")
        Kxt_ = Kxt.transpose(-2, -1).cpu()
        evecs_ = eig.eigenvectors.cpu()
        gamma_ = Kxt_ @ evecs_
        gamma = eig.eigenvectors[:, :T, :]
        del Kxt_
        del evecs_
        big_matrices = True

    accuracies = torch.zeros((sigy_grid.shape[1], B))
    for i in range(sigy_grid.shape[1]):
        eigvals = (eig.eigenvalues + sigy_grid[:, i:i+1]).unsqueeze(1)
        if big_matrices:
            gamma.copy_(gamma_)
            gamma /= eigvals
            pred_F = gamma @ alpha
        else:
            pred_F = (gamma / eigvals) @ alpha
        pred_Y = pred_F.argmax(-1)
        assert pred_Y.shape == (B, T)
        # if N == 10240:
        #     print("Saving to file /scratch2/ag919/predictions.pt")
        #     torch.save((pred_Y, test_Y), "/scratch2/ag919/predictions.pt")
        accuracies[i, :] = (pred_Y == test_Y).to(dtype=torch.get_default_dtype()).mean(-1)

    return sigy_grid, accuracies.t()


@experiment.capture
def fold_idx(n_samples, n_splits, device):
    for start in range(n_splits):
        test_idx = slice(start, n_samples, n_splits)
        idx = np.arange(n_samples)
        train_idx = np.delete(idx, idx[test_idx])
        yield torch.from_numpy(train_idx).cuda(), test_idx


@experiment.capture
def do_one_N(Kxx, Kxt, oh_train_Y, test_Y, n_grid_opt_points, n_splits, device,
             FY=None, lower=True):
    assert n_splits > 1
    torch.cuda.empty_cache()
    Kxx = torch.from_numpy(Kxx).cuda()
    Kxt = torch.from_numpy(Kxt).cuda()
    oh_train_Y = torch.from_numpy(oh_train_Y).cuda()
    test_Y = torch.from_numpy(test_Y).cuda()
    assert Kxx.dtype == torch.get_default_dtype()

    fold_eig = [eigdecompose(Kxx[:, train_idx, :][:, :, train_idx], lower=lower)
                for train_idx, _ in fold_idx(Kxx.shape[1], n_splits)]

    all_eigvals = torch.stack([e.eigenvalues for e in fold_eig], dim=1)
    min_eigvals = all_eigvals.min(2).values.min(1).values

    all_sigy_grid = torch.zeros((Kxx.shape[0], n_grid_opt_points), dtype=torch.get_default_dtype())
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

    if Kxx.shape[1] <= 10:
        # Skip cross_validation
        all_sigy_grid[:, :] = all_sigy_grid[:, 0, None]

    folds = None
    for (train_idx, test_idx), eig in zip(fold_idx(Kxx.shape[1], n_splits),
                                          fold_eig):
        _Kxt = Kxx[:, train_idx, :][:, :, test_idx]
        _oh_train_Y = oh_train_Y[:, train_idx, :]
        _test_Y = oh_train_Y[:, test_idx, :].argmax(-1)
        _, fold = accuracy_eig(eig, _Kxt, _oh_train_Y, _test_Y, all_sigy_grid.cuda())
        if folds is None:
            folds = fold
        else:
            folds += fold
    folds /= n_splits
    grid_acc = folds

    sigy = torch.zeros([Kxx.shape[0], 1], dtype=torch.get_default_dtype())
    assert len(grid_acc.shape) == 2
    for i in range(len(grid_acc)):
        sigy[i] = all_sigy_grid[i][grid_acc[i].argmax()].item()
    # sigy[...] = 5.442873064068619e-06
    # print("Went from {all_sigy_grid[0][grid_acc[0].argmax()].item()} to {sigy[0].item()}")

    return (all_sigy_grid, grid_acc), accuracy_eig(
        Kxx,
        Kxt, oh_train_Y, test_Y, sigy.cuda(), FY=FY,
        lower=lower)


@experiment.automain
def main_no_eig(kernel_matrix_path, _log, n_splits, i_SU):
    kernel_matrix_path = Path(kernel_matrix_path)
    train_set, test_set = SU.load_sorted_dataset(
        dataset_treatment="load_train_idx",
        train_idx_path=kernel_matrix_path)

    dtype = getattr(np, i_SU["default_dtype"])

    train_Y = dataset_targets(train_set)
    oh_train_Y = centered_one_hot(train_Y).astype(dtype)
    test_Y = dataset_targets(test_set)

    with open(kernel_matrix_path/"config.json", "r") as src, open(SU.base_dir()/"old_config.json", "w") as dst:
        dst.write(src.read())

    with h5py.File(kernel_matrix_path/"kernels.h5", "r") as f:
        N_layers, N_total, _ = f['Kxx'].shape

        all_N = list(itertools.takewhile(
            lambda a: a <= N_total,
            (2**i * 10 for i in itertools.count(0))))
        data = pd.DataFrame(index=range(N_layers), columns=all_N)
        data_max = pd.DataFrame(index=range(N_layers), columns=all_N)
        accuracy = pd.DataFrame(index=range(N_layers), columns=all_N)

        new_base_dir = SU.base_dir()/f"n_splits_{n_splits}"
        os.makedirs(new_base_dir, exist_ok=True)

        for layer in range(36, 52):  #reversed(data.index):
            Kxx = f['Kxx'][layer].astype(dtype)
            # mask = np.isnan(Kxx)
            mask = np.tril(np.ones(Kxx.shape, dtype=bool))
            Kxx[mask] = Kxx.T[mask]
            assert np.allclose(Kxx, Kxx.T)
            # Kxx = (Kxx + Kxx.T)/2

            # Kxt = f['Kxt'][layer].astype(dtype)
            # Fix my mistake of changing the kernel code in between the validation and test set calculation
            # if layer in (10, 11, 12, 13):
            #     import gpytorch
            #     from cnn_limits.layers import covariance_tensor
            #     filter_numel = 32**2
            #     kern = gpytorch.kernels.MaternKernel(nu=3/2, lengthscale=2)
            #     kern.lengthscale = 10**np.linspace(-1.5, 4.5, 25)[layer-1]
            #     norm = covariance_tensor(32, 32, kern).sum()
            #     Kxx *= filter_numel / norm
            #     # Kxt *= norm / filter_numel
            #     # import pdb; pdb.set_trace()

            for N in reversed(data.columns):
                # Permute within classes
                this_N_permutation = np.concatenate([
                    np.random.permutation(N_total//10) + i*N_total//10
                    for i in range(10)])
                assert np.all(sorted(this_N_permutation) == np.arange(N_total))

                data.loc[layer, N], accuracy.loc[layer, N] = [], []
                # For this many data sets
                Kxx_ = np.zeros((N_total//N, N, N), dtype=dtype)
                Kxt_ = np.zeros((N_total//N, N, 1), dtype=dtype)
                Y_ = np.zeros((N_total//N, N, oh_train_Y.shape[-1]), dtype=dtype)

                for i in range(N_total//N):
                    train_idx = this_N_permutation[i::N_total//N]
                    assert len(train_idx) == N
                    Kxx_[i] = Kxx[train_idx, :][:, train_idx]
                    # Kxt_[i] = Kxt[train_idx]
                    Y_[i] = oh_train_Y[train_idx]

                _d, _a = do_one_N(Kxx_, Kxt_, Y_, test_Y[:Kxt_.shape[-1]], n_splits=(5 if N==10 else n_splits), FY=None, lower=True)
                _d = (_d[0].cpu().numpy(), _d[1].cpu().numpy())
                _a = (_a[0].cpu().numpy().squeeze(-1), _a[1].cpu().numpy().squeeze(-1))
                _d_max = (None, _d[1].max(-1))
                data.loc[layer, N] = _d
                accuracy.loc[layer, N] = _a
                data_max.loc[layer, N] = _d_max
                for i in range(N_total//N):
                    _log.info(
                        f"For layer={layer}, N={N}, i={i}, sigy={_a[0][i]}; accuracy={_a[1][i]}, cv_accuracy={_d_max[1][i]}")

                pd.to_pickle(data, new_base_dir/"grid_acc.pkl.gz")
                pd.to_pickle(data_max, new_base_dir/"cv_accuracy.pkl.gz")
                pd.to_pickle(accuracy, new_base_dir/"accuracy.pkl.gz")
