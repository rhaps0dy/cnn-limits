import numpy as np
from torch.utils.data import DataLoader, Subset
from typing import Sequence, Tuple
import torch


def dataset_full(dset) -> Tuple[torch.Tensor, torch.Tensor]:
    return next(iter(DataLoader(dset, batch_size=len(dset))))


def dataset_targets(dset) -> np.ndarray:
    _, y = dataset_full(dset)
    return y.numpy()


def centered_one_hot(y: np.ndarray, N_classes=10) -> np.ndarray:
    return np.eye(N_classes)[y] - 1/N_classes


def fold_idx(n_data: int, n_splits: int, device: str) -> Sequence[Tuple[torch.Tensor, slice]]:
    """Iterator for the indices for each of `n_splits` folds. Each fold contains
    contiguous data points as the test set."""
    assert n_data % n_splits == 0
    split_size = n_data//n_splits
    for i in range(n_splits):
        test_idx = slice(i*split_size, (i+1)*split_size)
        idx = np.arange(n_data)
        train_idx = np.delete(idx, idx[test_idx])
        yield torch.from_numpy(train_idx).to(device=device), test_idx


def balanced_data_indices(train_Y, N_classes=10):
    """Returns permutation indices of `train_Y` such that its value is
    [0, 1, 2, 3, ..., N_classes-1, 0, 1, ..., N_classes-1, ...]

    This function assumes that `train_Y` has been ordered using
    `class_balanced_train_idx` from `sacred_utils.py`.
    """
    N_total = len(train_Y)
    this_N_permutation = []
    for i in range(10):
        perm = np.random.permutation(N_total//N_classes)
        this_N_permutation.append(perm+(i*N_total//N_classes))
    this_N_permutation = np.stack(this_N_permutation, 0).T.copy().reshape(-1)
    assert np.all(sorted(this_N_permutation) == np.arange(N_total))
    for i in range(N_total//N_classes):
        assert np.array_equal(train_Y[this_N_permutation[i*N_classes:(i+1)*N_classes]], np.arange(N_classes))
    return this_N_permutation
