import collections
import contextlib
import os
import pickle
import warnings
import git
import json
from pathlib import Path

import numpy as np
import sacred
import torch
import jug
import torchvision
from torch.utils.data import DataLoader, Subset, TensorDataset

import gpytorch

__all__ = ["load_sorted_dataset", "interlaced_argsort",
           "base_dir", "new_file"]

ingredient = sacred.Ingredient("i_SU")


@ingredient.config
def config():
    # GPytorch
    num_likelihood_samples = 20
    # Pytorch
    default_dtype = "float64"

    log_dir="/scratches/huygens/ag919/jug/test"

    # Dataset loading
    dataset_base_path = "/scratch/ag919/datasets/"
    dataset_name = "CIFAR10"
    ZCA_transform = False
    ZCA_bias = 1e-5

    # Whether to take the "test" set from the end of the training set
    test_is_validation = True
    dataset_treatment = "train_random_balanced"
    train_idx_path = None


@ingredient.pre_run_hook
def log_dir_hook(_run):
    "Creates log_dir and makes sure configuration and code matches"
    _run.debug = True  # let jug handle BarrierError correctly

    log_dir = base_dir()
    config = _run.config

    repo_path = Path(__file__).absolute().parent.parent
    repo = git.Repo(repo_path)
    diff = repo.git.diff()
    head = repo.head.commit.hexsha

    if log_dir.is_dir():
        with open(log_dir/"config.json", "r") as f:
            existing_config = json.load(f)
        if config != existing_config:
            raise ValueError(f"Not matching {str(log_dir/'config.json')}")

        def file_equal(path, value):
            with open(path, "r") as f:
                existing = f.read()
            if existing != value:
                raise ValueError(f"Not matching {str(path)}")
        file_equal(log_dir/"git.head", head)
    else:
        log_dir.mkdir(parents=True)
        with open(log_dir/"config.json", "w") as f:
            json.dump(config, f, sort_keys=True, indent=2)
            f.write("\n")
        with open(log_dir/"git.head", "w") as f:
            f.write(head)
    with open(log_dir/"git.diff", "w") as f:
        f.write(diff)
    jug.set_jugdir(str(log_dir/"jugdir/"))


@ingredient.pre_run_hook
def gpytorch_pre_run_hook(num_likelihood_samples, default_dtype, _seed):
    gpytorch.settings.num_likelihood_samples._set_value(num_likelihood_samples)
    # disable CG, it makes the eigenvalues of test Gaussian negative :(
    # It's also really slow for batched matrices for some reason
    gpytorch.settings.max_cholesky_size._set_value(1000000)
    torch.set_default_dtype(getattr(torch, default_dtype))
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)


@ingredient.post_run_hook
def print_experiment(_run):
    print(f"This was run {_run._id}")


# File handling
@ingredient.capture
def base_dir(log_dir):
    return Path(log_dir)


@contextlib.contextmanager
def new_file(relative_path, mode="wb"):
    full_path = os.path.join(base_dir(), relative_path)
    with open(full_path, mode) as f:
        yield f


# Datasets and sorted data sets
def load_dataset(dataset_name, dataset_base_path, additional_transforms=[]):
    dataset_base_path = Path(dataset_base_path)
    if dataset_name == "CIFAR10_ZCA_wrong":
        data = np.load(dataset_base_path/"CIFAR10_ZCA_wrong.npz")
        train = TensorDataset(*(torch.from_numpy(data[a]) for a in ("X", "y")))
        test = TensorDataset(*(torch.from_numpy(data[a]) for a in ("Xt", "yt")))
        return train, test

    elif dataset_name == "CIFAR10_ZCA_shankar_wrong":
        data = np.load(dataset_base_path/"cifar_10_zca_augmented_extra_zca_augment_en9fKkGMMg.npz")
        train = TensorDataset(
            torch.from_numpy(data["X_train"]).permute(0, 3, 1, 2),
            torch.from_numpy(data["y_train"]))
        test = TensorDataset(
            torch.from_numpy(data["X_test"]).permute(0, 3, 1, 2),
            torch.from_numpy(data["y_test"]))
        return train, test

    if dataset_name == "CIFAR10":
        dataset_base_path = dataset_base_path/dataset_name
    if len(additional_transforms) == 0:
        trans = torchvision.transforms.ToTensor()
    else:
        trans = torchvision.transforms.Compose([
            *additional_transforms,
            torchvision.transforms.ToTensor(),
        ])

    _dset = getattr(torchvision.datasets, dataset_name)
    train = _dset(dataset_base_path, train=True, download=True, transform=trans)
    test = _dset(dataset_base_path, train=False, transform=trans)
    return train, test


def whole_dset(dset):
    return next(iter(DataLoader(dset, batch_size=len(dset))))


@ingredient.capture
def _apply_zeromean(X, rgb_mean, do_zero_mean):
    if not do_zero_mean:
        return X
    rgb_mean = rgb_mean.to(X.device)
    X -= rgb_mean
    return X

def _apply_ZCA(X, W):
    orig_dtype = X.dtype
    shape = X.size()
    W = W.to(X.device)
    X = X.reshape((X.size(0), -1)).to(W.dtype) @ W
    return X.reshape(shape).to(orig_dtype)


def _norm_shankar(X):
    "Perform normalization like https://github.com/modestyachts/neural_kernels_code/blob/ef09d4441cfc901d7a845ffac88ddd4754d4602e/utils.py#L280"
    # Per-instance zero-mean
    X = X - X.mean((1, 2, 3), keepdims=True)
    # Normalize each instance
    sqnorm = X.pow(2).sum((1, 2, 3), keepdims=True)
    return X * sqnorm.add(1e-16).pow(-1/2)


@ingredient.capture
def do_transforms(train_set, test_set, ZCA_transform: bool, ZCA_bias: float):
    X, y = whole_dset(train_set)
    device = ('cuda' if torch.cuda.device_count() else 'cpu')
    X = X.to(device=device, dtype=torch.float64)
    if ZCA_transform:
        X = _norm_shankar(X)
        # The output of GPU and CPU SVD is different, so we do it in CPU and
        # then bring it back to `device`
        _, S, V = torch.svd(X.reshape((len(train_set), -1)).cpu())
        S = S.to(device)
        V = V.to(device)

        # X^T @ X  ==  (V * S^2) @ V.T
        S = S.pow(2).div(len(train_set)).add(ZCA_bias).sqrt()
        W = (V / S) @ V.t()
        X = _apply_ZCA(X, W)

    Xt, yt = whole_dset(test_set)
    Xt = Xt.to(device=device, dtype=torch.float64)
    if ZCA_transform:
        Xt = _norm_shankar(Xt)
        Xt = _apply_ZCA(Xt, W)

        W = W.cpu()
    else:
        W = None
    return TensorDataset(X.cpu(), y), TensorDataset(Xt.cpu(), yt), W


def class_balanced_train_idx(train_set, N_train, forbidden_indices=None):
    train_y = torch.tensor(train_set.targets)
    argsort_y = torch.argsort(train_y)
    N_classes = len(train_set.classes)

    if forbidden_indices is not None:
        assert isinstance(forbidden_indices, set)
        new_argsort_y = filter(lambda x: x.item() not in forbidden_indices,
                               argsort_y)
        new_argsort_y = torch.tensor(list(new_argsort_y), dtype=torch.int64)

        assert len(set(new_argsort_y.numpy()).intersection(forbidden_indices)) == 0
        assert len(new_argsort_y) + len(forbidden_indices) == len(argsort_y)
        argsort_y = new_argsort_y

    starting_for_class = [None] * N_classes
    for i, idx in enumerate(argsort_y):
        if starting_for_class[train_y[idx]] is None:
            starting_for_class[train_y[idx]] = i

    min_gap = len(train_set)
    for prev, nxt in zip(starting_for_class, starting_for_class[1:]):
        min_gap = min(min_gap, nxt-prev)
    assert min_gap >= N_train//N_classes, "Cannot draw balanced data set"
    train_idx_oneclass = torch.randperm(min_gap)[:N_train//N_classes]

    train_idx = torch.cat([argsort_y[train_idx_oneclass + start]
                            for start in starting_for_class])
    # Check that it is balanced
    count = collections.Counter(a.item() for a in train_y[train_idx])
    for label in range(N_classes):
        assert count[label] == N_train // N_classes, "new set not balanced"
    assert len(set(train_idx)) == N_train, "repeated indices"
    return train_idx.numpy()
