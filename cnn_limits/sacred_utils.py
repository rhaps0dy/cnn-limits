import contextlib
import os
import pickle
from pathlib import Path

import sacred
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, TensorDataset

import gpytorch

__all__ = ["load_sorted_dataset", "interlaced_argsort",
           "base_dir", "new_file"]

ingredient = sacred.Ingredient("i_SU")


@ingredient.config
def config():
    # GPytorch
    num_likelihood_samples = 10
    default_dtype = "float64"

    # Dataset loading
    dataset_base_path = "/scratch/ag919/datasets/"
    dataset_name = "CIFAR10"
    sorted_dataset_path = os.path.join(dataset_base_path, "interlaced_argsort")
    N_train = None
    N_test = None
    ZCA_transform = False
    ZCA_bias = 1e-5

    # Whether to take the "test" set from the end of the training set
    test_is_validation = True


# GPytorch
@ingredient.pre_run_hook
def gpytorch_pre_run_hook(num_likelihood_samples, default_dtype):
    gpytorch.settings.num_likelihood_samples._set_value(num_likelihood_samples)
    # disable CG, it makes the eigenvalues of test Gaussian negative :(
    gpytorch.settings.max_cholesky_size._set_value(1000000)
    torch.set_default_dtype(getattr(torch, default_dtype))


# File observer creation
def add_file_observer(experiment, default_dir="/scratch/ag919/logs"):
    log_dir = (os.environ['LOG_DIR'] if 'LOG_DIR' in os.environ
               else os.path.join(default_dir, experiment.path))
    experiment.observers.append(
        sacred.observers.FileStorageObserver(log_dir))


# File handling
@ingredient.capture
def base_dir(_run, _log):
    try:
        return Path(_run.observers[0].dir)
    except IndexError:
        _log.warning("This run has no associated directory, using `/tmp`")
        return Path("/tmp")


@contextlib.contextmanager
def new_file(relative_path, mode="wb"):
    full_path = os.path.join(base_dir(), relative_path)
    with open(full_path, mode) as f:
        yield f


# Datasets and sorted data sets
@ingredient.capture
def load_dataset(dataset_name, dataset_base_path, additional_transforms=[]):
    if dataset_name == "CIFAR10":
        dataset_base_path = os.path.join(dataset_base_path, dataset_name)
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


def interlaced_argsort(dset):
    y = torch.tensor(dset.targets)
    starting_for_class = [None] * len(dset.classes)
    argsort_y = torch.argsort(y)
    for i, idx in enumerate(argsort_y):
        if starting_for_class[y[idx]] is None:
            starting_for_class[y[idx]] = i

    for i in range(len(starting_for_class)-1):
        assert starting_for_class[i] < starting_for_class[i+1]
    assert starting_for_class[0] == 0

    init_starting = [a for a in starting_for_class] + [len(dset)]

    res = []
    while len(res) < len(dset):
        for i in range(len(starting_for_class)):
            if starting_for_class[i] < init_starting[i+1]:
                res.append(argsort_y[starting_for_class[i]].item())
                starting_for_class[i] += 1
    assert len(set(res)) == len(dset)
    assert len(res) == len(dset)
    return res


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


def do_transforms(train_set, test_set, ZCA: bool, ZCA_bias: float):
    X, y = whole_dset(train_set)
    device = ('cuda' if torch.cuda.device_count() else 'cpu')
    X = X.to(device=device, dtype=torch.float64)
    if ZCA:
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
    if ZCA:
        Xt = _norm_shankar(Xt)
        Xt = _apply_ZCA(Xt, W)
    return TensorDataset(X.cpu(), y), TensorDataset(Xt.cpu(), yt), W.cpu()

@ingredient.capture
def load_sorted_dataset(sorted_dataset_path, N_train, N_test, ZCA_transform, test_is_validation, ZCA_bias, _run):
    with _run.open_resource(os.path.join(sorted_dataset_path, "train.pkl"), "rb") as f:
        train_idx = pickle.load(f)
    with _run.open_resource(os.path.join(sorted_dataset_path, "test.pkl"), "rb") as f:
        test_idx = pickle.load(f)
    train_set, test_set = load_dataset()

    if test_is_validation:
        assert N_train+N_test <= len(train_set), "Train+validation sets too large"
        test_set = Subset(train_set, train_idx[-N_test-1:-1])
    else:
        test_set = Subset(test_set, test_idx[:N_test])

    train_set = Subset(train_set, train_idx[:N_train])
    train, test, _ = do_transforms(train_set, test_set, ZCA=ZCA_transform, ZCA_bias=ZCA_bias)
    return train, test
