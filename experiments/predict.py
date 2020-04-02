import sacred
import jax.numpy as np
import jax.scipy as sp
import numpy as onp
import os
import cnn_limits
import re
import jax
import pickle
from torch.utils.data import Subset, DataLoader

experiment = sacred.Experiment("predict")
cnn_limits.sacred_utils.add_file_observer(experiment, __name__)
load_dataset = experiment.capture(cnn_limits.load_dataset)

@experiment.capture
def load_sorted_dataset(sorted_dataset_path, N_train, N_test):
    with experiment.open_resource(os.path.join(sorted_dataset_path, "train.pkl"), "rb") as f:
        train_idx = pickle.load(f)
    with experiment.open_resource(os.path.join(sorted_dataset_path, "test.pkl"), "rb") as f:
        test_idx = pickle.load(f)
    train_set, test_set = load_dataset()
    return (Subset(train_set, train_idx[:N_train]),
            Subset(test_set, test_idx[:N_test]))

@experiment.config
def config():
    dataset_base_path = "/scratch/ag919/datasets/"
    sorted_dataset_path = os.path.join(dataset_base_path, "interlaced_argsort")
    dataset_name = "CIFAR10"

    kernel_matrix_path = "/scratch/ag919/logs/save_new/166"

    N_train = 50000
    N_test = 10000



def get_last_full(present):
    for i, r in enumerate(present):
        if not r:
            return i
    return len(present)


_collect_exp = re.compile(r"^0*([0-9]+)_0*([0-9]+)\.npy$")
@experiment.capture
def collect_kernel_matrix(path, _log, symmetric=False):
    fnames = os.listdir(path)
    fpaths = list(os.path.join(path, f) for f in fnames)
    indices = (tuple(map(int, _collect_exp.match(name).groups()))
               for name in fnames)
    idx, jdx = zip(*indices)
    # B=batch size.
    n_layers, B, _ = onp.load(fpaths[0]).shape

    present_squares = onp.zeros((max(idx)//B + 1, max(jdx)//B + 1), bool)
    for i, j in zip(idx, jdx):
        present_squares[i//B, j//B] = True
        if symmetric:
            present_squares[j//B, i//B] = True

    # Find suitable all-present region
    full_cols = present_squares.all(axis=0)
    N_full_col = get_last_full(full_cols) * B
    _log.debug(f"N full col for {path}: {N_full_col}")

    full_rows = present_squares[:, :N_full_col].all(axis=1)
    N_full_row = get_last_full(full_rows) * B
    _log.debug(f"N full row for {path}: {N_full_row}")

    # n_layers last for more contiguous memory accesses
    K = onp.zeros((N_full_row, N_full_col, n_layers), dtype=onp.float32)
    for fpath, i, j in zip(fpaths, idx, jdx):
        if i+B < N_full_row and j+B < N_full_col:
            K[i:i+B, j:j+B, :] = np.load(fpath).transpose((1, 2, 0))
            if symmetric and i!=j:
                K[j:j+B, i:i+B, :] = K[i:i+B, j:j+B, :].transpose((1, 0, 2))
    return np.asarray(K.transpose((2, 0, 1)), dtype=np.float64)


def dataset_targets(dset):
    _, y = next(iter(DataLoader(dset, batch_size=len(dset))))
    return np.asarray(y.numpy())


def centered_one_hot(y, N=10):
    oh = y[:, None] == np.arange(N)
    return (oh.astype(np.float64)*N - 1) / (N-1)


def _predict_gp(Kxx, Kxt, y):
    # TODO Make the Eye a jitter constant
    K = Kxx + np.eye(Kxx.shape[0], dtype=Kxx.dtype)
    Ky = sp.linalg.solve(K, centered_one_hot(y), sym_pos=True)
    f = Kxt.transpose((1, 0)) @ Ky
    return np.argmax(f, axis=1)
predict_gp = jax.jit(jax.vmap(_predict_gp, (0, 0, None)))

def accuracy(y, pred):
    return (y == pred).astype(np.float64).mean(1)


# TODO Find optimal likelihood for fourier features
# TODO check the non-pooled one, compared with google brain paper

@experiment.automain
def main(kernel_matrix_path, _log):
    if 'JAX_ENABLE_X64' not in os.environ:
        raise RuntimeError("Need environment variable JAX_ENABLE_X64=1")

    train_set, test_set = load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    test_Y = dataset_targets(test_set)

    Kxx = collect_kernel_matrix(os.path.join(kernel_matrix_path, "Kxx"), symmetric=True)
    Kxt = collect_kernel_matrix(os.path.join(kernel_matrix_path, "Kxt"))
    effective_N = min(Kxx.shape[1], Kxt.shape[1])

    test_Y = test_Y[:Kxt.shape[2]]

    accuracies = {}
    N = 100
    while N <= effective_N:
        acc = accuracy(test_Y, predict_gp(Kxx[:, :N, :N], Kxt[:, :N, :], train_Y[:N]))
        _log.info(f"Accuracies with {N}: {acc}")
        accuracies[N] = acc
        N += 100
    with open("out.pkl", "wb") as f:
        pickle.dump(accuracies, f)


