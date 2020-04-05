import sacred
import numpy as np
import scipy as sp
import scipy.linalg
import collections
import os
from pathlib import Path
import cnn_limits
import re
import pickle
import tqdm
from torch.utils.data import Subset, DataLoader

experiment = sacred.Experiment("predict")
cnn_limits.sacred_utils.add_file_observer(experiment, __name__)
load_dataset = experiment.capture(cnn_limits.load_dataset)
base_dir = experiment.capture(cnn_limits.base_dir)
new_file = cnn_limits.def_new_file(base_dir)

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
    layer_step = 12



def get_last_full(present):
    for i, r in enumerate(present):
        if not r:
            return i
    return len(present)


_collect_exp = re.compile(r"^0*([0-9]+)_0*([0-9]+)\.npy$")
@experiment.capture
def collect_kernel_matrix(layer_idx, path, _log, symmetric=False):
    fnames = list(sorted(path.iterdir(), key=lambda t: t.stat().st_ctime))
    fpaths = list(path/f for f in fnames)
    indices = (tuple(map(int, _collect_exp.match(name).groups()))
               for name in fnames)
    idx, jdx = zip(*indices)
    # B=batch size.
    n_layers, B, _ = np.load(fpaths[0]).shape

    present_squares = np.zeros((max(idx)//B + 1, max(jdx)//B + 1), bool)
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

    in_memory = len(range(n_layers)[layer_idx])
    K = np.zeros((in_memory, N_full_row, N_full_col), dtype=np.float32)
    for fpath, i, j in zip(tqdm.tqdm(fpaths), idx, jdx):
        if i+B < N_full_row and j+B < N_full_col:
            K[:, i:i+B, j:j+B] = np.load(fpath)[layer_idx, :, :]
            if symmetric and i!=j:
                K[:, j:j+B, i:i+B] = K[:, i:i+B, j:j+B].transpose((0, 2, 1))
    return K


@experiment.capture
def dataset_targets(dset):
    _, y = next(iter(DataLoader(dset, batch_size=len(dset))))
    return np.asarray(y.numpy())


def centered_one_hot(y, N=10):
    oh = y[:, None] == np.arange(N)
    return (oh.astype(np.float64)*N - 1) / (N-1)


def predict_gp(Kxx, Kxt, y):
    jitter_acc = 0.
    K = Kxx.astype(np.float64, copy=True)
    # -inf, then from -26 to 10 (inclusive) in increments of 4
    for log2_jitter in [0]: #[-np.inf, *range(-26, 11, 4)]:
        jitter = 2**log2_jitter
        try:
            K.flat[::K.shape[0]+1] += jitter - jitter_acc
            Ky = sp.linalg.solve(K, centered_one_hot(y), sym_pos=True)
            f = Kxt.transpose((1, 0)) @ Ky
            return np.argmax(f, axis=1), jitter
        except np.linalg.LinAlgError:
            jitter_acc += jitter
    return None, jitter


def accuracy(y, pred):
    if pred is None:
        return -1
    return (y == pred).astype(np.float64).mean(-1)


# TODO Find optimal likelihood for fourier features
# TODO check the non-pooled one, compared with google brain paper

@experiment.automain
def main(kernel_matrix_path, layer_step, _log):
    train_set, test_set = load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    test_Y = dataset_targets(test_set)

    kernel_matrix_path = Path(kernel_matrix_path)
    Kxx_dir = Path(kernel_matrix_path)/"Kxx"

    with open(Kxx_dir/os.listdir(Kxx_dir)[0], "rb") as f:
        N_layers = np.load(f).shape[0]

    layer_slices = [slice(i, min(i+layer_step, N_layers))
                    for i in range(0, N_layers, layer_step)]

    accuracies = collections.defaultdict(lambda: [None]*N_layers)
    jitters = collections.defaultdict(lambda: [None]*N_layers)
    for layer_slice in layer_slices:
        Kxx_sl = collect_kernel_matrix(layer_slice, kernel_matrix_path/"Kxx", symmetric=True)
        Kxt_sl = collect_kernel_matrix(layer_slice, kernel_matrix_path/"Kxt")
        for layer in range(layer_slice.start, layer_slice.stop, 1):
            Kxx = Kxx_sl[layer - layer_slice.start]
            Kxt = Kxt_sl[layer - layer_slice.start]
            effective_N = min(Kxx.shape[0], Kxt.shape[0])
            test_Y = test_Y[:Kxt.shape[1]]
            for N in filter(lambda x: x<=effective_N,
                            [*range(100, 1000, 100),
                             *range(1000, 10000, 1000),
                             *range(10000, 50000, 5000),
                             effective_N]):
                pred_Y, jitter = predict_gp(Kxx[:N, :N], Kxt[:N, :], train_Y[:N])
                acc = accuracy(test_Y, pred_Y)
                _log.info(f"Accuracies at N={N}, layer={layer}, jitter={jitter}: {acc}")
                accuracies[N][layer] = acc
                jitters[N][layer] = jitter

                # Overwrite the file each time; so that if the experiment is
                # interrupted we keep intermediate results
                with new_file("accuracies.pkl") as f:
                    pickle.dump(dict(accuracies), f)
                with new_file("jitters.pkl") as f:
                    pickle.dump(dict(jitters), f)
