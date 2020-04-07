import sacred
import numpy as np
import scipy.linalg
import h5py
from cnn_gp import create_h5py_dataset
import collections
import os
from pathlib import Path
import cnn_limits
import re
import pickle
import tqdm
from torch.utils.data import Subset, DataLoader
import faulthandler
faulthandler.enable()

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

    N_train = None
    N_test = None



def get_last_full(present):
    for i, r in enumerate(present):
        if not r:
            return i
    return len(present)


def nan_shape(M, symmetric=False):
    present = ~np.isnan(M)
    if symmetric:
        present |= present.T
    full_cols = present.all(axis=0)
    N_full_col = get_last_full(full_cols)
    full_rows = present[:, :N_full_col].all(axis=1)
    N_full_row = get_last_full(full_rows)
    if symmetric:
        assert N_full_row == N_full_col
    return N_full_row, N_full_col


_collect_exp = re.compile(r"^0*([0-9]+)_0*([0-9]+)\.npy$")
@experiment.capture
def collect_kernel_matrix(layer_idx, path, _log, symmetric=False):
    fpaths = list(sorted(path.iterdir(), key=lambda t: t.stat().st_ctime))
    indices = (tuple(map(int, _collect_exp.match(path.name).groups()))
               for path in fpaths)
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
            # if symmetric and i!=j:
            #     K[:, j:j+B, i:i+B] = K[:, i:i+B, j:j+B].transpose((0, 2, 1))
    return K

@experiment.capture
def collect_kernel_matrix2(f, name, path, _log, symmetric=False):
    fpaths = list(sorted(path.iterdir(), key=lambda t: t.stat().st_ctime))
    indices = (tuple(map(int, _collect_exp.match(path.name).groups()))
               for path in fpaths)
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

    K = create_h5py_dataset(f, B, name, False, N_full_row, N_full_col)
    K.resize(n_layers, axis=0)
    for fpath, i, j in zip(tqdm.tqdm(fpaths), idx, jdx):
        K[:, i:i+B, j:j+B] = np.load(fpath)
    return K


@experiment.capture
def dataset_targets(dset):
    _, y = next(iter(DataLoader(dset, batch_size=len(dset))))
    return np.asarray(y.numpy())


def centered_one_hot(y, N=10):
    oh = y[:, None] == np.arange(N)
    return (oh.astype(np.float64)*N - 1) / (N-1)


def _meta_cholesky_jitter(cholesky_f, Kxx, _log):
    # -inf, then from -26 to 10 (inclusive) in increments of 4
    jitters = [0., *[2**l2j for l2j in range(-26, 11, 4)]]
    # binary search
    # L, R = 0, len(jitters)
    # while L < R:
    for m in range(len(jitters)):
        # m = (L+R)//2
        try:
            _log.debug("Copying K...")
            K = Kxx.astype(np.float64, copy=True, order='F')
            K.flat[::K.shape[0]+1] += jitters[m]
            _log.debug("Attempting Cholesky...")
            K = cholesky_f(K)
            # We know that Kxx can be inverted with jitters[m], or bigger ones.
            # Thus jitter <= jitters[m]
            # R = m
            break
        except np.linalg.LinAlgError:
            # Inverting failed, thus jitters[m] < jitter
            # L = m+1
            pass
    if m == len(jitters):
        return None, -1
    return K, jitters[m]

try:
    from cnn_limits import magma
    n_gpu = 1
    @experiment.capture
    def cholesky(Kxx, _log, lower=True):
        K, jitter = _meta_cholesky_jitter(lambda K: magma.potrf(K, lower=True, n_gpu=n_gpu), Kxx, _log)
        return K, jitter
        _log.debug("Testing K...")
        Kxx.flat[::Kxx.shape[0]+1] += jitter
        # idx = slice(41234, 41334, 1)
        idx = slice(None, None, None)
        K_ = np.tril(K)
        assert np.allclose(K_[idx]@K_[idx].T, np.nanmean(np.stack([Kxx[idx, idx], Kxx[idx, idx].T], -1), -1))
        return K, jitter

except OSError:
    print("Warning: could not load magma. Proceed? ")
    input()
    @experiment.capture
    def cholesky(Kxx, _log, lower=True):
        return _meta_cholesky_jitter(
            lambda K: scipy.linalg.cholesky(K, lower=lower, overwrite_a=True, check_finite=False),
            Kxx)

@experiment.capture
def predict_gp_prepare(Lxx, Kxt, y, _log):
    _log.debug("Solving system...")
    assert Lxx.dtype == np.float64
    A = scipy.linalg.solve_triangular(Lxx, Kxt, lower=True, check_finite=False)
    b = scipy.linalg.solve_triangular(Lxx, centered_one_hot(y), lower=True,
                                      check_finite=False)
    if np.any(np.isnan(A)) or np.any(np.isnan(b)):
        import pdb; pdb.set_trace()
    return A.T, b


def accuracy(y, pred):
    if pred is None:
        return -1
    return (y == pred).astype(np.float64).mean(-1)

# TODO Find optimal likelihood for fourier features
@experiment.command
def collate_kernel_matrix(kernel_matrix_path, _log):
    kernel_matrix_path = Path(kernel_matrix_path)
    with h5py.File(base_dir()/"kernels.h5", "w") as f:
        collect_kernel_matrix2(f, "Kxx", kernel_matrix_path/"Kxx", symmetric=True)
        collect_kernel_matrix2(f, "Kxt", kernel_matrix_path/"Kxt", symmetric=False)

@experiment.automain
def mainv2(kernel_matrix_path, _log):
    train_set, test_set = load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    test_Y = dataset_targets(test_set)
    kernel_matrix_path = Path(kernel_matrix_path)

    with h5py.File(kernel_matrix_path/"kernels.h5", "r") as f:
        N_layers, _, _ = f['Kxx'].shape
        Kxx = np.empty(f['Kxx'].shape[1:], dtype=np.float32)  # We copy it with an astype later
        Kxt = np.empty(f['Kxt'].shape[1:], dtype=np.float64)
        accuracies = collections.defaultdict(lambda: [None]*N_layers)
        jitters = {}

        for layer in range(N_layers-1, -1, -1):
            f['Kxx'].read_direct(Kxx, source_sel=np.s_[layer, :, :])
            f['Kxt'].read_direct(Kxt, source_sel=np.s_[layer, :, :])
            effective_N, Nt = nan_shape(Kxt)
            effective_N = min(nan_shape(Kxx)[0], effective_N)

            all_Ns = reversed([*range(100, 1000, 100),
                               *range(1000, 10000, 1000),
                               *range(10000, 50000, 5000),
                               effective_N])
            Lxx, jitter = cholesky(Kxx[:effective_N, :effective_N], lower=True)
            Lxx = Lxx.astype(np.float64, copy=False)
            gp_KtL, gp_Ly = predict_gp_prepare(Lxx, Kxt[:effective_N, :Nt], train_Y[:effective_N])
            del Lxx

            jitters[layer] = jitter
            for N in filter(lambda x: x<=effective_N, all_Ns):
                pred_F = gp_KtL[:, :N] @ gp_Ly[:N, :]
                pred_Y = np.argmax(pred_F, axis=1)
                acc = accuracy(test_Y[:Nt], pred_Y)
                _log.info(f"Accuracies at N={N}, layer={layer}, jitter={jitter}: {acc}")
                accuracies[N][layer] = acc

                # Overwrite the files each time; so that if the experiment is
                # interrupted we keep intermediate results
                with new_file("accuracies.pkl") as write_f:
                    pickle.dump(dict(accuracies), write_f)
            with new_file("jitters.pkl") as write_f:
                pickle.dump(dict(jitters), write_f)
            del gp_KtL
            del gp_Ly
            del pred_F
            del pred_Y
