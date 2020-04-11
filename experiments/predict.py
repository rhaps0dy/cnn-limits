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
from torch.utils.data import DataLoader, Subset

import cnn_limits

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
    feature_paths = [f"/scratch/ag919/logs/mc_nn/{e}/mc.h5" for e in [2,3,5,7]]

    N_train = None
    N_test = None
    N_classes = 10
    layer_range = (2, 999, 3)
    train_do_range = False


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


@experiment.capture
def dataset_targets(dset):
    _, y = next(iter(DataLoader(dset, batch_size=len(dset))))
    return np.asarray(y.numpy())


@experiment.capture
def centered_one_hot(y, N_classes=10):
    oh = y[:, None] == np.arange(N_classes)
    return (oh.astype(np.float64)*N_classes - 1) / (N_classes-1)


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


def jax_reg_chol(FF, sigy):
    Sxx = FF + jnp.diag(jnp.full((FF.shape[0],), sigy))
    Lxx = jnp.linalg.cholesky(Sxx)
    return Lxx


def jax_kernel_lik(Kxx, y, log_sigy):
    """Likelihood for Bayesian linear regression, using the "kernel matrix"
    version. More efficient when data points < features
    """
    n_train, _ = Kxx.shape
    log_sigy = jnp.squeeze(log_sigy)
    sigy = jnp.exp(log_sigy)
    Lxx = jax_reg_chol(Kxx, sigy)


def jax_linear_lik(FF, FY, y, n_train, log_sigy):
    """Likelihood for Bayesian linear regression, using the "design matrix" version
    of the calculation. More efficient when features < data points"""
    n_features, n_out = FY.shape
    log_sigy = jnp.squeeze(log_sigy)
    sigy = jnp.exp(log_sigy)
    Lxx = jax_reg_chol(FF, sigy)
    A = jax.scipy.linalg.solve_triangular(Lxx, FY, lower=True)

    logdet_Lxx_part = -n_out * jnp.log(jnp.diag(Lxx)).sum()
    exp_part = -.5*(y - A.ravel() @ A.ravel()) / sigy
    scalar_part = -.5*(n_train*n_out)*(math.log(2*math.pi)) - .5*n_out*(n_train-n_features)*log_sigy
    return (logdet_Lxx_part + exp_part + scalar_part)


def likelihood_cholesky(FF, FY, F_test, train_Y, grid_opt_points=100, lower=True):
    assert lower
    FF = jnp.asarray(FF)
    FY = jnp.asarray(FY)
    train_Y = jnp.asarray(train_Y)
    y = train_Y.ravel() @ train_Y.ravel()  # tr(YY^T)
    n_train, _ = train_Y.shape

    _lik = jax.jit(jax.partial(jax_linear_lik, FF, FY, y, n_train))
    grid_x = jnp.log(jnp.square(jnp.linspace(1e-3, 110, grid_opt_points)))[::-1]
    likelihoods = []
    for g in grid_x:
        y = _lik(g)
        if np.isnan(y):
            break
        likelihoods.append(y)
    if len(likelihoods) == 0:
        return np.nan, np.nan, None, None, (grid_x[:0], grid_x[:0])
    likelihoods = np.stack(likelihoods)
    grid_x = np.asarray(grid_x[:len(likelihoods)])
    max_i = likelihoods.argmax()
    sigy = np.exp(grid_x[max_i])
    lik = likelihoods[max_i]
    print(f"Found max sigy={sigy}, lik={lik}")

    Lxx = np.asarray(jax_reg_chol(FF, jnp.asarray(sigy)))
    FtL = jax.scipy.linalg.solve_triangular(Lxx, jnp.asarray(F_test), lower=True).T
    Ly = jax.scipy.linalg.solve_triangular(Lxx, FY, lower=True)
    return sigy, lik, np.asarray(FtL), np.asarray(Ly), (grid_x, likelihoods)


try:
    from cnn_limits import magma
    n_gpu = 1
    @experiment.capture
    def cholesky(Kxx, _log, lower=True):
        K, jitter = _meta_cholesky_jitter(
            lambda K: magma.potrf(K, lower=True, n_gpu=n_gpu), Kxx, _log)
        return K, jitter
        _log.debug("Testing K...")
        Kxx.flat[::Kxx.shape[0]+1] += jitter
        # idx = slice(41234, 41334, 1)
        idx = slice(None, None, None)
        K_ = np.tril(K)
        assert np.allclose(
            K_[idx]@K_[idx].T,
            np.nanmean(np.stack([Kxx[idx, idx], Kxx[idx, idx].T], -1), -1))
        return K, jitter

except OSError:
    print("Warning: Could not load MAGMA.")
    @experiment.capture
    def cholesky(Kxx, _log, lower=True):
        return _meta_cholesky_jitter(
            # Note: _meta_cholesky_jitter copies Kxx, so overwrite_a=True is correct
            lambda K: scipy.linalg.cholesky(K, lower=lower, overwrite_a=True, check_finite=False),
            Kxx, _log)

@experiment.capture
def predict_gp_prepare(Lxx, Kxt, y, _log):
    _log.debug("Solving system...")
    assert Lxx.dtype == np.float64
    A = scipy.linalg.solve_triangular(Lxx, Kxt, lower=True, check_finite=False)
    b = scipy.linalg.solve_triangular(Lxx, y, lower=True, check_finite=False)
    if np.any(np.isnan(A)) or np.any(np.isnan(b)):
        import pdb; pdb.set_trace()
    return A.T, b


def accuracy(y, pred):
    if pred is None:
        return -1
    return (y == pred).astype(np.float64).mean(-1)

# TODO Find optimal likelihood for fourier features


def log_range(N):
    r = itertools.chain(
        range(500, 1000, 500),
        range(1000, 10000, 1000),
        range(10000, 50000, 5000),
        range(50000, 100000, 10000))
    return [*itertools.takewhile(lambda n: n < N, r), N]


def read_features(files, name, out, layer, N, prev_N):
    prev_features = 0
    assert N-prev_N >= 0
    for f in files:
        features = prev_features + f[name].shape[1]
        f[name].read_direct(
                out, source_sel=np.s_[layer, :, prev_N:N],
                dest_sel=np.s_[prev_features:features, :N-prev_N])
        prev_features = features


@experiment.command
def mc_nn(feature_paths, layer_range, train_do_range, _log):
    train_set, test_set = load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    test_Y = dataset_targets(test_set)
    train_Y_oh = centered_one_hot(train_Y)

    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(h5py.File(f, 'r')) for f in feature_paths]

        N_layers, _, N_train = files[0]['F_train'].shape
        _, _, N_test = files[0]['F_test'].shape
        N_features = sum(f['F_train'].shape[1] for f in files)
        FF = np.zeros((N_features, N_features), dtype=np.float64)
        FY = np.zeros((N_features, train_Y_oh.shape[1]), dtype=np.float64)
        F_train = np.empty((N_features, 50000), dtype=np.float64)
        F_test = np.empty((N_features, N_test), dtype=np.float64)

        if train_do_range:
            list_of_N = log_range(N_train)
        else:
            list_of_N = [N_train]
        acc_idx = pd.MultiIndex.from_product(
            [list_of_N, log_range(N_features)],
            names=['N_train', 'N_features'])
        _layer_range = range(layer_range[0],
                             min(N_layers, layer_range[1]),
                             layer_range[2])
        accuracies = pd.DataFrame(index=_layer_range, columns=acc_idx)
        list_of_N = accuracies.columns.levels[0]
        jitters = pd.DataFrame(index=accuracies.index, columns=list_of_N)
        likelihoods = pd.DataFrame(index=accuracies.index, columns=list_of_N)
        opt_grid = {}
        for layer in accuracies.index:
            read_features(files, 'F_test', F_test, layer, N=F_test.shape[1], prev_N=0)
            for prev_N, N in zip([0, *list_of_N], list_of_N):
                _log.debug(f"Reading files for N={N}, layer={layer}")
                read_features(files, 'F_train', F_train, layer, N, prev_N)

                _log.debug("Updating outer product")
                _Ft = F_train[:, :N-prev_N]
                FF += _Ft @ _Ft.T
                FY += _Ft @ train_Y_oh[prev_N:N]

                _log.debug("Cholesky & prediction")
                (jitters[N][layer], likelihoods[N][layer], regression_FtL,
                 regression_Ly, opt_grid[N, layer]
                 ) = likelihood_cholesky(FF, FY, F_test, train_Y_oh[:N], lower=True)
                if regression_FtL is None:
                    continue

                for features in accuracies.columns.levels[1]:
                    pred_F = regression_FtL[:, :features] @ regression_Ly[:features, :]
                    pred_Y = np.argmax(pred_F, axis=1)
                    acc = accuracy(test_Y[:N_test], pred_Y)
                    _log.info(f"Accuracies at N={N}, features={features}, layer={layer}, jitter={jitters[N][layer]}: {acc}")
                    accuracies[N, features][layer] = acc

            accuracies.to_pickle(base_dir()/"accuracies.pkl.gz")
            jitters.to_pickle(base_dir()/"jitters.pkl.gz")
            likelihoods.to_pickle(base_dir()/"likelihoods.pkl.gz")
            with new_file("opt_grid.pkl") as write_f:
                pickle.dump(opt_grid, write_f)
            del regression_Ly
            del regression_FtL
            del pred_F
            del pred_Y

# TODO figure out bug in the 


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

            Lxx, jitter = cholesky(Kxx[:effective_N, :effective_N], lower=True)
            Lxx = Lxx.astype(np.float64, copy=False)
            gp_KtL, gp_Ly = predict_gp_prepare(Lxx, Kxt[:effective_N, :Nt], centered_one_hot(train_Y[:effective_N]))
            del Lxx

            jitters[layer] = jitter
            for N in reversed(log_range(effective_N)):
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
