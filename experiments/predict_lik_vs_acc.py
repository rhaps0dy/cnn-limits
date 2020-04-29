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
import torch
from torch.utils.data import DataLoader, Subset

import cnn_limits
import cnn_limits.sacred_utils as SU
from cnn_gp import create_h5py_dataset

faulthandler.enable()

experiment = sacred.Experiment("predict_lik_vs_acc", [SU.ingredient])
if __name__ == '__main__':
    SU.add_file_observer(experiment)


@experiment.config
def config():
    kernel_matrix_path = "/scratch/ag919/logs/save_new/166"
    feature_paths = [f"/scratch/ag919/logs/mc_nn/{e}/mc.h5" for e in [2,3,5,7,
                                                                      10,11,12,13,14,15,16,
                                                                      17,10,19,20,21,22,23]]
    N_classes = 10
    layer_range = (2, 999, 3)
    n_grid_opt_points = 1000

    N_files = 9999
    eig_engine = "scipy"


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


    @experiment.capture
    def likelihood_cholesky(Kxx, Kxt, oh_train_Y, test_Y, n_grid_opt_points, _log, eig_engine, FY=None, lower=True):
        """Determines optimal likelihood noise for the Gaussian N(y | 0, Kxx).

        Let Lxx = cholesky(Kxx + σ_y² I). Returns solve(Lxx, oh_train_Y) and
        solve(Lxx, Kxt).
        """
        N, D = oh_train_Y.shape
        Kxt = jnp.asarray(Kxt)
        oh_train_Y = jnp.asarray(oh_train_Y)
        test_Y = jnp.asarray(test_Y)
        if FY is not None:
            FY = jnp.asarray(FY)
        """In what follows, σ_y is the variance, not the standard deviation.

        For j=1..D, the marginal likelihood is
              -1/2 ∑_j [y_jᵀ(Kxx + σ_y I)⁻¹y_j      + logdet(Kxx + σ_y I) + N log(2π)]
            = -D/2  [(∑_j y_jᵀ(Kxx + σ_y I)⁻¹y_j)/D + logdet(Kxx + σ_y I) + N log(2π)]

            = -D/2 [N log(2π) + ∑_i ((∑_j α_ij)/D / (λ_i + σ_y) + log(λ_i + σ_y))]

        using the eigendecomposition. α_ij = [Qᵀ oh_train_Y]²_ij. We'll assume
        all eigenvalues λ_i are positive.

        We have to find the argmax with respect to σ_y. The negative likelihood
        is a sum over i of:

            -2/D Lik(σ_y) + C = β_i / (λ_i + σ_y) + log(λ_i + σ_y)

        for β_i = (∑_j α_ij)/D. The derivative is

            -β_i/(λ_i + σ_y)² + 1/(λ_i + σ_y),

        which has its only zero at σ_y = β_i - λ_i. Thus, the minimum σ_y of
        the sum over i (thus, the maximum of the likelihood) has to be between
        min_i(β_i - λ_i) and max_i(β_i - λ_i). We will find it with a
        logarithmic grid search.
        """
        _log.debug("Calculating eigendecomposition")
        if isinstance(Kxx, magma.EigenOut):
            eig = magma.EigenOut(*map(jnp.asarray, Kxx))
        else:
            if eig_engine == "jax":
                Kxx = jnp.asarray(Kxx, dtype=jnp.float64)
                eig = magma.EigenOut(*jax.scipy.linalg.eigh(Kxx, lower=lower, eigvals_only=False, check_finite=False))
            else:
                if eig_engine == "magma":
                    eig = magma.syevd(Kxx, vectors=True, lower=lower)
                    # eig = magma.syevd(Kxx.astype(np.float64, copy=True), vectors=True, lower=lower)
                elif eig_engine == "scipy":
                    eig = magma.EigenOut(*scipy.linalg.eigh(Kxx.astype(np.float64, copy=False),
                                                            lower=lower, eigvals_only=False, check_finite=False))
                else:
                    raise ValueError(f"eig_engine={eig_engine}")
                eig = magma.EigenOut(*map(jnp.asarray, eig))
        _log.debug("Calculating alpha and beta")
        if FY is None:
            FY = oh_train_Y
            calculate_primal = True
        else:
            calculate_primal = False
        alpha = eig.vecs.T @ FY
        beta = (alpha[:, None, :] @ alpha[:, :, None]).ravel() / D

        """If the smallest eigenvalue of Kxx is negative, the analysis above does not
        apply. We first need to add jitter to it until the smallest eigenvalue
        is positive. In that case:

            eigvals = eig.vals + abs(eig.vals.min())

        Then, we compute (sigy_bounds := beta-eigvals) as usual, to calculate
        the interval where the optimal σ_y lies. Finally, we need to add
        `abs(eig.vals.min())` back to this interval, so the result is:

            sigy_bounds = beta - (eig.vals + abs(eig.vals.min())) + abs(eig.vals.min())
                        = beta - eig.vals

        The computations are thus the same.
        """
        if eig.vals.min() <= 0:
            eigval_floor = np.nextafter(np.abs(eig.vals.min()), np.inf)
        else:
            eigval_floor = 0.
        if calculate_primal:
            sigy_bounds = beta - eig.vals
            # Sigy cannot be negative, and has to be big enough to make Kxx
            # positive semi-definite
            min_sigy = max(sigy_bounds.min(), eigval_floor)
            max_sigy = sigy_bounds.max()
            if max_sigy < 0:
                # Optimal likelihood is with zero sigy
                min_sigy = max_sigy = 0

            def lik_fn(sigy):
                grid_vals = eig.vals + jnp.expand_dims(sigy, -1)
                a = (beta / grid_vals).sum(-1) + jnp.log(grid_vals).sum(-1)
                return -D/2*(N*math.log(2*math.pi) + a)
        else:
            """Evaluate the dual likelihood:
                -D/2 [N log(2π) + (N-F) log(σ_y) + tr_YY/(D σ_y)
                     + ∑_i log(λ_i + σ_y) - β_i/(σ_y(λ_i + σ_y))]
            where tr_YY = tr{YᵀY} = (∑_j y_jᵀy_j).

            The dual (negative) likelihood derivative is:
                (N-F)/σ_y - (tr_YY/D)/σ_y² + (∑_i positive terms)


            The positive terms imply that that part of the likelihood increase
            monotonically. Thus, an upper bound on the optimal sigy is the
            location of the first term's minimum:
                (N-F)/σ_y - (tr_YY/D)/σ_y² = 0
                => σ_y = tr_YY/D / (N-F)
            """
            n_feat, _ = FY.shape
            tr_YY__D = (oh_train_Y.ravel() @ oh_train_Y.ravel())/D
            min_sigy = eigval_floor
            max_sigy = tr_YY__D / (N-n_feat)

            def lik_fn(sigy):
                grid_vals = eig.vals + jnp.expand_dims(sigy, -1)
                a = (N-n_feat)*jnp.log(sigy) + jnp.log(grid_vals).sum(-1)
                b = (tr_YY__D - (beta / grid_vals).sum(-1)) / sigy
                return -D/2*(N*math.log(2*math.pi) + a+b)

        print("Eigval_floor, min_sigy, max_sigy=", eigval_floor, min_sigy, max_sigy)
        grid = jnp.exp(jnp.linspace(max(jnp.log(eigval_floor), -36), 0., n_grid_opt_points))
        likelihoods = lik_fn(grid)
        gamma = Kxt.T @ eig.vecs

        def jax_acc(sigy):
            pred_F = (gamma / (eig.vals + sigy)) @ alpha
            pred_Y = jnp.argmax(pred_F, axis=-1)
            return (pred_Y == test_Y).mean(-1)

        accuracies = [float(jax_acc(sigy)) for sigy in grid]
        return tuple(map(np.asarray, (grid, likelihoods, accuracies))) #, eig.vals)))

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


def log_range(N):
    if N == 0:
        return []
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
def mc_nn(feature_paths, layer_range, N_files, _log):
    train_set, test_set = load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    test_Y = dataset_targets(test_set)
    oh_train_Y = centered_one_hot(train_Y)

    assert len(feature_paths) == 1 or N_files == 1, "notimplemented"

    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(h5py.File(f, 'r')) for f in feature_paths[:N_files]]

        N_layers, _, N_train = files[0]['F_train'].shape
        _, _, N_test = files[0]['F_test'].shape
        N_features = min(files[0]['F_train'].shape[1],
                         files[0]['F_test'].shape[1])

        FF = np.zeros((N_features, N_features), dtype=np.float64)
        FY = np.zeros((N_features, oh_train_Y.shape[1]), dtype=np.float64)

        list_of_N = [5000, 10000]
        _layer_range = range(layer_range[0],
                             min(N_layers, layer_range[1]),
                             layer_range[2])
        data = {}
        for layer in _layer_range:
            _log.debug(f"Reading files for layer={layer}")
            F_train = files[0]['F_train'][layer, :, :]
            F_test = files[0]['F_test'][layer, :, :]
            if layer%3 == 1:  # is TICK
                # Compensate size of filter
                F_train /= 32**2
                F_test /= 32**2

            for N in list_of_N:
                _log.debug(f"Calculating outer product for N={N}")
                FF = F_train[:, :N] @ F_train[:, :N].T
                FY = F_train[:, :N] @ oh_train_Y[:N]

                for n_feat in log_range(N_features):
                    _log.debug(f"Cholesky & prediction for N={N}, n_feat={n_feat}, layer={layer}")
                    data[layer, N, n_feat] = likelihood_cholesky(
                        Kxx=FF[:n_feat, :n_feat] / n_feat,
                        Kxt=F_test[:n_feat, :] / n_feat,
                        oh_train_Y=oh_train_Y[:N],
                        test_Y=test_Y,
                        FY=FY[:n_feat])
                pd.to_pickle(data, base_dir()/"grid_lik_acc.pkl.gz")

@experiment.command
def dual_mc_nn(feature_paths, layer_range, N_files, _log):
    train_set, test_set = load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    test_Y = dataset_targets(test_set)
    oh_train_Y = centered_one_hot(train_Y)

    feature_paths = feature_paths[:N_files]

    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(h5py.File(f, 'r')) for f in feature_paths]
        N_layers, _, N_train = files[-1]['F_train'].shape
        _, _, N_test = files[-1]['F_test'].shape
        N_features = sum(f['F_train'].shape[1] for f in files)

        # max_ind_N_features = max(f['F_train'].shape[1] for f in files)
        # F_train = np.empty((max_ind_N_features, N_train), np.float32)

        # list_of_N = [N_train]  #log_range(N_train) + [N_features]  # For double descent
        # print(N_features, list_of_N)
        list_of_F = log_range(N_features)
        _layer_range = range(layer_range[0],
                             min(N_layers, layer_range[1]),
                             layer_range[2])
        accuracies = pd.DataFrame(index=_layer_range, columns=list_of_F)
        jitters = pd.DataFrame(index=accuracies.index, columns=accuracies.columns)
        likelihoods = pd.DataFrame(index=accuracies.index, columns=accuracies.columns)
        data = {}

        Kxx = np.empty((N_train, N_train), dtype=np.float64)
        Kxt = np.empty((N_train, N_test), dtype=np.float64)

        with h5py.File(base_dir()/"kernels.h5", "w") as kernel_f:
            create_h5py_dataset(kernel_f, N_train, "Kxx_vecs",   diag=False, N=N_train, N2=N_train)
            create_h5py_dataset(kernel_f, N_train, "Kxx_vals", diag=True,  N=N_train, N2=None)
            create_h5py_dataset(kernel_f, N_train, "Kxt",     diag=False, N=N_train, N2=N_test)
            # kernel_f["Kxx"].resize(N_layers, axis=0)
            # kernel_f["Kxt"].resize(N_layers, axis=0)
            kernel_f["Kxx_vecs"].resize(3*len(files), axis=0)
            kernel_f["Kxx_vals"].resize(3*len(files), axis=0)
            kernel_f["Kxt"].resize(3*len(files), axis=0)
            for layer in accuracies.index:
                Kxx[...] = 0
                Kxt[...] = 0
                _log.debug("Loading from all the files...")
                # if layer == 107:
                #     Kxx = np.load("/scratch/ag919/logs/predict_lik_vs_acc/Kxx.npy")
                #     Kxt = np.load("/scratch/ag919/logs/predict_lik_vs_acc/Kxt.npy")
                # elif layer == 105:
                #     Kxx = np.load("/scratch/ag919/logs/predict_lik_vs_acc/Kxx2.npy")
                #     Kxt = np.load("/scratch/ag919/logs/predict_lik_vs_acc/Kxt2.npy")
                # else:
                for out_i, (f, name) in enumerate(zip(files, feature_paths)):
                    _log.debug(f"Loading file {name}")
                    F_train = torch.from_numpy(f['F_train'][layer, :, :N_train]).to(dtype=torch.float64, device='cuda')
                    F_test = torch.from_numpy(f['F_test'][layer, :, :N_test]).to(dtype=torch.float64, device='cuda')

                    factor = math.sqrt(N_features)
                    if layer%3 == 1:  # is TICK
                        # Compensate size of filter
                        factor = factor * 32**2
                    F_train /= factor
                    F_test /= factor

                    # The overal Kxx and Kxt will be "multiplied" by the number
                    # of files.
                    Kxx += (F_train.t() @ F_train).cpu().numpy()
                    Kxt += (F_train.t() @ F_test).cpu().numpy()
                    print(Kxx[:5, :5])
                    eig = magma.syevd(Kxx.copy(), vectors=True, lower=True)
                    kernel_f["Kxx_vecs"][3*out_i + layer%3, :, :] = eig.vecs
                    kernel_f["Kxx_vals"][3*out_i + layer%3, :] = eig.vals
                    kernel_f["Kxt"][3*out_i + layer%3, :, :] = Kxt

                # for N in reversed(accuracies.columns):
                #     data[layer, N] = likelihood_cholesky(
                #         Kxx[:N, :N], Kxt[:N], oh_train_Y[:N], test_Y, FY=None,
                #         lower=True)
                #     pd.to_pickle(data, base_dir()/"grid_lik_acc.pkl.gz")


@experiment.main
def main_no_eig(kernel_matrix_path):
    train_set, test_set = SU.load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    oh_train_Y = centered_one_hot(train_Y)
    test_Y = dataset_targets(test_set)
    kernel_matrix_path = Path(kernel_matrix_path)

    with h5py.File(kernel_matrix_path/"kernels.h5", "r") as f:
        N_layers, _, _ = f['Kxx'].shape

        all_N = [2**i * 10 for i in range(8)]  # up to 1280
        data = pd.DataFrame(index=range(N_layers), columns=all_N)

        for layer in data.index:
            Kxx = f['Kxx'][layer]
            Kxt = f['Kxt'][layer]
            for N in data.columns:
                data.loc[layer, N] = likelihood_cholesky(
                    Kxx[:N, :N], Kxt[:N], oh_train_Y[:N], test_Y, FY=None,
                    lower=True)
            pd.to_pickle(data, SU.base_dir()/"grid_lik_acc.pkl.gz")


@experiment.automain
def mainv2(kernel_matrix_path, _log):
    train_set, test_set = SU.load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    oh_train_Y = centered_one_hot(train_Y)
    test_Y = dataset_targets(test_set)
    kernel_matrix_path = Path(kernel_matrix_path)

    with h5py.File(kernel_matrix_path/"kernels.h5", "r") as f:
        N_layers, _, _ = f['Kxx_vecs'].shape
        Kxx_vecs = np.empty(f['Kxx_vecs'].shape[1:], dtype=np.float64)  # We copy it with an astype later
        Kxx_vals = np.empty(f['Kxx_vals'].shape[1:], dtype=np.float64)  # We copy it with an astype later
        Kxt = np.empty(f['Kxt'].shape[1:], dtype=np.float64)
        data = {}

        for layer in range(N_layers-1, -1, -1):
            f['Kxx_vecs'].read_direct(Kxx_vecs, source_sel=np.s_[layer, :, :])
            f['Kxx_vals'].read_direct(Kxx_vals, source_sel=np.s_[layer, :])
            f['Kxt'].read_direct(Kxt, source_sel=np.s_[layer, :, :])
            if (np.any(np.isnan(Kxx_vecs))
                or np.any(np.isnan(Kxx_vals))
                or np.any(np.isnan(Kxt))):
                print(f"Found nan at layer {layer}")
                continue
            effective_N, Nt = Kxt.shape
            # effective_N, Nt = nan_shape(Kxt)
            # effective_N = min(nan_shape(Kxx)[0], effective_N)
            # print(f"effective_N={effective_N}, Nt={Nt}")
            # for N in reversed(log_range(effective_N)):
            if True:
                N = effective_N
                data[layer, N] = likelihood_cholesky(
                    # Kxx[:N, :N],
                    magma.EigenOut(Kxx_vals, Kxx_vecs),
                    Kxt[:N], oh_train_Y[:N], test_Y, FY=None,
                    lower=True)
                pd.to_pickle(data, base_dir()/"grid_lik_acc.pkl.gz")
