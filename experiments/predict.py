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

# TODO remove sgi=False
experiment = sacred.Experiment("predict", save_git_info=False)
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
    feature_paths = [f"/scratch/ag919/logs/mc_nn/{e}/mc.h5" for e in [2,3,5,7,
                                                                      10,11,12,13,14,15,16,
                                                                      17,10,19,20,21,22,23]]
    N_train = None
    N_test = None
    N_classes = 10
    layer_range = (2, 999, 3)
    train_do_range = False
    n_grid_opt_points = 100


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

    logdet_Lxx_part = -n_out * jnp.log(jnp.diag(Lxx)).sum() - .5*n_out*(n_train-n_features)*log_sigy
    exp_part = -.5*(y - A.ravel() @ A.ravel()) / sigy
    scalar_part = -.5*(n_train*n_out)*(math.log(2*math.pi))
    return (logdet_Lxx_part + exp_part + scalar_part)


def likelihood_cholesky(FF, FY, F_test, train_Y, n_grid_opt_points=100, lower=True):
    assert lower
    FF = jnp.asarray(FF)
    FY = jnp.asarray(FY)
    train_Y = jnp.asarray(train_Y)
    y = train_Y.ravel() @ train_Y.ravel()  # tr(YY^T)
    n_train, _ = train_Y.shape

    _lik = jax.jit(jax.partial(jax_linear_lik, FF, FY, y, n_train))
    grid_x = jnp.log(jnp.square(jnp.linspace(1e-3, 110, n_grid_opt_points)))[::-1]
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


    @experiment.capture
    def likelihood_cholesky_kernel(Kxx, Kxt, oh_train_Y, n_grid_opt_points, _log, FY=None, lower=True):
        """Determines optimal likelihood noise for the Gaussian N(y | 0, Kxx).

        Let Lxx = cholesky(Kxx + σ_y² I). Returns solve(Lxx, oh_train_Y) and
        solve(Lxx, Kxt).
        """
        N, D = oh_train_Y.shape
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
        eig = magma.syevd(Kxx.astype(np.float64, copy=True), vectors=True, lower=lower)
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
        eigval_floor = max(-eig.vals.min() + np.finfo(eig.vals.dtype).eps,
                           np.finfo(eig.vals.dtype).eps)
        if calculate_primal:
            sigy_bounds = beta - eig.vals
            # Sigy cannot be negative, and has to be big enough to make Kxx
            # positive semi-definite
            min_sigy = max(sigy_bounds.min(), eigval_floor)
            max_sigy = sigy_bounds.max()
            if max_sigy < 0:
                # Optimal likelihood is with zero sigy
                min_sigy = max_sigy = 0
                n_grid_opt_points = 1

            def lik_fn(sigy):
                grid_vals = eig.vals + np.expand_dims(sigy, -1)
                a = (beta / grid_vals).sum(-1) + np.log(grid_vals).sum(-1)
                assert not np.isnan(a).any()
                return -D/2*(N*np.log(2*np.pi) + a)

            def d_lik_fn(sigy):
                grid_vals = eig.vals + np.expand_dims(sigy, -1)
                a = -(beta / grid_vals**2).sum(-1) + (1/grid_vals).sum(-1)
                assert not np.isnan(a).any()
                return -D/2*a
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
                grid_vals = eig.vals + np.expand_dims(sigy, -1)
                a = (N-n_feat)*np.log(sigy) + np.log(grid_vals).sum(-1)
                b = (tr_YY__D - (beta / grid_vals).sum(-1)) / sigy
                assert not np.isnan(a).any() and not np.isnan(b).any()
                return -D/2*(N*np.log(2*np.pi) + a+b)

            def d_lik_fn(sigy):
                grid_vals = eig.vals + np.expand_dims(sigy, -1)
                grid2_vals = eig.vals + 2*np.expand_dims(sigy, -1)
                a = (N-n_feat)/sigy + (1/grid_vals).sum(-1)
                b = (tr_YY__D - (beta * grid2_vals / grid_vals**2).sum(-1)) / (-sigy**2)
                assert not np.isnan(a).any() and not np.isnan(b).any()
                return -D/2*(a+b)

        dlik_at_min = d_lik_fn(min_sigy)
        dlik_at_max = d_lik_fn(max_sigy)
        if dlik_at_min < 0 and dlik_at_max < 0:
            if d_lik_fn(eigval_floor) <= 4*np.finfo(eig.vals.dtype).eps:
                if eigval_floor < min_sigy:
                    _log.warn(f"maximum was at eigval_floor={eigval_floor}, less than min_sigy={min_sigy}.")
                min_sigy = max_sigy = eigval_floor
            else:
                _log.warn(f"maximum somewhere between eigval_floor={eigval_floor} and min_sigy={min_sigy}.")
                min_sigy = eigval_floor
                max_sigy = min_sigy
        elif dlik_at_min > 0 and dlik_at_max > 0:
            _log.warn(f"maximum not between min_sigy={min_sigy} and max_sigy={max_sigy}.")
            if d_lik_fn(max_sigy + 100) > 0:
                raise ValueError("Definitely not a precision error")
        if min_sigy == max_sigy:
            sigy = min_sigy
        else:
            sigy = scipy.optimize.brentq(d_lik_fn, min_sigy, max_sigy,
                                         maxiter=n_grid_opt_points, disp=True)
        likelihood = lik_fn(sigy)
        # grid = np.square(np.linspace(
        #     np.sqrt(min_sigy), np.sqrt(max_sigy), n_grid_opt_points))
        # likelihoods = lik_fn(grid)
        # opt_i = likelihoods.argmax()
        # sigy = grid[opt_i]

        _log.debug("Cholesky decomposition using optimal sigy")
        L = Kxx.astype(np.float64, copy=True)
        L.flat[::L.shape[0]+1] += sigy
        magma.potrf(L, lower=lower)

        _log.debug("Solving triangular linear systems")
        if not lower:
            L = L.T
        FtL = scipy.linalg.solve_triangular(L, Kxt, lower=True).T
        Ly = scipy.linalg.solve_triangular(L, FY, lower=True)
        return sigy, likelihood, FtL, Ly, (None, None)

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
@experiment.command
def dual_mc_nn(feature_paths, layer_range, _log):
    train_set, test_set = load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    test_Y = dataset_targets(test_set)
    kernel_matrix_path = Path(kernel_matrix_path)

    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(h5py.File(f, 'r')) for f in feature_paths]
        N_layers, _, N_train = files[-1]['F_train'].shape
        _, _, N_test = files[-1]['F_test'].shape
        N_features = sum(f['F_train'].shape[1] for f in files)

        # max_ind_N_features = max(f['F_train'].shape[1] for f in files)
        # F_train = np.empty((max_ind_N_features, N_train), np.float32)

        list_of_N = log_range(N_train)
        _layer_range = range(layer_range[0],
                             min(N_layers, layer_range[1]),
                             layer_range[2])
        accuracies = pd.DataFrame(index=_layer_range, columns=list_of_N)
        jitters = {}
        # likelihoods = pd.DataFrame(index=accuracies.index, columns=list_of_N)


        Kxx = np.empty((N_train, N_train), dtype=np.float32)  # We copy it with an astype later
        Kxt = np.empty((N_train, N_test), dtype=np.float64)

        for layer in accuracies.index:
            Kxx[...] = 0
            Kxt[...] = 0
            for f in files:
                # f['F_train'].read_direct(
                #     F_train,
                #     source_sel=np.s_[layer, :, :],
                #     dest_sel=np.s_[:f['F_train'].shape[1], :])
                F_train = f['F_train'][layer, :, :]
                Kxx += F_train.T @ F_train
                Kxt += F_train.T @ f['F_test'][layer, :, :]

            effective_N = N_train
            Lxx, jitter = cholesky(Kxx[:effective_N, :effective_N], lower=True)
            Lxx = Lxx.astype(np.float64, copy=False)
            gp_KtL, gp_Ly = predict_gp_prepare(Lxx, Kxt[:effective_N, :Nt], centered_one_hot(train_Y[:effective_N]))
            del Lxx

            jitters[layer] = jitter
            for N in reversed(log_range(effective_N)):
                pred_F = gp_KtL[:, :N] @ gp_Ly[:N, :]
                pred_Y = np.argmax(pred_F, axis=1)
                acc = accuracy(test_Y[:N_test], pred_Y)
                _log.info(f"Accuracies at N={N}, layer={layer}, jitter={jitter}: {acc}")
                accuracies.loc[layer, N] = acc

                # Overwrite the files each time; so that if the experiment is
                # interrupted we keep intermediate results
                accuracies.to_pickle(base_dir()/"accuracies.pkl")
            with new_file("jitters.pkl") as write_f:
                pickle.dump(dict(jitters), write_f)
            del gp_KtL
            del gp_Ly
            del pred_F
            del pred_Y



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
