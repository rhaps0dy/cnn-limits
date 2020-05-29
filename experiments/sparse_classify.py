import collections
from neural_tangents import stax
import contextlib
import faulthandler
import itertools
import math
import os
import pickle
from pathlib import Path

import h5py
import jax
import jax.scipy.linalg
import jax.numpy as jnp
import numpy as np
import pandas as pd
import sacred
import scipy.linalg
import scipy.optimize
import torch
from torch.utils.data import DataLoader, Subset

import cnn_limits.models
import cnn_limits.models_torch
import cnn_limits.sacred_utils as SU
from cnn_gp import create_h5py_dataset
from cnn_limits.layers import proj_relu_kernel
from experiments.predict_cv_acc import (dataset_targets, centered_one_hot,
                                        EigenOut, eigdecompose, accuracy_eig, fold_idx)
from experiments.predict_cv_acc import experiment as predict_cv_acc_experiment
from cnn_limits.sparse import patch_kernel_fn, patch_kernel_fn_torch
from nigp.tbx import PrintTimings

faulthandler.enable()

experiment = sacred.Experiment("sparse_classify", [SU.ingredient, predict_cv_acc_experiment])
if __name__ == '__main__':
    SU.add_file_observer(experiment)

@experiment.config
def config():
    batch_size = 1000
    print_interval = 2.
    model = "CNTK_nopool"
    stride = 1
    model_args = dict()
    N_inducing = 4096


@experiment.capture
def do_one_N_chol(Kuu, Kux, Kut, oh_train_Y, test_Y, n_splits, predict_cv_acc, _log):
    n_grid_opt_points = predict_cv_acc['n_grid_opt_points']
    if n_splits == -1:
        raise NotImplementedError("leave-one-out")

    try:
        Luu = np.linalg.cholesky(Kuu)
    except np.linalg.LinAlgError:
        for i in range(-6, 6, 2):
            _log.warning(f"Adding 10^{i} to Kuu")
            K_ = Kuu + (10**i) * np.eye(Kuu.shape[0])
            try:
                Luu = np.linalg.cholesky(K_)
                break
            except np.linalg.LinAlgError:
                pass
    Luu_Kux = scipy.linalg.solve_triangular(Luu, Kux, lower=True)

    fold_eig = []
    for train_idx, test_idx in fold_idx(Kux.shape[1], n_splits):
        A = Luu_Kux[:, train_idx] @ oh_train_Y[train_idx, :]
        B = Luu_Kux[:, test_idx]
        # # Luu_ = np.linalg.cholesky(Kuu[train_idx, :][:, train_idx] + 9.737369588754647e-06 * np.eye(Kuu[train_idx].shape[0]))
        # # A = scipy.linalg.solve_triangular(Luu_, oh_train_Y[train_idx, :])
        # # B = scipy.linalg.solve_triangular(Luu_, Kux[train_idx, :][:, test_idx])
        # # eig = eigdecompose(np.eye(Kuu[train_idx].shape[0])) #A @ A.T)
        # A = oh_train_Y[train_idx, :]
        # B = Kux[train_idx, :][:, test_idx]
        # eig = eigdecompose(Kuu[train_idx, :][:, train_idx]
        #                    + 9.737369588754637e-06 * np.eye(Kuu[train_idx].shape[0]))
        eig = eigdecompose(Luu_Kux @ Luu_Kux.T)
        fold_eig.append((eig, A, B))

    min_eigval = min(*[e.vals.min() for e, _, _ in fold_eig])
    eigval_floor = max(0., -min_eigval)
    while min_eigval + eigval_floor <= 0.0:
        eigval_floor = np.nextafter(eigval_floor, np.inf)

    sigy_grid = np.exp(np.linspace(
        max(np.log(eigval_floor), -36), 5, n_grid_opt_points))
    # Make sure that none is lower than eigval_floor
    sigy_grid = np.clip(sigy_grid, eigval_floor, np.inf)

    folds = []
    for (train_idx, test_idx), (eig, A, B) in zip(fold_idx(Kux.shape[1], n_splits),
                                                  fold_eig):
        _oh_train_Y = oh_train_Y[train_idx, :]
        _test_Y = oh_train_Y[test_idx, :].argmax(-1)
        _, fold = accuracy_eig(eig, Kxt=B, oh_train_Y=_oh_train_Y, test_Y=_test_Y,
                               sigy_grid=sigy_grid, FY=A)
        folds.append(fold)

    grid_acc = np.mean(folds, axis=0)
    sigy = np.expand_dims(sigy_grid[grid_acc.argmax()], 0)
    Luu_Kut = scipy.linalg.solve_triangular(Luu, Kut, lower=True)

    return (sigy_grid, grid_acc), accuracy_eig(
        # Kuu, Kut,
        eigdecompose(Luu_Kux @ Luu_Kux.T, lower=True), Luu_Kut,
        oh_train_Y, test_Y, sigy, FY=Luu_Kux @ oh_train_Y)


def make_sigy_grid(min_eigval, n_grid_opt_points):
    eigval_floor = max(0., -min_eigval)
    while min_eigval + eigval_floor <= 0.0:
        eigval_floor = np.nextafter(eigval_floor, np.inf)

    sigy_grid = np.exp(np.linspace(
        max(np.log(eigval_floor), -36), 5, n_grid_opt_points))
    # Make sure that none is lower than eigval_floor
    sigy_grid = np.clip(sigy_grid, eigval_floor, np.inf)
    return sigy_grid

@experiment.capture
def find_cv_opt(Kuu_eig, Kux, Kut, oh_train_Y, n_splits, predict_cv_acc):
    Luu_Kux = (Kuu_eig.vecs * Kuu_eig.vals**-.5)  @ Kux

    fold_eig = []
    for train_idx, test_idx in fold_idx(Kux.shape[1], n_splits):
        A = Luu_Kux[:, train_idx] @ oh_train_Y[train_idx, :]
        B = Luu_Kux[:, test_idx]
        eig = eigdecompose(A @ A.T)
        fold_eig.append((eig, A, B))

    min_eigval = min(*[e.vals.min() for e, _, _ in fold_eig])
    sigy_grid = make_sigy_grid(min_eigval, predict_cv_acc['n_grid_opt_points'])

    folds = []
    for (train_idx, test_idx), (eig, A, B) in zip(fold_idx(Kux.shape[1], n_splits),
                                                  fold_eig):
        _oh_train_Y = oh_train_Y[train_idx, :]
        _test_Y = oh_train_Y[test_idx, :].argmax(-1)
        _, fold = accuracy_eig(eig, Kxt=B, oh_train_Y=_oh_train_Y, test_Y=_test_Y,
                               sigy_grid=sigy_grid, FY=A)
        folds.append(fold)
    grid_acc = np.mean(folds, axis=0)
    return sigy_grid, grid_acc


@experiment.capture
def do_one_N(Kuu, Kux, Kut, oh_train_Y, test_Y, n_splits, predict_cv_acc, _log):
    if n_splits == -1:
        raise NotImplementedError("leave-one-out")

    Kuu_eig = eigdecompose(Kuu)
    outer_sigy_grid = make_sigy_grid(Kuu_eig.vals.min(), 100)
    max_acc = -np.inf
    for outer_sigy in outer_sigy_grid:
        sigy_grid, grid_acc = find_cv_opt(EigenOut(vals=Kuu_eig.vals + outer_sigy, vecs=Kuu_eig.vecs),
                                          Kux, Kut, oh_train_Y, n_splits)
        sigy = np.expand_dims(sigy_grid[grid_acc.argmax()], 0)
        max_grid_acc = grid_acc.max()
        _log.info(f"For outer_sigy={outer_sigy}: sigy={sigy}, acc={max_grid_acc}, max_acc={max_acc}")
        if max_grid_acc > max_acc:
            max_acc = max_grid_acc
            max_outer_sigy = outer_sigy
    _log.info(f"Max acc: {max_acc}, outer_sigy={max_outer_sigy}")

    Kuu_eig = EigenOut(vals=Kuu_eig.vals + max_outer_sigy, vecs=Kuu_eig.vecs)
    Luu_Kux = (Kuu_eig.vecs * Kuu_eig.vals**-.5)  @ Kux
    Luu_Kut = (Kuu_eig.vecs * Kuu_eig.vals**-.5)  @ Kut

    return (sigy_grid, grid_acc), accuracy_eig(
        # Kuu, Kut,
        eigdecompose(Luu_Kux @ Luu_Kux.T, lower=True), Luu_Kut,
        oh_train_Y, test_Y, sigy, FY=Luu_Kux @ oh_train_Y)


@experiment.command
def test_rbf(_log, i_SU):
    import gpytorch
    torch.set_default_dtype(torch.float64)

    kern = gpytorch.kernels.RBFKernel()
    kern.lengthscale = 3.
    X = torch.linspace(0, 99, i_SU['N_train'])[:, None][torch.randperm(i_SU['N_train'])]
    Xt = torch.linspace(0.01, 29.01, i_SU['N_test'])[:, None]
    Z = Xt + 0.02

    K = kern(torch.cat((X, Xt), dim=0)).evaluate()
    Y = gpytorch.lazy.NonLazyTensor(K).zero_mean_mvn_samples(10).t().detach().numpy()

    with torch.no_grad():
        Kxx = K[:X.size(0), :X.size(0)].detach().numpy()
        Kxt = K[:X.size(0), X.size(0):].detach().numpy()
        oh_train_Y = Y[:X.size(0)]
        test_Y = Y[X.size(0):].argmax(1)
        Kzx = kern(Z, X).evaluate().detach().numpy()
        Kzt = kern(Z, Xt).evaluate().detach().numpy()
        Kzz = kern(Z).evaluate().detach().numpy()

    from experiments.predict_cv_acc import do_one_N as orig_do_one_N

    # data, accuracy = orig_do_one_N(Kxx, Kxt, oh_train_Y, test_Y, n_splits=4)
    data, accuracy = do_one_N_chol(Kzz, Kzx, Kzt, oh_train_Y, test_Y, n_splits=4)

    (sigy, acc) = map(np.squeeze, accuracy)
    _log.info(f"For RBF kernel, N={X.size(0)}, sigy={sigy}; accuracy={acc}, cv_accuracy={np.max(data[1])}")




@experiment.command
def stored_kernel(predict_cv_acc, _log):
    kernel_matrix_path = predict_cv_acc['kernel_matrix_path']
    multiply_var = predict_cv_acc['multiply_var']
    apply_relu = predict_cv_acc['apply_relu']
    n_splits = predict_cv_acc['n_splits']

    kernel_matrix_path = Path(kernel_matrix_path)
    train_set, test_set = SU.load_sorted_dataset(
        dataset_treatment="load_train_idx",
        train_idx_path=kernel_matrix_path)

    train_Y = dataset_targets(train_set)
    oh_train_Y = centered_one_hot(train_Y).astype(np.float64)
    test_Y = dataset_targets(test_set)

    with h5py.File(kernel_matrix_path/"kernels.h5", "r") as f:
        N_layers, N_total, _ = f['Kxx'].shape

        all_N = list(itertools.takewhile(
            lambda a: a <= N_total,
            (2**i * 10 for i in itertools.count(0))))
        data = pd.DataFrame(index=range(N_layers), columns=all_N)
        accuracy = pd.DataFrame(index=range(N_layers), columns=all_N)

        new_base_dir = SU.base_dir()/f"n_splits_{n_splits}"
        os.makedirs(new_base_dir, exist_ok=True)

        for layer in reversed(data.index):
            Kxx = f['Kxx'][layer].astype(np.float64)
            mask = np.triu(np.ones(Kxx.shape, dtype=np.bool), k=1)
            Kxx[mask] = Kxx.T[mask]
            assert np.array_equal(Kxx, Kxx.T)
            assert np.all(np.isfinite(Kxx))

            Kxt = f['Kxt'][layer].astype(np.float64)
            try:
                Kx_diag = f['Kx_diag'][layer].astype(np.float64)
                Kt_diag = f['Kt_diag'][layer].astype(np.float64)
            except KeyError:
                Kx_diag = np.diag(Kxx)

            if multiply_var:
                assert np.allclose(np.diag(Kxx), 1.)
                Kxx *= np.sqrt(Kx_diag[:, None]*Kx_diag)
                Kxt *= np.sqrt(Kx_diag[:, None]*Kt_diag)
            else:
                assert not np.allclose(np.diag(Kxx), 1.)

            if apply_relu:
                prod12 = Kx_diag[:, None] * Kx_diag
                Kxx = np.asarray(proj_relu_kernel(Kxx, prod12, False))
                Kxx = Kxx * prod12    # Use covariance, not correlation matrix
                prod12_t = Kx_diag[:, None] * Kt_diag
                Kxt = np.asarray(proj_relu_kernel(Kxt, prod12_t, False))
                Kxt = Kxt * prod12_t

            for N in [1280]: #reversed(data.columns):
                # Made a mistake, the same label are all contiguous in the training set.
                train_idx = slice(0, N_total, N_total//N)
                data.loc[layer, N], accuracy.loc[layer, N] = do_one_N_chol(
                    Kxx[train_idx, train_idx], Kxx[train_idx, :],
                    Kxt[train_idx, :], oh_train_Y, test_Y,
                    n_splits=n_splits)
                (sigy, acc) = map(np.squeeze, accuracy.loc[layer, N])
                _log.info(f"For layer={layer}, N={N}, sigy={sigy}; accuracy={acc}, cv_accuracy={np.max(data.loc[layer, N][1])}")
                pd.to_pickle(data, new_base_dir/"grid_acc.pkl.gz")
                pd.to_pickle(accuracy, new_base_dir/"accuracy.pkl.gz")

@experiment.capture
def torch_interdomain_kernel(model, model_args, stride):
    no_pool = getattr(cnn_limits.models_torch, model)(channels=1, **model_args)
    no_pool = no_pool.cuda()

    nngp_zz_fn = patch_kernel_fn_torch(no_pool, (stride, stride), W_cov=None)

    def kern_zz_fn(i, z1, j, z2):
        z1 = z1.unsqueeze(0).to(device='cuda', dtype=torch.float32)
        z2 = (None if z2 is None else z2.unsqueeze(0).to(device='cuda', dtype=torch.float32))
        return nngp_zz_fn(i, j, z1, z2).cpu().numpy()

    def kern_zx_fn(i, z1, x2):
        z1 = z1.unsqueeze(0).to(device='cuda', dtype=torch.float32)
        x2 = x2.to(device='cuda', dtype=torch.float32)
        _, _, _, W = x2.shape
        if W%stride != 0:
            raise NotImplementedError("W%stride != 0")

        res = nngp_zz_fn(i, -(W//stride)+1, z1, x2)
        for j in range(-(W//stride)+2, (W//stride)):
            res += nngp_zz_fn(i, j, z1, x2)
        return res.cpu().numpy()

    return kern_zz_fn, kern_zx_fn, NotImplemented


@experiment.capture
def interdomain_kernel(model, model_args, stride):
    no_pool = getattr(cnn_limits.models, model)(channels=1, **model_args)

    _, _, _pool_kernel_fn = stax.serial(no_pool, stax.GlobalAvgPool(), stax.Flatten())

    _, _, _no_pool_kfn = no_pool
    _kern_zz_fn = patch_kernel_fn(_no_pool_kfn, (stride, stride), W_cov=None)

    def _nngp_zz_fn(i, j, z1, z2):
        print(f"When jitting: use i={i}, j={j}")
        return jnp.sum(_kern_zz_fn(i, j, z1, z2, get='nngp').nngp, (-1, -2))
    #_nngp_zz_fn = jax.jit(_nngp_zz_fn, static_argnums=(0, 1))

    _compile_cache = {}
    def nngp_zz_fn(i, j, z1, z2):
        try:
            return _compile_cache[i, j](z1, z2)
        except KeyError:
            _compile_cache[i, j] = jax.jit(lambda x, y: _nngp_zz_fn(i, j, x, y))
            return nngp_zz_fn(i, j, z1, z2)

    @jax.jit
    def _kern_x_fn(x1):
        # NCHW->NHWC
        x1 = jnp.moveaxis(x1, 1, -1)
        return _pool_kernel_fn(x1, x2=None, get='var1')

    def zx_kernel_fn(i, z1, x):
        assert isinstance(i, int)

        z1 = jnp.expand_dims(z1, 0)
        # NCHW->NHWC
        z1 = jnp.moveaxis(z1, 1, -1)
        x = jnp.moveaxis(x, 1, -1)

        _, _, W, _ = x.shape
        if W%stride != 0:
            raise NotImplementedError("W%stride != 0")

        res = None
        for j in range(-(W//stride)+1, (W//stride)):
            out = nngp_zz_fn(i, j, z1, x)
            if res is None:
                res = out
            else:
                res = res + out
        return res
    zx_kernel_fn = jax.jit(zx_kernel_fn, static_argnums=(0,))

    def zz_kernel_fn(i, j, z1, z2):
        assert isinstance(i, int) and isinstance(j, int)
        z1 = jnp.moveaxis(jnp.expand_dims(z1, 0), 1, -1)
        z2 = jnp.moveaxis(jnp.expand_dims(z2, 0), 1, -1)
        return nngp_zz_fn(i, j, z1, z2)
    #zz_kernel_fn = jax.jit(zz_kernel_fn, static_argnums=(0, 1))


    def kern_zz_fn(i, z1, j, z2):
        z1 = jnp.asarray(z1.numpy())
        if z2 is not None:
            z2 = jnp.asarray(z2.numpy())
        else:
            z2 = z1
        return zz_kernel_fn(i, j, z1, z2)

    def kern_zx_fn(i, z1, x2):
        z1 = jnp.asarray(z1.numpy())
        x2 = jnp.asarray(x2.numpy())
        return zx_kernel_fn(i, z1, x2)

    def kern_x_fn(x):
        return _kern_x_fn(jnp.asarray(x.numpy()))

    return kern_zz_fn, kern_zx_fn, kern_x_fn


@experiment.command
def test_kernels():
    train_set, test_set = SU.load_sorted_dataset()
    __kern_zz_fn, __kern_zx_fn, kern_x_fn = interdomain_kernel()
    kern_zz_fn, kern_zx_fn, _ = torch_interdomain_kernel()

    Z = train_set[0][0][:, :5, :5]
    X = Z.unsqueeze(0)

    print("x")
    print(kern_x_fn(X))

    out = None
    _, _, H, W = X.shape
    for i in range(-H+1, H):
        if out is None:
            out = kern_zx_fn(i, Z, X)
        else:
            out = kern_zx_fn(i, Z, X) + out
    print("zx")
    print(out)

    out = None
    for i in range(-4, 5):
        for j in range(-4, 5):
            if out is None:
                out = kern_zz_fn(i, Z, j, Z)
            else:
                out = kern_zz_fn(i, Z, j, Z) + out
    print("zz")
    print(out)

@experiment.automain
def main(N_inducing, batch_size, print_interval, _log):
    train_set, test_set = SU.load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    oh_train_Y = centered_one_hot(train_Y).astype(np.float64)
    test_Y = dataset_targets(test_set)

    kern_zz_fn, _, _ = torch_interdomain_kernel()
    _, kern_zx_fn, _ = interdomain_kernel()
    _, img_h, _ = train_set[0][0].shape

    Kux = np.zeros((N_inducing, len(train_set)), np.float64)
    Kut = np.zeros((N_inducing, len(test_set)), np.float64)
    Kuu = np.zeros((N_inducing, N_inducing), np.float64)
    Kux[...] = np.nan
    Kut[...] = np.nan
    Kuu[...] = np.nan

    Z_X_idx = []
    Z_is = []
    existing_inducing = set()

    timings_obj = PrintTimings(
        desc="sparse_classify",
        print_interval=print_interval)
    timings = timings_obj(
            itertools.count(),
            ((len(train_set) + len(test_set))//batch_size + N_inducing) * N_inducing)


    data_series = pd.Series()
    accuracy_series = pd.Series()

    milestone = 4
    with h5py.File(SU.base_dir()/"kernels.h5", "w") as h5_file:
        h5_file.create_dataset("Kuu", shape=(1, *Kuu.shape), dtype=np.float32,
                               fillvalue=np.nan, chunks=(1, 128, 128),
                               maxshape=(None, *Kuu.shape))
        h5_file.create_dataset("Kux", shape=(1, *Kux.shape), dtype=np.float32,
                               fillvalue=np.nan, chunks=(1, 1, Kux.shape[1]),
                               maxshape=(None, *Kux.shape))
        h5_file.create_dataset("Kut", shape=(1, *Kut.shape), dtype=np.float32,
                               fillvalue=np.nan, chunks=(1, 1, Kut.shape[1]),
                               maxshape=(None, *Kut.shape))

        for step in range(N_inducing):
            # Select new inducing point
            while True:
                Z_i_in_X = np.random.randint(len(train_set))
                Z_i = np.random.randint(-img_h+1, img_h)
                if (Z_i_in_X, Z_i) not in existing_inducing:
                    break
            existing_inducing.add((Z_i_in_X, Z_i))
            Z_X_idx.append(Z_i_in_X)
            Z_is.append(Z_i)

            assert len(Z_is) == len(Z_X_idx) and len(Z_is) == step+1 and len(existing_inducing) == step+1

            current_Z, _ = train_set[Z_X_idx[-1]]
            current_Zi = Z_is[-1]

            timings_obj.desc = f"Updating Kuu (inducing point #{step})"
            for a in range(step+1):
                other_Z, _ = train_set[Z_X_idx[a]]
                Kuu[a, step] = Kuu[step, a] = np.squeeze(kern_zz_fn(
                    Z_is[a], other_Z,
                    current_Zi, current_Z), axis=(0, 1))
                next(timings)
            h5_file["Kuu"][0, step, :step+1] = Kuu[step, :step+1]
            h5_file["Kuu"][0, :step+1, step] = Kuu[:step+1, step]

            timings_obj.desc = f"Updating Kux (inducing point #{step})"
            for slice_start, slice_end, (train_x, _) in zip(
                    itertools.count(start=0, step=batch_size),
                    itertools.count(start=batch_size, step=batch_size),
                    DataLoader(train_set, batch_size=batch_size, shuffle=False)):
                Kux[step, slice_start:slice_end] = np.squeeze(
                    kern_zx_fn(current_Zi, current_Z, train_x), axis=0)
                next(timings)
            h5_file["Kux"][0, step, :] = Kux[step, :]

            timings_obj.desc = f"Updating Kut (inducing point #{step})"
            for slice_start, slice_end, (test_x, _) in zip(
                    itertools.count(start=0, step=batch_size),
                    itertools.count(start=batch_size, step=batch_size),
                    DataLoader(test_set, batch_size=batch_size, shuffle=False)):
                Kut[step, slice_start:slice_end] = np.squeeze(
                    kern_zx_fn(current_Zi, current_Z, test_x), axis=0)
                next(timings)
            h5_file["Kut"][0, step, :] = Kut[step, :]


            if step+1 == milestone or step+1 == N_inducing:
                milestone *= 2
                _log.info(f"Performing classification (n. inducing #{step+1})")
                _Kuu = Kuu[:len(Z_X_idx), :len(Z_X_idx)]
                _Kux = Kux[:len(Z_X_idx)]
                _Kut = Kut[:len(Z_X_idx)]
                assert not np.any(np.isnan(_Kuu))
                assert not np.any(np.isnan(_Kux))
                assert not np.any(np.isnan(_Kut))

                data, accuracy = do_one_N_chol(_Kuu, _Kux, _Kut, oh_train_Y, test_Y,
                                               n_splits=4)
                data_series.loc[step+1] = data
                accuracy_series.loc[step+1] = accuracy
                pd.to_pickle(data_series, SU.base_dir()/"grid_acc.pkl.gz")
                pd.to_pickle(accuracy_series, SU.base_dir()/"accuracy.pkl.gz")
                pd.to_pickle((Z_X_idx, Z_is), SU.base_dir()/"inducing_indices.pkl.gz")

                (sigy, acc) = map(np.squeeze, accuracy)
                _log.info(
                    f"For RBF kernel, N_inducing={step+1}, sigy={sigy}; accuracy={acc}, cv_accuracy={np.max(data[1])}")



