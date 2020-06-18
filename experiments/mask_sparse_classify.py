import collections
from neural_tangents import stax
import contextlib
import faulthandler
import itertools
import math
import sys
import signal
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
from cnn_limits.layers import covariance_tensor, CorrelatedConv
from experiments.predict_cv_acc import (dataset_targets, centered_one_hot,
                                        EigenOut, eigdecompose, accuracy_eig, fold_idx)
from experiments.predict_cv_acc import experiment as predict_cv_acc_experiment
import experiments.sparse_classify
from cnn_limits.sparse import patch_kernel_fn, patch_kernel_fn_torch, InducingPatches, mask_and_start_idx
import gpytorch
from cnn_limits.tbx import PrintTimings

faulthandler.enable()

experiment = sacred.Experiment("mask_sparse_classify", [SU.ingredient, predict_cv_acc_experiment, experiments.sparse_classify.experiment])
if __name__ == '__main__':
    SU.add_file_observer(experiment)

@experiment.config
def config():
    batch_size = 1000
    print_interval = 5.
    model = "CNTK_nopool"
    stride = 1
    model_args = dict(depth=14)
    N_inducing = 15000
    lengthscale = 10.
    inducing_strat = "conditional_variance"
    sample_in_init = False
    inducing_training_multiple = 5
    inducing_start=0
    inducing_end=10000
    do_only_inducing=True
    inducing_list_path="/homes/ag919/Programacio/cnn-limits/for_all_inducing_indices.pkl.gz"


do_one_N_chol = experiments.sparse_classify.do_one_N_chol


@experiment.capture
def interdomain_kernel(img_w, model, model_args, stride, lengthscale):
    no_pool = getattr(cnn_limits.models, model)(channels=1, **model_args)

    if lengthscale is None:
        pool_W_cov = None
    else:
        gpytorch_kern = gpytorch.kernels.MaternKernel(nu=3/2)
        gpytorch_kern.lengthscale = lengthscale

        pool_W_cov = covariance_tensor(32//stride, 32//stride, gpytorch_kern)
        # pool = CorrelatedConv(1, (32//stride, 32//stride), (1, 1), padding='VALID',
        #                       W_cov_tensor=pool_W_cov)
    pool = stax.GlobalAvgPool()

    _, _, _pool_kernel_fn = stax.serial(no_pool, pool, stax.Flatten())
    _, _, _no_pool_kfn = no_pool
    _kern_zz_fn, sum_of_covs = patch_kernel_fn(_no_pool_kfn, (stride, stride), W_cov=pool_W_cov)
    get = ('nngp', 'ntk')
    def kern_zz_fn(z1, start_idx1, mask1, z2, start_idx2, mask2):
        return _kern_zz_fn(z1, start_idx1, mask1, z2, start_idx2, mask2, get=get)

    if sum_of_covs is None:
        offset = 32//stride-1
        sum_of_covs = np.zeros((64//stride-1, 64//stride-1), dtype=np.float32)
        for i in range(sum_of_covs.shape[0]):
            for j in range(sum_of_covs.shape[1]):
                sum_of_covs[i, j] = (32//stride - abs(i-offset))*(32//stride - abs(j-offset))

    def kern_x_fn(x1, x2):
        out = _pool_kernel_fn(x1, x2=x2, get=get)
        return jnp.stack([out.nngp, out.ntk], 0)

    _, _, all_start_idx, all_mask = mask_and_start_idx(
        stride, img_w, range(-(img_w//stride)+1, (img_w//stride)), None, None)
    all_start_idx = jnp.asarray(np.expand_dims(all_start_idx, 1))
    all_mask = jnp.asarray(np.expand_dims(all_mask, 1))
    _zz_batched = jax.vmap(kern_zz_fn, (None, None, None, None, 0, 0), 0)

    def kern_zx_fn(z1, start_idx1, mask1, x2):
        k = _zz_batched(z1, start_idx1, mask1, x2, all_start_idx, all_mask)
        return k.sum(0)

    return tuple(map(jax.jit, (kern_zz_fn, kern_zx_fn, kern_x_fn))), sum_of_covs


@experiment.command
def test_kernels(stride):
    train_set, test_set = SU.load_sorted_dataset()

    X = train_set[0][0].unsqueeze(0).transpose(1, -1).numpy()
    X2 = train_set[1][0].unsqueeze(0).transpose(1, -1).numpy()
    _, _, img_w, _ = X.shape
    (kern_zz_fn, kern_zx_fn, kern_x_fn), _ = interdomain_kernel(img_w)

    print("x")
    print(kern_x_fn(X, X2))

    _, _, all_start_idx, all_mask = mask_and_start_idx(
        stride, img_w, range(-(img_w//stride)+1, (img_w//stride)), None, None)
    Z = jnp.tile(X, (len(all_start_idx), 1, 1, 1))

    rand_i1 = np.arange(len(all_start_idx))
    np.random.shuffle(rand_i1)
    rand_i2 = np.arange(len(all_start_idx))
    np.random.shuffle(rand_i2)

    Kzx = kern_zx_fn(Z, all_start_idx[rand_i1], all_mask[rand_i1], X2)
    print("zx")
    print(jnp.sum(Kzx, -2))

    Z2 = jnp.tile(X2, (len(all_start_idx), 1, 1, 1))
    Kzz = kern_zz_fn(Z, all_start_idx[rand_i1], all_mask[rand_i1], Z2, all_start_idx[rand_i2], all_mask[rand_i2])
    print("zz")
    print(jnp.sum(Kzz, (-2, -1)))


@experiment.capture
def _conditional_variance_initialisation(inducing, inducing_training_set, kern_zz_fn, N_inducing, sample_in_init, batch_size, inducing_training_multiple, _log, timings=None):
    """
    Reference: TODO add several references,
    TODO: IF M ==1 this throws errors, currently throws an assertion error, but should fix
    Initializes based on variance of noiseless GP fit on inducing points currently in active set
    Complexity: O(NM) memory, O(NM^2) time
    :param training_inputs: [N,D] numpy array,
    :param M: int, number of points desired. If threshold is None actual number returned may be less than M
    :param kernel: kernelwrapper object
    :return: inducing inputs, indices,
    [M,D] np.array to use as inducing inputs,  [M], np.array of ints indices of these inputs in training data array
    """
    M = N_inducing
    assert M > 1
    N = inducing_training_set.Z.shape[0]
    true_len = N // inducing_training_multiple
    _log.info(f"Length of training set: N={N}, true_len={true_len}")

    perm = np.random.permutation(N)  # permute entries so tiebreaking is random

    training_inputs = jnp.asarray(inducing_training_set.Z[perm])
    start_idx = jnp.asarray(inducing_training_set.start_idx[perm])
    mask = jnp.asarray(inducing_training_set.mask[perm])

    # note this will throw an out of bounds exception if we do not update each entry
    indices = np.zeros(M, dtype=int) + N
    diag_kern_zz_fn = jax.vmap(kern_zz_fn)
    di = np.zeros((N,), dtype=np.float64)
    di[...] = np.nan
    for b_slice in batch_slices(N, batch_size):
        di[b_slice] = np.squeeze(diag_kern_zz_fn(
            training_inputs[b_slice, None, ...], start_idx[b_slice, None, ...], mask[b_slice, None, ...],
            training_inputs[b_slice, None, ...], start_idx[b_slice, None, ...], mask[b_slice, None, ...]), (1, 2))
    assert not np.any(np.isnan(di))

    L = np.zeros((N,), dtype=np.float64)

    if sample_in_init:
        indices[0] = sample_discrete(di)
    else:
        indices[0] = np.argmax(di)  # select first point, add to index 0
    ci = np.zeros((M - 1, N))  # [M,N]
    for m in range(M - 1):
        j = int(indices[m])  # int
        yield perm[j]%true_len, inducing_training_set.i[perm[j]]
        new_Z = training_inputs[j:j + 1]  # [1,D]
        new_start_idx = start_idx[j:j+1]
        new_mask = mask[j:j+1]
        dj = np.sqrt(di[j])  # float
        cj = ci[:m, j]  # [m, 1]
        for b_slice in batch_slices(N, 5*batch_size):
            L[b_slice] = np.squeeze(kern_zz_fn(
                training_inputs[b_slice], start_idx[b_slice], mask[b_slice], new_Z, new_start_idx, new_mask), axis=1)
            if timings is not None:
                next(timings)
        ei = (L - np.dot(cj, ci[:m])) / dj
        ci[m, :] = ei
        di -= ei ** 2
        if sample_in_init:
            indices[m + 1] = sample_discrete(di)
        else:
            indices[m + 1] = np.argmax(di)  # select first point, add to index 0
        _log.info(f"Remaining variance: sum(di)={np.sum(np.clip(di, 0, None))}")
        # sum of di is tr(Kff-Qff), if this is small things are ok
        # if np.sum(np.clip(di, 0, None)) < self.threshold:
        #     indices = indices[:m]
        #     warnings.warn("ConditionalVariance: Terminating selection of inducing points early.")
        #     break
    j = int(indices[-1])
    yield perm[j]%true_len, inducing_training_set.i[perm[j]]


@experiment.capture
def _random_strategy(inducing, inducing_training_set, stride, N_inducing):
    existing_inducing = set()
    max_rint, img_h, _, _ = inducing_training_set.Z.shape
    while len(existing_inducing) < N_inducing:
        Z_i_in_X = np.random.randint(max_rint)
        Z_i = np.random.randint(-(img_h//stride)+1, (img_h//stride))
        if (Z_i_in_X, Z_i) not in existing_inducing:
            yield (Z_i_in_X, Z_i)
            existing_inducing.add((Z_i_in_X, Z_i))


@experiment.capture
def inducing_point_indices(inducing, inducing_training_set, kern_zz_fn, inducing_strat):
    if inducing_strat == "random":
        return _random_strategy(inducing, inducing_training_set)
    elif inducing_strat == "conditional_variance":
        return _conditional_variance_initialisation(inducing, inducing_training_set, kern_zz_fn)
    else:
        raise ValueError(inducing_strat)


def draw_patch_i(n_samples, p):
    assert np.all(p>=0)
    assert p.shape[0] == p.shape[1], "This might break if p is not square"
    idx = np.random.choice(p.shape[0], n_samples, replace=True, p=p.sum(1)/p.sum())
    return idx-p.shape[0]//2


def batch_slices(N, batch_size):
    return (slice(slice_start, slice_end) for slice_start, slice_end in zip(
        range(0, N, batch_size),
        itertools.count(start=batch_size, step=batch_size)))

@experiment.command
def select_inducing_points(N_inducing, batch_size, print_interval, stride, _log, inducing_strat, inducing_training_multiple):
    train_set, test_set = SU.load_sorted_dataset()
    img_c, img_h, img_w = train_set[0][0].shape
    (kern_zz_fn, kern_zx_fn, _), sum_of_covs = interdomain_kernel(img_w)

    inducing_i = draw_patch_i(N_inducing, sum_of_covs)
    inducing_Z = np.random.permutation(50000)[:N_inducing]
    print(inducing_i, inducing_Z)
    pd.to_pickle((inducing_i, inducing_Z), SU.base_dir()/"inducing_indices.pkl.gz")

@experiment.command
def save_inducing_points(N_inducing, batch_size, print_interval, stride, _log, inducing_strat, inducing_training_multiple, do_only_inducing, inducing_start, inducing_end, inducing_list_path):
    train_set, test_set = SU.load_sorted_dataset()
    img_c, img_h, img_w = train_set[0][0].shape
    (kern_zz_fn, kern_zx_fn, _), sum_of_covs = interdomain_kernel(img_w)

    train_x = np.concatenate([
        _train_x.transpose(1, -1).to(torch.float32).numpy()
        for (_train_x, _) in DataLoader(train_set, batch_size=batch_size, shuffle=False)], 0)

    test_x = np.concatenate([
        _test_x.transpose(1, -1).to(torch.float32).numpy()
        for (_test_x, _) in DataLoader(test_set, batch_size=batch_size, shuffle=False)], 0)

    inducing_i, inducing_Z = pd.read_pickle(inducing_list_path)

    inducing = InducingPatches(
        train_x[inducing_Z],
        # np.zeros((N_inducing, img_h, img_w, img_c), np.float32), # 50 MB
        list(map(int, inducing_i)),
        *mask_and_start_idx(stride, img_h, inducing_i, None, None)[2:])
    assert inducing.Z.shape == (N_inducing, img_h, img_w, img_c)

    timings_obj = PrintTimings(
        desc="sparse_classify",
        print_interval=print_interval)

    with h5py.File(SU.base_dir()/"kernels.h5", "w") as h5_file:
        if do_only_inducing:
            h5_file.create_dataset("Kuu", shape=(4, N_inducing, N_inducing), dtype=np.float32,
                                fillvalue=np.nan, chunks=(1, 128, 128),
                                maxshape=(None, N_inducing, N_inducing))
            timings = timings_obj(
                itertools.count(),
                (N_inducing//batch_size)**2)
            print("DOING ONLY INDUCING")
            for b_slice_x in batch_slices(N_inducing, batch_size):
                for b_slice_y in batch_slices(b_slice_x.stop, batch_size):
                    h5_file["Kuu"][:, b_slice_x, b_slice_y] = kern_zz_fn(
                    inducing.Z[b_slice_x], inducing.start_idx[b_slice_x],
                    inducing.mask[b_slice_x],
                    inducing.Z[b_slice_y], inducing.start_idx[b_slice_y],
                    inducing.mask[b_slice_y])
                    next(timings)
        else:
            h5_file.create_dataset("Kux", shape=(4, N_inducing, len(train_set)), dtype=np.float32,
                                   fillvalue=np.nan, chunks=(1, 1, len(train_set)),
                                   maxshape=(None, N_inducing, len(train_set)))
            h5_file.create_dataset("Kut", shape=(4, N_inducing, len(test_set)), dtype=np.float32,
                                   fillvalue=np.nan, chunks=(1, 1, len(test_set)),
                                   maxshape=(None, N_inducing, len(test_set)))
            timings = timings_obj(
                itertools.count(),
                ((inducing_end-inducing_start)*(len(train_set) + len(test_set))//batch_size))

            for i in range(inducing_start, inducing_end):
                current_Z = slice(i, i+1)

                for b_slice in batch_slices(len(train_set), batch_size):
                    h5_file["Kux"][:, current_Z, b_slice] = kern_zx_fn(
                        inducing.Z[current_Z], inducing.start_idx[current_Z],
                        inducing.mask[current_Z], train_x[b_slice])
                    next(timings)
                for b_slice in batch_slices(len(test_set), batch_size):
                    h5_file["Kut"][:, current_Z, b_slice] = kern_zx_fn(
                        inducing.Z[current_Z], inducing.start_idx[current_Z],
                        inducing.mask[current_Z], test_x[b_slice])
                    next(timings)


@experiment.command
def classify(kernel_matrix_path, _log, i_SU):
    assert i_SU["dataset_treatment"] == "no_treatment"
    train_set, test_set = SU.load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    oh_train_Y = centered_one_hot(train_Y).astype(np.float64)
    test_Y = dataset_targets(test_set)

    with h5py.File(kernel_matrix_path, 'r') as f:
        mask_Kux = ~(np.isnan(f["Kux"][0, ...]).any(-1))
        mask_Kut = ~(np.isnan(f["Kut"][0, ...]).any(-1))
        mask = mask_Kux & mask_Kut

        for i in range(f["Kuu"].shape[0]):
            Kuu = f["Kuu"][i, mask, :][:, mask].astype(np.float64)
            Kuu_mask = np.triu(np.ones(Kuu.shape, dtype=bool))
            Kuu[Kuu_mask] = Kuu.T[Kuu_mask]
            Kux = f["Kux"][i, mask, :].astype(np.float64)
            Kut = f["Kut"][i, mask, :].astype(np.float64)
            assert not np.any(np.isnan(Kuu))
            assert not np.any(np.isnan(Kux))
            assert not np.any(np.isnan(Kut))

            data, accuracy = do_one_N_chol(Kuu, Kux, Kut, oh_train_Y, test_Y, n_splits=4)
            (sigy, acc) = map(np.squeeze, accuracy)
            _log.info(
                f"For i={i}, N_inducing={Kuu.shape[0]}, sigy={sigy}; accuracy={acc}, cv_accuracy={np.max(data[1])}")



@experiment.automain
def main(N_inducing, batch_size, print_interval, stride, _log, inducing_strat, inducing_training_multiple):
    train_set, test_set = SU.load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    oh_train_Y = centered_one_hot(train_Y).astype(np.float64)
    test_Y = dataset_targets(test_set)

    img_c, img_h, img_w = train_set[0][0].shape
    (kern_zz_fn, kern_zx_fn, _), sum_of_covs = interdomain_kernel(img_w)

    Kux = np.zeros((N_inducing, len(train_set)), np.float64)
    Kut = np.zeros((N_inducing, len(test_set)), np.float64)
    Kuu = np.zeros((N_inducing, N_inducing), np.float64)
    Kux[...] = np.nan
    Kut[...] = np.nan
    Kuu[...] = np.nan

    inducing = InducingPatches(
        Z=np.zeros((N_inducing, img_h, img_w, img_c), np.float32), # 50 MB
        i=[],
        start_idx=np.zeros((N_inducing, 2), dtype=int),
        mask=np.zeros((N_inducing, img_h), dtype=bool))
    Z_X_idx = []

    timings_obj = PrintTimings(
        desc="sparse_classify",
        print_interval=print_interval)
    timings = timings_obj(
        itertools.count(),
        ((len(train_set) + len(test_set))//batch_size * N_inducing))

    data_series = pd.Series()
    accuracy_series = pd.Series()

    signalling = {'SIGHUP': False}
    def sighup_signal_handler(sig, frame):
        _log.info("Received SIGHUP")
        signalling['SIGHUP'] = True
    signal.signal(signal.SIGHUP, sighup_signal_handler)

    train_x = np.concatenate([
        _train_x.transpose(1, -1).to(torch.float32).numpy()
        for (_train_x, _) in DataLoader(train_set, batch_size=batch_size, shuffle=False)], 0)

    test_x = np.concatenate([
        _test_x.transpose(1, -1).to(torch.float32).numpy()
        for (_test_x, _) in DataLoader(test_set, batch_size=batch_size, shuffle=False)], 0)

    if inducing_strat == "random":
        inducing_training_set = InducingPatches(train_x, None, None, None)
    else:
        inducing_training_set_i = draw_patch_i(inducing_training_multiple*len(train_x), sum_of_covs)
        np.save(SU.base_dir()/"inducing_training_set_i.npy", inducing_training_set_i)
        inducing_training_set = InducingPatches(
            np.tile(train_x, (inducing_training_multiple, 1, 1, 1)),  # Z=
            inducing_training_set_i,  # i=
            *mask_and_start_idx(stride, img_h, inducing_training_set_i, None, None)[2:])

    milestone = 4
    with h5py.File(SU.base_dir()/"kernels.h5", "w") as h5_file:
        h5_file.create_dataset("Kuu", shape=(1, *Kuu.shape), dtype=np.float64,
                               fillvalue=np.nan, chunks=(1, 128, 128),
                               maxshape=(None, *Kuu.shape))
        h5_file.create_dataset("Kux", shape=(1, *Kux.shape), dtype=np.float64,
                               fillvalue=np.nan, chunks=(1, 1, Kux.shape[1]),
                               maxshape=(None, *Kux.shape))
        h5_file.create_dataset("Kut", shape=(1, *Kut.shape), dtype=np.float64,
                               fillvalue=np.nan, chunks=(1, 1, Kut.shape[1]),
                               maxshape=(None, *Kut.shape))

        for step, (Z_i_in_X, Z_i) in enumerate(inducing_point_indices(
                inducing, inducing_training_set, kern_zz_fn)):
            # Select new inducing point
            current_Z = slice(step, step+1)
            inducing.Z[step] = torch.transpose(
                train_set[Z_i_in_X][0], 0, -1).numpy()
            inducing.i.append(Z_i)
            Z_X_idx.append(Z_i_in_X)
            mask_and_start_idx(stride, img_h, inducing.i[current_Z],
                               out_start_idx=inducing.start_idx[current_Z],
                               out_mask=inducing.mask[current_Z])

            _log.info(f"Updating Kuu (inducing point #{step})")
            for b_slice in batch_slices(step+1, batch_size//10):
                _end = min(b_slice.stop, step+1)
                Kuu[step, b_slice.start:_end] = Kuu[b_slice.start:_end, step] = np.squeeze(kern_zz_fn(
                    inducing.Z[current_Z], inducing.start_idx[current_Z],
                    inducing.mask[current_Z],
                    inducing.Z[b_slice], inducing.start_idx[b_slice],
                    inducing.mask[b_slice],
                ), axis=0)[:_end - b_slice.start]
            h5_file["Kuu"][0, step, :step+1] = Kuu[step, :step+1]
            h5_file["Kuu"][0, :step+1, step] = Kuu[:step+1, step]

            timings_obj.desc = f"Updating Kux (inducing point #{step})"
            for b_slice in batch_slices(len(train_set), batch_size):
                Kux[step, b_slice] = np.squeeze(kern_zx_fn(
                    inducing.Z[current_Z], inducing.start_idx[current_Z],
                    inducing.mask[current_Z], train_x[b_slice]), axis=0)
                next(timings)
            h5_file["Kux"][0, step, :] = Kux[step, :]

            timings_obj.desc = f"Updating Kut (inducing point #{step})"
            for b_slice in batch_slices(len(test_set), batch_size):
                Kut[step, b_slice] = np.squeeze(kern_zx_fn(
                    inducing.Z[current_Z], inducing.start_idx[current_Z],
                    inducing.mask[current_Z], test_x[b_slice]), axis=0)
                next(timings)
            h5_file["Kut"][0, step, :] = Kut[step, :]


            if step+1 == milestone or step+1 == N_inducing or signalling['SIGHUP']:
                signalling['SIGHUP'] = False
                milestone = min(milestone*2, milestone+512)
                _log.info(f"Performing classification (n. inducing #{step+1})")
                _Kuu = Kuu[:len(inducing.i), :len(inducing.i)]
                _Kux = Kux[:len(inducing.i)]
                _Kut = Kut[:len(inducing.i)]
                assert not np.any(np.isnan(_Kuu))
                assert not np.any(np.isnan(_Kux))
                assert not np.any(np.isnan(_Kut))

                data, accuracy = do_one_N_chol(_Kuu, _Kux, _Kut, oh_train_Y, test_Y,
                                               n_splits=4)
                data_series.loc[step+1] = data
                accuracy_series.loc[step+1] = accuracy
                pd.to_pickle(data_series, SU.base_dir()/"grid_acc.pkl.gz")
                pd.to_pickle(accuracy_series, SU.base_dir()/"accuracy.pkl.gz")
                pd.to_pickle((Z_X_idx, inducing.i), SU.base_dir()/"inducing_indices.pkl.gz")

                (sigy, acc) = map(np.squeeze, accuracy)
                _log.info(
                    f"For N_inducing={step+1}, sigy={sigy}; accuracy={acc}, cv_accuracy={np.max(data[1])}")



