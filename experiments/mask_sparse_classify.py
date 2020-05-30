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
import experiments.sparse_classify
from cnn_limits.sparse import patch_kernel_fn, patch_kernel_fn_torch, InducingPatches, mask_and_start_idx
from nigp.tbx import PrintTimings

faulthandler.enable()

experiment = sacred.Experiment("mask_sparse_classify", [SU.ingredient, predict_cv_acc_experiment, experiments.sparse_classify.experiment])
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

do_one_N_chol = experiments.sparse_classify.do_one_N_chol


@experiment.capture
def interdomain_kernel(img_w, model, model_args, stride):
    no_pool = getattr(cnn_limits.models, model)(channels=1, **model_args)

    _, _, _pool_kernel_fn = stax.serial(no_pool, stax.GlobalAvgPool(), stax.Flatten())
    _, _, _no_pool_kfn = no_pool
    kern_zz_fn = patch_kernel_fn(_no_pool_kfn, (stride, stride), W_cov=None)
    kern_zz_fn = jax.jit(kern_zz_fn)

    @jax.jit
    def kern_x_fn(x1):
        return _pool_kernel_fn(x1, x2=None, get='var1')

    _, _, all_start_idx, all_mask = mask_and_start_idx(
        stride, img_w, range(-(img_w//stride)+1, (img_w//stride)), None, None)
    all_start_idx = jnp.asarray(np.expand_dims(all_start_idx, 1))
    all_mask = jnp.asarray(np.expand_dims(all_mask, 1))
    _zz_batched = jax.vmap(kern_zz_fn, (None, None, None, None, 0, 0), 2)

    @jax.jit
    def kern_zx_fn(z1, start_idx1, mask1, x2):
        k = _zz_batched(z1, start_idx1, mask1, x2, all_start_idx, all_mask)
        return k.sum(2)

    return kern_zz_fn, kern_zx_fn, kern_x_fn


@experiment.command
def test_kernels(stride):
    train_set, test_set = SU.load_sorted_dataset()

    X = train_set[0][0].unsqueeze(0).transpose(1, -1).numpy()
    _, _, img_w, _ = X.shape
    kern_zz_fn, kern_zx_fn, kern_x_fn = interdomain_kernel(img_w)

    print("x")
    print(kern_x_fn(X))

    _, _, all_start_idx, all_mask = mask_and_start_idx(
        stride, img_w, range(-(img_w//stride)+1, (img_w//stride)), None, None)
    Z = jnp.tile(X, (len(all_start_idx), 1, 1, 1))

    Kzx = kern_zx_fn(Z, all_start_idx, all_mask, X)
    print("zx")
    print(jnp.sum(Kzx, 0))

    Kzz = kern_zz_fn(Z, all_start_idx, all_mask, Z, all_start_idx, all_mask)
    print("zz")
    print(jnp.sum(Kzz, (0, 1)))


@experiment.automain
def main(N_inducing, batch_size, print_interval, stride, _log):
    train_set, test_set = SU.load_sorted_dataset()
    train_Y = dataset_targets(train_set)
    oh_train_Y = centered_one_hot(train_Y).astype(np.float64)
    test_Y = dataset_targets(test_set)

    img_c, img_h, img_w = train_set[0][0].shape
    kern_zz_fn, kern_zx_fn, _ = interdomain_kernel(img_w)

    Kux = np.zeros((N_inducing, len(train_set)), np.float64)
    Kut = np.zeros((N_inducing, len(test_set)), np.float64)
    Kuu = np.zeros((N_inducing, N_inducing), np.float64)
    Kux[...] = np.nan
    Kut[...] = np.nan
    Kuu[...] = np.nan

    inducing = InducingPatches(
        Z=np.zeros((N_inducing, img_h, img_w, img_c), np.float32), # 50 MB
        i=[],
        start_idx=np.zeros((N_inducing, 2)),
        mask=np.zeros((N_inducing, img_h), dtype=bool))
    existing_inducing = set()

    timings_obj = PrintTimings(
        desc="sparse_classify",
        print_interval=print_interval)
    timings = timings_obj(
        itertools.count(),
        ((len(train_set) + len(test_set))//batch_size * N_inducing))

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
                Z_i = np.random.randint(-(img_h//stride)+1, (img_h//stride))
                if (Z_i_in_X, Z_i) not in existing_inducing:
                    break
            existing_inducing.add((Z_i_in_X, Z_i))
            current_Z = slice(step, step+1)
            inducing.Z[step] = torch.transpose(
                train_set[Z_i_in_X][0], 1, -1).numpy()
            inducing.i.append(Z_i)
            mask_and_start_idx(stride, img_h, inducing.i[current_Z],
                               out_start_idx=inducing.start_idx[current_Z],
                               out_mask=inducing.mask[current_Z])

            _log.info("Updating Kuu (inducing point #{step})")
            Kuu[step, :step+1] = Kuu[:step+1, step] = np.squeeze(kern_zz_fn(
                inducing.Z[current_Z], inducing.start_idx[current_Z],
                inducing.mask[current_Z],
                inducing.Z[:step+1], inducing.start_idx[:step+1],
                inducing.mask[:step+1],
            ), axis=0)
            h5_file["Kuu"][0, step, :step+1] = Kuu[step, :step+1]
            h5_file["Kuu"][0, :step+1, step] = Kuu[:step+1, step]

            timings_obj.desc = f"Updating Kux (inducing point #{step})"
            for slice_start, slice_end, (train_x, _) in zip(
                    itertools.count(start=0, step=batch_size),
                    itertools.count(start=batch_size, step=batch_size),
                    DataLoader(train_set, batch_size=batch_size, shuffle=False)):
                Kux[step, slice_start:slice_end] = np.squeeze(kern_zx_fn(
                    inducing.Z[current_Z], inducing.start_idx[current_Z],
                    inducing.mask[current_Z],
                    train_x.transpose(1, -1).to(torch.float32).numpy()), axis=0)
                next(timings)
            h5_file["Kux"][0, step, :] = Kux[step, :]

            timings_obj.desc = f"Updating Kut (inducing point #{step})"
            for slice_start, slice_end, (test_x, _) in zip(
                    itertools.count(start=0, step=batch_size),
                    itertools.count(start=batch_size, step=batch_size),
                    DataLoader(test_set, batch_size=batch_size, shuffle=False)):
                Kux[step, slice_start:slice_end] = np.squeeze(kern_zx_fn(
                    inducing.Z[current_Z], inducing.start_idx[current_Z],
                    inducing.mask[current_Z],
                    test_x.transpose(1, -1).to(torch.float32).numpy()), axis=0)
                next(timings)
            h5_file["Kut"][0, step, :] = Kut[step, :]


            if step+1 == milestone or step+1 == N_inducing:
                milestone = min(milestone*2, milestone+512)
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
                    f"For N_inducing={step+1}, sigy={sigy}; accuracy={acc}, cv_accuracy={np.max(data[1])}")



