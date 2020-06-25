"""
Save a kernel matrix to disk
"""
import itertools
import os
import pickle

import h5py
import jax
import jax.numpy as np
import sacred
import numpy as onp
from torch.utils.data import DataLoader, Subset
import torch
import jug

import cnn_limits.models
import cnn_limits.sacred_utils as SU
from cnn_limits.iteration import (DiagIterator, ProductIterator,
                                  create_h5py_dataset, PrintTimings)

experiment = sacred.Experiment("save_sweep", [SU.ingredient])


@experiment.config
def config():
    batch_size = 5

    chunk_size = 10
    model = "CNTK14_split_cpu"

    model_kwargs = {}
    do_Kxt = True
    do_Kxx = True

    N_train = None
    N_test = None


def jax_model(model, model_kwargs):
    print("loading model", model)
    return getattr(cnn_limits.models, model)(channels=1, **model_kwargs)


def jitted_kernel_fn(kernel_fn, W_covs):
    if W_covs is not None:
        W_covs = onp.stack(W.ravel() for W in W_covs)
    W_covs = np.asarray(W_covs, dtype=np.float64)
    def kern_(x1, x2, same, diag):
        assert not diag
        x1 = np.moveaxis(x1, 1, -1)
        x2 = (None if same else np.moveaxis(x2, 1, -1))
        y = kernel_fn(x1, x2, get=('nngp', 'ntk'))
        if not isinstance(y, list):
            y = [y]
        outs = []
        for _y in y:
            outs = outs + [_y.nngp, _y.ntk]
        outs = np.stack(outs, axis=0)
        assert outs.dtype == np.float32

        if W_covs is None:
            return outs
        outs = outs.astype(np.float64)
        arr = outs.reshape((-1, W_covs.shape[-1], 1))
        k = np.dot(W_covs, arr)
        k = k.reshape((W_covs.shape[0] * outs.shape[0], *outs.shape[1:3]))
        return k
    kern_ = jax.jit(kern_, static_argnums=(2, 3))
    dtype = onp.float32
    def kern(x1, x2, same, diag):
        x1 = np.asarray(x1.numpy()).astype(dtype)
        x2 = (None if same else np.asarray(x2.numpy()).astype(dtype))
        return kern_(x1, x2, same, diag)
    return kern

@experiment.command
def test_kernels(model, model_kwargs):
    train_set, test_set = SU.load_sorted_dataset()
    (_, _, kernel_fn), W_covs = jax_model(model, model_kwargs)
    all_kern = jitted_kernel_fn(kernel_fn, W_covs)
    import torch
    X = torch.stack([train_set[0][0], train_set[1][0]], 0)
    k1 = all_kern(X, None, True, False)

    (_, _, kernel_fn) = cnn_limits.models.Myrtle10_base(pooling=False)
    dense_kern = jitted_kernel_fn(kernel_fn, None)
    k2 = dense_kern(X, None, True, False)
    assert np.allclose(k1[0, :, :], k2[0])
    assert np.allclose(k1[0, :, :] + k1[1, :, :], k2[1])

    (_, _, kernel_fn) = cnn_limits.models.Myrtle10_base(pooling=True)
    meanpool_kern = jitted_kernel_fn(kernel_fn, None)
    k3 = meanpool_kern(X, None, True, False)
    assert np.allclose(k1[-2, :, :], k3[-2])
    assert np.allclose(k1[-2, :, :] + k1[-1, :, :], k3[-1])


@jug.TaskGenerator
def kern_save(model, model_kwargs, dataset_name, dataset_base_path,
              train_test_idx, chunk_size, batch_size, dset_names, i, j,
              lower_triangular):

    (_, _, kernel_fn), W_covs = jax_model(model, model_kwargs)
    kernel_fn = jitted_kernel_fn(kernel_fn, W_covs)
    del W_covs

    train_set, test_set = SU.load_dataset(dataset_name, dataset_base_path)
    sets = {"train": train_set, "test": test_set}
    idxs = {"train": train_test_idx[0], "test": train_test_idx[1]}

    X1, X2 = (
        Subset(sets[name], idxs[name][_i:_i+chunk_size])
        for name, _i in zip(dset_names, (i, j)))

    if lower_triangular:
        it = ProductIterator(batch_size, X1, None)
    else:
        it = ProductIterator(batch_size, X1, X2)
    timings = PrintTimings(print_interval=5.)

    out = None
    for same, (i, (x, _y)), (j, (x2, _y2)) in timings(it):
        k = kernel_fn(x, x2, same, False)
        if out is None:
            out = onp.empty((k.shape[0], chunk_size, chunk_size), dtype=k.dtype)
            out[...] = onp.nan
        out[:, i:i+len(x), j:j+len(x2)] = k
    return out


@jug.TaskGenerator
def save_hdf5(path, **kwargs):
    with h5py.File(path, "w") as f:
        for name, (shape, chunks) in kwargs.items():
            if len(chunks) == 0:
                pass
            (_, _, k), *_ = chunks
            f.create_dataset(
                name, dtype=k.dtype, fillvalue=onp.nan,
                shape=(k.shape[0], *shape),
                chunks=(1, *shape),
                maxshape=(None, *shape))
            for i, j, chunk in chunks:
                f[name][:, i:i+chunk.shape[1], j:j+chunk.shape[2]] = chunk


@jug.TaskGenerator
def train_test_idx(dataset_name, dataset_base_path, N_train, N_test, _seed):
    train_set, test_set = SU.load_dataset(dataset_name, dataset_base_path)
    N_train = (len(train_set) if N_train is None else N_train)
    N_test = (len(test_set) if N_test is None else N_test)

    torch.manual_seed(_seed)
    train_idx = SU.class_balanced_train_idx(train_set, N_train)
    test_idx = SU.class_balanced_train_idx(test_set, N_test)

    return train_idx, test_idx


@experiment.main
def main(chunk_size, do_Kxt, do_Kxx, N_train, N_test, i_SU, _seed, batch_size, model, model_kwargs):
    _ret_train_test_idx = train_test_idx(
        i_SU['dataset_name'], i_SU['dataset_base_path'], N_train, N_test,
        _seed)

    def kernel_save_fn(dset_names, i, j, lower_triangular):
        return kern_save(model=model,
                         model_kwargs=model_kwargs,
                         dataset_name=i_SU['dataset_name'],
                         dataset_base_path=i_SU['dataset_base_path'],
                         train_test_idx=_ret_train_test_idx,
                         chunk_size=chunk_size,
                         batch_size=batch_size,
                         dset_names=dset_names,
                         i=i, j=j,
                         lower_triangular=lower_triangular)

    jug.barrier()
    len_train_idx, len_test_idx = map(len, _ret_train_test_idx.value())
    print(f"len(train_idx)={len_train_idx}, len(test_idx)={len_test_idx}")

    Kxx_chunks = []
    Kxt_chunks = []
    for i in range(0, len_train_idx, chunk_size):
        if do_Kxx:
            for j in range(0, i+1, chunk_size):
                value = kernel_save_fn(("train", "train"), i, j, (i==j))
                Kxx_chunks.append((i, j, value))

        if do_Kxt:
            for j in range(0, len_test_idx, chunk_size):
                value = kernel_save_fn(("train", "test"), i, j, False)
                Kxt_chunks.append((i, j, value))

    return save_hdf5(
        SU.base_dir()/"kernels.h5",
        Kxx=((len_train_idx, len_train_idx), Kxx_chunks),
        Kxt=((len_train_idx, len_test_idx), Kxt_chunks))


# Important that this be outside of an `if __name__ == "__main__"`
# so it gets called when the module is imported.
experiment.run_commandline()

