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

import cnn_limits.models
import cnn_limits.sacred_utils as SU
from cnn_gp import DiagIterator, ProductIterator, create_h5py_dataset
from cnn_limits.tbx import PrintTimings

experiment = sacred.Experiment("save_new", [SU.ingredient])
if __name__ == '__main__':
    SU.add_file_observer(experiment)


@experiment.config
def config():
    batch_size = 5

    n_workers = 1
    worker_rank = 0
    print_interval = 2.
    model = "google_NNGP"

    use_ntk = False
    save_variance = False
    internal_lengthscale = None
    skip_iterations = 0
    Kxx_mask_path = None
    do_Kxt = True
    do_Kxx = True


@experiment.capture
def kern_iterator(f, kern_name, X, X2, diag, batch_size, worker_rank, n_workers, i_SU):
    if kern_name in f.keys():
        print("Skipping {} (group exists)".format(kern_name))
        return
    N = len(X)
    N2 = N if X2 is None else len(X2)
    out = create_h5py_dataset(f, batch_size, kern_name, diag, N, N2, dtype=getattr(onp, i_SU["default_dtype"]))

    if diag:
        # Don't split the load for diagonals, they are cheap
        it = DiagIterator(batch_size, X, X2)
    else:
        it = ProductIterator(batch_size, X, X2, worker_rank=worker_rank,
                             n_workers=n_workers)
    return iter(it), out

def schedule_kernel(kernel_fn, iterate, diag, mask):
    same, (i, (x, _y)), (j, (x2, _y2)) = iterate
    if mask is not None and not np.any(mask[i:i+x.shape[0], j:j+x2.shape[0]]):
        return iterate, None
    return iterate, kernel_fn(x, x2, same, diag)


def kern_save(iterate, k_in, W_covs, out, diag, mask):
    same, (i, (x, _y)), (j, (x2, _y2)) = iterate
    if mask is not None and not np.any(mask[i:i+x.shape[0], j:j+x2.shape[0]]):
        return
    # k = W_covs @ onp.asarray(k_in, dtype=np.float64).reshape((-1, W_covs.shape[1])).T
    # k = k.reshape((W_covs.shape[0], len(x), len(x2)))
    k = k_in
    s = k.shape
    assert k.dtype == np.float64
    try:
        if len(s) == 2:
            out[:s[0], i:i+s[1]] = k
        else:
            out[:s[0], i:i+s[1], j:j+s[2]] = k
    except TypeError:
        out.resize(k.shape[0], axis=0)
        return kern_save(iterate, k_in, W_covs, out, diag, mask)


@experiment.command
def generate_sorted_dataset_idx(sorted_dataset_path):
    train_set, test_set = SU.load_dataset()
    os.makedirs(sorted_dataset_path, exist_ok=True)
    train_idx = cnn_limits.interlaced_argsort(train_set)
    with SU.new_file(os.path.join(sorted_dataset_path, "train.pkl")) as f:
        pickle.dump(train_idx, f)
    test_idx = cnn_limits.interlaced_argsort(test_set)
    with SU.new_file(os.path.join(sorted_dataset_path, "test.pkl")) as f:
        pickle.dump(test_idx, f)


## JAX Model
@experiment.capture
def jax_model(model, internal_lengthscale):
    print("loading model", model)
    if model in cnn_limits.models.need_internal_lengthscale:
        return getattr(cnn_limits.models, model)(internal_lengthscale, channels=1)
    return getattr(cnn_limits.models, model)(channels=1)


@experiment.capture
def jitted_kernel_fn(kernel_fn, use_ntk):
    def kern_(x1, x2, same, diag):
        if diag:
            get = "var1"
        elif use_ntk:
            get = "ntk"
        else:
            get = "nngp"
        x1 = np.moveaxis(x1, 1, -1)
        x2 = (None if same else np.moveaxis(x2, 1, -1))
        y = kernel_fn(x1, x2, get=get)
        if isinstance(y, list):
            return np.stack(y)
        return np.expand_dims(y, 0)
    kern_ = jax.jit(kern_, static_argnums=(2, 3))
    # dtype = getattr(onp, i_SU["default_dtype"])
    dtype = onp.float32
    def kern(x1, x2, same, diag):
        x1 = np.asarray(x1.numpy()).astype(dtype)
        x2 = (None if same else np.asarray(x2.numpy()).astype(dtype))
        return kern_(x1, x2, same, diag)
    return kern


@experiment.command
def wrong_zca_dataset():
    train_set, test_set = SU.load_dataset()
    train_set, test_set, W = SU.do_transforms(train_set, test_set)
    X, y = SU.whole_dset(train_set)
    Xt, yt = SU.whole_dset(test_set)
    np.savez("/scratch/ag919/datasets/CIFAR10_ZCA_wrong.npz", X=X, y=y, Xt=Xt, yt=yt, W=W)


@experiment.main
def main(worker_rank, print_interval, n_workers, save_variance, skip_iterations, Kxx_mask_path, do_Kxt, do_Kxx):
    train_set, test_set = SU.load_sorted_dataset()
    _, _, kernel_fn = jax_model()
    kern = jitted_kernel_fn(kernel_fn)
    W_covs = None
    print(f"len(train_set)={len(train_set)}")
    print(f"len(test_set)={len(test_set)}")

    if Kxx_mask_path is None:
        Kxx_mask = None
    else:
        Kxx_mask = np.load(Kxx_mask_path)

    with h5py.File(SU.base_dir()/"kernels.h5", "w") as f:
        timings_obj = PrintTimings(
            desc=f"Kxx&Kxt (worker {worker_rank}/{n_workers})",
            print_interval=print_interval)

        if save_variance and worker_rank == 0:
            Kx_diag, Kx_diag_out = kern_iterator(
                f, kern_name="Kx_diag", X=train_set, X2=None, diag=True)
            for it in timings_obj(Kx_diag):
                kern_save(it, kern, Kx_diag_out, True, None)

            Kt_diag, Kt_diag_out = kern_iterator(
                f, kern_name="Kt_diag", X=test_set, X2=None, diag=True)
            for it in timings_obj(Kt_diag):
                kern_save(it, kern, Kt_diag_out, True, None)

        Kxx, Kxx_out = kern_iterator(
            f, kern_name="Kxx", X=train_set, X2=None,     diag=False)
        Kxt, Kxt_out = kern_iterator(
            f, kern_name="Kxt", X=train_set, X2=test_set, diag=False)
        timings = timings_obj(itertools.count(), len(Kxx) + len(Kxt))

        assert f["Kxx"].dtype == onp.float64

        Kxx_ongoing = do_Kxx
        Kxt_ongoing = do_Kxt
        _t = 0
        while _t < skip_iterations:
            # Run iterations without doing any work
            try:
                next(Kxx); _t = next(timings)
            except StopIteration:
                Kxx_ongoing = False
            try:
                next(Kxt); _t = next(timings)
            except StopIteration:
                Kxt_ongoing = False

        if Kxx_ongoing:
            prev_Kxx = schedule_kernel(kern, next(Kxx), False, Kxx_mask); next(timings)
        if Kxt_ongoing:
            prev_Kxt = schedule_kernel(kern, next(Kxt), False, None); next(timings)
        while Kxx_ongoing or Kxt_ongoing:
            if Kxx_ongoing:
                try:
                    next_Kxx = schedule_kernel(kern, next(Kxx), False, Kxx_mask); next(timings)
                    kern_save(prev_Kxx[0], prev_Kxx[1], W_covs, Kxx_out, False, Kxx_mask)
                    prev_Kxx = next_Kxx
                except StopIteration:
                    Kxx_ongoing = False
                    kern_save(prev_Kxx[0], prev_Kxx[1], W_covs, Kxx_out, False, Kxx_mask)
                    del prev_Kxx
                    del next_Kxx
            if Kxt_ongoing:
                try:
                    next_Kxt = schedule_kernel(kern, next(Kxt), False, None); next(timings)
                    kern_save(prev_Kxt[0], prev_Kxt[1], W_covs, Kxt_out, False, None)
                    prev_Kxt = next_Kxt
                except StopIteration:
                    Kxt_ongoing = False
                    kern_save(prev_Kxt[0], prev_Kxt[1], W_covs, Kxt_out, False, None)
                    del prev_Kxt
                    del next_Kxt


if __name__ == '__main__':
    experiment.run_commandline()
