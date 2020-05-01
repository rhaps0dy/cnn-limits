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
from torch.utils.data import DataLoader, Subset

import cnn_limits.models
import cnn_limits.sacred_utils as SU
from cnn_gp import DiagIterator, ProductIterator, create_h5py_dataset
from neural_tangents import stax
from neural_tangents.utils.kernel import Marginalisation as M
from nigp.tbx import PrintTimings

experiment = sacred.Experiment("save_new", [SU.ingredient])
if __name__ == '__main__':
    SU.add_file_observer(experiment)


@experiment.config
def config():
    batch_size = 10

    n_workers = 1
    worker_rank = 0
    print_interval = 2.
    model = "google_NNGP"

    use_ntk = False
    save_variance = False


@experiment.capture
def kern_iterator(f, kern_name, X, X2, diag, batch_size, worker_rank, n_workers):
    if kern_name in f.keys():
        print("Skipping {} (group exists)".format(kern_name))
        return
    N = len(X)
    N2 = N if X2 is None else len(X2)
    out = create_h5py_dataset(f, batch_size, kern_name, diag, N, N2)

    if diag:
        # Don't split the load for diagonals, they are cheap
        it = DiagIterator(batch_size, X, X2)
    else:
        it = ProductIterator(batch_size, X, X2, worker_rank=worker_rank,
                             n_workers=n_workers)
    return iter(it), out


def kern_save(iterate, kernel_fn, out, diag):
    same, (i, (x, _y)), (j, (x2, _y2)) = iterate
    k = kernel_fn(x, x2, same, diag)
    s = k.shape
    try:
        if len(s) == 2:
            out[:s[0], i:i+s[1]] = k
        else:
            out[:s[0], i:i+s[1], j:j+s[2]] = k
    except TypeError:
        out.resize(k.shape[0], axis=0)
        return kern_save(iterate, kernel_fn, out, diag)


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
def jax_model(model):
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
    def kern(x1, x2, same, diag):
        x1 = np.asarray(x1.numpy())
        x2 = (None if same else np.asarray(x2.numpy()))
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
def main(worker_rank, print_interval, n_workers, save_variance):
    train_set, test_set = SU.load_sorted_dataset()
    _, _, kernel_fn = jax_model()
    kern = jitted_kernel_fn(kernel_fn)

    with h5py.File(SU.base_dir()/"kernels.h5", "w") as f:
        timings_obj = PrintTimings(
            desc=f"Kxx$Kxt (worker {worker_rank}/{n_workers})",
            print_interval=print_interval)

        if save_variance and worker_rank == 0:
            Kx_diag, Kx_diag_out = kern_iterator(
                f, kern_name="Kx_diag", X=train_set, X2=None, diag=True)
            for it in timings_obj(Kx_diag):
                kern_save(it, kern, Kx_diag_out, True)

            Kt_diag, Kt_diag_out = kern_iterator(
                f, kern_name="Kt_diag", X=test_set, X2=None, diag=True)
            for it in timings_obj(Kt_diag):
                kern_save(it, kern, Kt_diag_out, True)

        Kxx, Kxx_out = kern_iterator(
            f, kern_name="Kxx", X=train_set, X2=None,     diag=False)
        Kxt, Kxt_out = kern_iterator(
            f, kern_name="Kxt", X=train_set, X2=test_set, diag=False)
        timings = timings_obj(itertools.count(), len(Kxx) + len(Kxt))

        Kxx_ongoing = Kxt_ongoing = True
        while Kxx_ongoing or Kxt_ongoing:
            if Kxx_ongoing:
                try:
                    kern_save(next(Kxx), kern, Kxx_out, False); next(timings)
                except StopIteration:
                    Kxx_ongoing = False
            if Kxt_ongoing:
                try:
                    kern_save(next(Kxt), kern, Kxt_out, False); next(timings)
                    kern_save(next(Kxt), kern, Kxt_out, False); next(timings)
                except StopIteration:
                    Kxt_ongoing = False


if __name__ == '__main__':
    experiment.run_commandline()
