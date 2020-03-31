"""
Save a kernel matrix to disk
"""
import os
import sacred
import contextlib
import itertools

import jax.numpy as np
import jax
import pickle

from cnn_gp import DiagIterator, ProductIterator
from nigp.tbx import PrintTimings
import cnn_limits
import cnn_limits.models
from torch.utils.data import Subset, DataLoader
from neural_tangents.utils.kernel import Marginalisation as M
from neural_tangents import stax


experiment = sacred.Experiment("save_new")
if __name__ == '__main__':
    experiment.observers.append(
        sacred.observers.FileStorageObserver("/scratch/ag919/logs/save_new"))


@experiment.config
def config():
    dataset_base_path = "/scratch/ag919/datasets/"

    batch_size = 12
    N_train = 1200
    N_test = 2004
    sorted_dataset_path = os.path.join(dataset_base_path, "interlaced_argsort")
    dataset_name = "CIFAR10"
    max_n_functions = 50

    n_workers = 1
    worker_rank = 0
    print_interval = 2.


@experiment.capture
def base_dir(_run, _log):
    try:
        return _run.observers[0].dir
    except IndexError:
        _log.warning("This run has no associated directory, using `/tmp`")
        return "/tmp"


@contextlib.contextmanager
def new_file(relative_path):
    full_path = os.path.join(base_dir(), relative_path)
    with open(full_path, "wb") as f:
        yield f


@experiment.capture
def kern_iterator(kern_name, X, X2, diag, batch_size, worker_rank, n_workers):
    this_dir = os.path.join(base_dir(), kern_name)
    if os.path.exists(this_dir):
        print("Skipping {} (directory exists)".format(kern_name))
        return
    os.makedirs(this_dir)

    if diag:
        # Don't split the load for diagonals, they are cheap
        it = DiagIterator(batch_size, X, X2)
    else:
        it = ProductIterator(batch_size, X, X2, worker_rank=worker_rank,
                             n_workers=n_workers)
    N = len(X)
    N2 = N if X2 is None else len(X2)
    zN = len(str(N))
    zN2 = len(str(N2))

    if diag:
        def path_fn(i, j):
            return f"{kern_name}/{str(i).zfill(zN)}.npy"
    else:
        def path_fn(i, j):
            return f"{kern_name}/{str(i).zfill(zN)}_{str(j).zfill(zN)}.npy"
    return iter(it), path_fn


def kern_save(iterate, kernel_fn, path_fn):
    same, (i, (x, _y)), (j, (x2, _y2)) = iterate
    k = kernel_fn(x, x2, same, False)
    with new_file(path_fn(i, j)) as f:
        np.save(f, k)


load_dataset = experiment.capture(cnn_limits.load_dataset)

@experiment.command
def generate_sorted_dataset_idx(sorted_dataset_path):
    train_set, test_set = load_dataset()
    os.makedirs(sorted_dataset_path, exist_ok=True)
    train_idx = cnn_limits.interlaced_argsort(train_set)
    with new_file(os.path.join(sorted_dataset_path, "train.pkl")) as f:
        pickle.dump(train_idx, f)
    test_idx = cnn_limits.interlaced_argsort(test_set)
    with new_file(os.path.join(sorted_dataset_path, "test.pkl")) as f:
        pickle.dump(test_idx, f)


@experiment.capture
def load_sorted_dataset(sorted_dataset_path, N_train, N_test):
    with experiment.open_resource(os.path.join(sorted_dataset_path, "train.pkl"), "rb") as f:
        train_idx = pickle.load(f)
    with experiment.open_resource(os.path.join(sorted_dataset_path, "test.pkl"), "rb") as f:
        test_idx = pickle.load(f)
    train_set, test_set = load_dataset()
    return (Subset(train_set, train_idx[:N_train]),
            Subset(test_set, test_idx[:N_test]))


## JAX Model
def jax_model():
    return cnn_limits.models.Myrtle5Correlated()


def jitted_kernel_fn(kernel_fn):
    def kern_(x1, x2, same, diag):
        get = ("var1" if diag else "nngp")
        x1 = np.moveaxis(x1, 1, -1)
        x2 = (None if same else np.moveaxis(x2, 1, -1))
        y = kernel_fn(x1, x2, get=get)
        if isinstance(y, list):
            return np.stack(y)
        return y
    kern_ = jax.jit(kern_, static_argnums=(2, 3))
    def kern(x1, x2, same, diag):
        x1 = np.asarray(x1.numpy())
        x2 = (None if same else np.asarray(x2.numpy()))
        return kern_(x1, x2, same, diag)
    return kern


@experiment.main
def main(worker_rank, print_interval, n_workers):
    train_set, test_set = load_sorted_dataset()
    _, _, kernel_fn = jax_model()
    kern = jitted_kernel_fn(kernel_fn)

    Kxx, Kxx_path_fn = kern_iterator(kern_name="Kxx", X=train_set, X2=None,     diag=False)
    Kxt, Kxt_path_fn = kern_iterator(kern_name="Kxt", X=train_set, X2=test_set, diag=False)
    timings = PrintTimings(desc=f"Kxx+Kxt (worker {worker_rank}/{n_workers})",
                           print_interval=print_interval)(itertools.count(), len(Kxx) + len(Kxt))
    try:
        while True:
            kern_save(next(Kxx), kern, Kxx_path_fn)
            next(timings)
            kern_save(next(Kxt), kern, Kxt_path_fn)
            next(timings)
            kern_save(next(Kxt), kern, Kxt_path_fn)
            next(timings)
    except StopIteration:
        pass


if __name__ == '__main__':
    experiment.run_commandline()
