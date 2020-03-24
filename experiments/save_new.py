"""
Save a kernel matrix to disk
"""
import os
import sacred
import contextlib

import jax.numpy as np
import jax

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

    batch_size = 200
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
def save_K(kern, kern_name, X, X2, diag, batch_size, worker_rank, n_workers,
           print_interval):
    """
    Saves a kernel to the h5py file `f`. Creates its dataset with name `name`
    if necessary.
    """
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
    timings = PrintTimings(desc=f"{kern_name} (worker {worker_rank}/{n_workers})",
                           print_interval=print_interval)
    N = len(X)
    N2 = N if X2 is None else len(X2)
    zN = len(str(N))
    zN2 = len(str(N2))

    for same, (i, (x, _y)), (j, (x2, _y2)) in timings(it):
        k = kern(x, x2, same, diag)
        # if np.any(np.isinf(k)) or np.any(np.isnan(k)):
        #     print(f"About to write a nan or inf for {i},{j}")
        #     import ipdb; ipdb.set_trace()
        if diag:
            name = f"{kern_name}/{str(i).zfill(zN)}.npy"
        else:
            name = f"{kern_name}/{str(i).zfill(zN)}_{str(j).zfill(zN)}.npy"
        with new_file(name) as f:
            np.save(f, k)


load_dataset = experiment.capture(cnn_limits.load_dataset)


def model():
    return stax.serial(
        cnn_limits.models.PreResNetNoPooling(20),
        stax.Flatten(),
    )


@experiment.command
def mc_approx(max_n_functions):
    train_set, test_set = load_dataset()
    init_fn, _apply_fn, _ = model()
    apply_fn = jax.jit(lambda params, x: _apply_fn(params, np.moveaxis(x, 1, -1)))
    init_fn = jax.jit(init_fn)

    for _ in range(max_n_functions):
        loader = torch.utils.data.DataLoader()





@experiment.automain
def main(worker_rank):
    train_set, test_set = load_dataset()
    # train_set = Subset(train_set, range(40000))
    _, _, kernel_fn = model()
    # _, _, kernel_fn = cnn_limits.models.Myrtle5()
    # def kernel_fn(x1, x2, **kwargs):
    #     x1 = np.mean(x1, -1)
    #     x2 = (x1 if x2 is None else np.mean(x2, -1))
    #     x1 = np.reshape(x1, (-1, 1, 32, 32, 1, 1))
    #     x2 = np.reshape(x2, (1, -1, 1, 1, 32, 32))
    #     return np.mean(x1 * x2, (-4, -3, -2, -1))

    value_and_grad_kernel_fn = jax.value_and_grad(lambda x, y, get: np.sum(kernel_fn(x, y, get=get)), (0, 1))
    def kern_(x1, x2, same, diag):
        get = ("var1" if diag else "nngp")
        x1 = np.moveaxis(x1, 1, -1)
        x2 = (None if same else np.moveaxis(x2, 1, -1))
        #y = kernel_fn(x1, x2, get=get) #, marginalization={'marginal': M.NO, 'cross': M.NO, 'spec': 'NHWC'})
        y, grad = value_and_grad_kernel_fn(x1, x2, get)
        return grad
    kern_ = jax.jit(kern_, static_argnums=(2, 3))

    def kern(x1, x2, same, diag):
        x1 = np.asarray(x1.numpy())
        x2 = (None if same else np.asarray(x2.numpy()))
        return kern_(x1, x2, same, diag)

    save_K(kern,     kern_name="Kxx",     X=train_set, X2=None,      diag=False)
    save_K(kern,     kern_name="Kxtx",    X=test_set,  X2=train_set, diag=False)

    if worker_rank == 0:
        save_K(kern, kern_name="Kt_diag", X=test_set,  X2=None,      diag=True)
