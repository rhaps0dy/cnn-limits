"""
Save a kernel matrix to disk
"""
import torch
import importlib
import os
import sacred
import contextlib

import jax.numpy as np
import jax
from neural_tangents import stax
from neural_tangents.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                  FanOut, Flatten, GeneralConv, Identity, Relu, MaxPool)


#_, _, kernel_fn = stax.serial(Dense(1), FanOut(3))
#from jax import random
#key = random.PRNGKey(443)
#key, key1, key2 = random.split(key, 3)
#x1 = random.normal(key1, (2, 4))
#x2 = random.normal(key2, (3, 4))
#kernel_fn(x1, x2)

from cnn_limits.models import PreResNetNoPooling, PreResNet, Myrtle5




from cnn_gp import DatasetFromConfig, DiagIterator, ProductIterator
from nigp.tbx import PrintTimings
from gpytorch.kernels import JaxFnWrapper, jax2torch, torch2jax


experiment = sacred.Experiment("save_new")
if __name__ == '__main__':
    experiment.observers.append(
        sacred.observers.FileStorageObserver("/scratch/ag919/logs/save_new"))


@experiment.config
def config():
    dataset_base_path = "/scratch/ag919/datasets/"

    batch_size = 200
    # dataset_name = "MNIST"
    config_name = "cifar10"

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


@stax.layer
def explicit_serial(*layers):
    init_fns, apply_fns, kernel_fns = zip(*layers)
    init_fn, apply_fn = stax.ostax.serial(*zip(init_fns, apply_fns))

    def kernel_fn(kernel):
        kernels = [kernel]
        for f in kernel_fns:
            kernels.append(f(kernels[-1]))
        return kernels[-1]

    stax._set_input_req_attr(kernel_fn, kernel_fns)
    return init_fn, apply_fn, kernel_fn




@experiment.command
def test_kernels(config_name, dataset_base_path):
    config = importlib.import_module(f"configs.{config_name}")
    dataset = DatasetFromConfig(dataset_base_path, config)
    model = config.initial_model

    loader = iter(torch.utils.data.DataLoader(dataset.train, batch_size=4))
    x, _ = next(loader)
    x2, _ = next(loader)

    k_torch = model(x, x2, False, False)

    _, _, kernel_fn = PreResNet(PreResNetNoPooling(32), 1)
    kernel_fn = jax.jit(kernel_fn, (2, 3))

    x = np.moveaxis(np.asarray(x.numpy()), 1, -1)
    x2 = np.moveaxis(np.asarray(x2.numpy()), 1, -1)
    k_jax = kernel_fn(x, x2, "nngp", "auto")
    print(k_torch)
    print(k_jax)


@experiment.automain
def main(worker_rank, config_name, dataset_base_path):
    config = importlib.import_module(f"configs.{config_name}")
    dataset = DatasetFromConfig(dataset_base_path, config)

    # model = config.initial_model.cuda()
    # def kern(x, x2, same, diag):
    #     with torch.no_grad():
    #         return model(x.cuda(), x2.cuda(), same,
    #                      diag).detach().cpu().numpy()

    _, _, kernel_fn = PreResNet(PreResNetNoPooling(32), 1)
    # _, _, kernel_fn  = Myrtle5()
    kernel_fn = jax.jit(kernel_fn, (2, 3))
    def kernel_fn_(x, x2):
        return kernel_fn(x, x2, "nngp", "auto")

    # value_and_grad_kernel_fn = jax.jit(jax.value_and_grad(kernel_fn_, (0, 1)))
    @jax.curry
    @jax.jit
    def vjp_forward(x, x2, v):
        y, vjp = jax.vjp(kernel_fn_, x, x2)
        return vjp(v)

    def kern(x, x2, same, diag):
        x = np.moveaxis(np.asarray(x.numpy()), 1, -1)
        x2 = np.moveaxis(np.asarray(x2.numpy()), 1, -1)
        # y, vjp = jax.vjp(kernel_fn_, x, x2)
        # grad_x, grad_x2 = vjp(np.ones_like(y))
        y = kernel_fn_(x, x2)
        return y
        # grad_x, grad_x2 = vjp_forward(x, x2)(np.ones_like(y))
        # grad_x, grad_x2 = grad_x.block_until_ready(), grad_x2.block_until_ready()
        # return kernel_fn(x, x2, "nngp", "auto")
        # return value_and_grad_kernel_fn(x, x2)[0]
        return y

    def jax2torch_kern(x, x2, same, diag):
        x = x.cuda().transpose(1, -1)
        x2 = x2.cuda().transpose(1, -1)
        x, x2 = map(torch2jax, (x, x2))
        return jax2torch(kernel_fn(x, x2, "nngp", "auto")).cpu().numpy()

    def __kern(x, x2, same, diag):
        x = x.cuda().transpose(1, -1).requires_grad_(True)
        x2 = x2.cuda().transpose(1, -1).requires_grad_(True)
        # with torch.no_grad():
        loss = JaxFnWrapper.apply(kernel_fn_, vjp_forward, x, x2)
        # loss.backward()
        return loss.cpu().detach().numpy()

    save_K(kern, kern_name="Kxx",     X=dataset.train,      X2=None,          diag=False)
    save_K(kern, kern_name="Kxvx",    X=dataset.validation, X2=dataset.train, diag=False)
    save_K(kern, kern_name="Kxtx",    X=dataset.test,       X2=dataset.train, diag=False)

    if worker_rank == 0:
        save_K(kern, kern_name="Kv_diag", X=dataset.validation, X2=None, diag=True)
        save_K(kern, kern_name="Kt_diag", X=dataset.test,       X2=None, diag=True)
