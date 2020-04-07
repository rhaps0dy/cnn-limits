import os
import sacred
import itertools

import jax.numpy as np
import numpy as onp
import jax
import pickle
import h5py
import tqdm

from cnn_gp import DiagIterator, ProductIterator
from nigp.tbx import PrintTimings
import cnn_limits
import cnn_limits.models
from torch.utils.data import Subset, DataLoader
from neural_tangents.utils.kernel import Marginalisation as M
from neural_tangents import stax

experiment = sacred.Experiment("mc_nn")
cnn_limits.sacred_utils.add_file_observer(experiment, __name__)
load_dataset = experiment.capture(cnn_limits.load_dataset)
base_dir = experiment.capture(cnn_limits.base_dir)
new_file = cnn_limits.def_new_file(base_dir)

@experiment.config
def config():
    dataset_base_path = "/scratch/ag919/datasets/"

    batch_size = 200**2
    N_train = None
    N_test = None
    sorted_dataset_path = os.path.join(dataset_base_path, "interlaced_argsort")
    dataset_name = "CIFAR10"
    max_n_samples = 80000

    print_interval = 2.
    model = "google_NNGP_sampling"
    n_channels=16


def create_dataset(f, batch_size, name, N):
    chunk_shape = (1, 1, min(batch_size, N))
    shape = (1, 1, N)
    maxshape = (None, None, N)
    return f.create_dataset(name, shape=shape, dtype=onp.float32,
                            fillvalue=onp.nan, chunks=chunk_shape, maxshape=maxshape)

@experiment.capture
def load_sorted_dataset(sorted_dataset_path, N_train, N_test):
    with experiment.open_resource(os.path.join(sorted_dataset_path, "train.pkl"), "rb") as f:
        train_idx = pickle.load(f)
    with experiment.open_resource(os.path.join(sorted_dataset_path, "test.pkl"), "rb") as f:
        test_idx = pickle.load(f)
    train_set, test_set = load_dataset()
    return (Subset(train_set, train_idx[:N_train]),
            Subset(test_set, test_idx[:N_test]))


@experiment.capture
def populate(F, sample_i, params, apply_fn, data, batch_size, _log):
    F.resize(max(sample_i+1, F.shape[1]), 1)
    for i, (x, _y) in enumerate(DataLoader(data, batch_size=batch_size)):
        data_i = slice(i*batch_size, (i+1)*batch_size)
        try:
            out = apply_fn(params, x)
            F[:, sample_i, data_i] = out
        except TypeError:  # should only run once
            _log.debug(f"Resizing data set {F}")
            F.resize(out.shape[0], axis=0)
            F[:, sample_i, data_i] = out


@experiment.automain
def main(batch_size, model, max_n_samples, n_channels, print_interval, _seed):
    train_set, test_set = load_sorted_dataset()
    input_shape = train_set.dataset.data.shape

    _orig_init_fn, _orig_apply_fn, _ = getattr(cnn_limits.models, model)(n_channels)
    init_fn = jax.jit(lambda rng: _orig_init_fn(rng, input_shape))
    @jax.jit
    def _apply_fn(params, x):
        x_nhwc = np.moveaxis(x, 1, -1)
        # dimensions: layer, readout, batch, 1
        out_l_b_r = _orig_apply_fn(params, x_nhwc)
        print([o.shape for o in out_l_b_r])
        out_lb_r = np.squeeze(np.concatenate(out_l_b_r, 0), -1)  # layer*readout, batch
        return out_lb_r

    def apply_fn(params, x):
        return _apply_fn(params, np.asarray(x.numpy()))

    key = jax.random.PRNGKey(_seed)

    with h5py.File(base_dir()/"mc.h5", "w") as f:
        F_train = create_dataset(f, batch_size, "F_train", len(train_set))
        F_test = create_dataset(f, batch_size, "F_test", len(test_set))

        print_timings = PrintTimings(print_interval=print_interval)
        for sample_i in print_timings(range(max_n_samples), total=max_n_samples):
            _, key = jax.random.split(key, 2)
            _, params = init_fn(key)
            populate(F_train, sample_i, params, apply_fn, train_set)
            populate(F_test, sample_i, params, apply_fn, test_set)
