import itertools
import os
import pickle

import h5py
import jax
import jax.numpy as np
import numpy as onp
import sacred
import tqdm
from torch.utils.data import DataLoader, Subset

import cnn_limits
import cnn_limits.models
from neural_tangents import stax
from neural_tangents.utils.kernel import Marginalisation as M
from nigp.tbx import PrintTimings

experiment = sacred.Experiment("mc_nn")
cnn_limits.sacred_utils.add_file_observer(experiment, __name__)
load_dataset = experiment.capture(cnn_limits.load_dataset)
base_dir = experiment.capture(cnn_limits.base_dir)
new_file = cnn_limits.def_new_file(base_dir)

@experiment.config
def config():
    dataset_base_path = "/scratch/ag919/datasets/"

    batch_size = 200
    N_train = None
    N_test = None
    sorted_dataset_path = os.path.join(dataset_base_path, "interlaced_argsort")
    dataset_name = "CIFAR10"
    max_n_samples = 3000

    print_interval = 2.
    model = "google_NNGP_sampling"
    n_channels=16
    N_reps=1
    kernel_matrix_path = "/scratch/ag919/logs/save_new/4/kernels.h5"


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
            F.resize(out.shape[0], axis=0)
            F[:, sample_i, data_i] = out

@experiment.command
def check_approx(batch_size, model, max_n_samples, n_channels, print_interval,
                 _seed, N_reps, kernel_matrix_path):
    print("Loading dataset")
    train_set, test_set = load_sorted_dataset()
    input_shape = train_set.dataset.data.shape

    print("Creating Jax network", model)
    _orig_init_fn, _orig_apply_fn, kernel_fn = getattr(cnn_limits.models, model)(n_channels, N_reps)
    print("compiliung jax init")
    init_fn = jax.jit(lambda rng: _orig_init_fn(rng, input_shape))
    print("compilngi jax apply")
    @jax.jit
    def _apply_fn(params, x):
        x_nhwc = np.moveaxis(x, 1, -1)
        # dimensions: layer, readout, batch, 1
        out_l_b_r = _orig_apply_fn(params, x_nhwc)
        print([o.shape for o in out_l_b_r])
        out_lb_r = np.squeeze(np.stack(out_l_b_r, 0), -1)  # layer*readout, batch
        return out_lb_r

    def apply_fn(params, x):
        return _apply_fn(params, np.asarray(x.numpy()))

    key= jax.random.PRNGKey(_seed)
    # all_keys = jax.random.split(, max_n_samples)

    print("creating batches")
    assert batch_size >= len(train_set) and batch_size >= len(test_set)
    train, _ = next(iter(DataLoader(train_set, batch_size=batch_size)))
    test, _ = next(iter(DataLoader(test_set, batch_size=batch_size)))
    train = np.moveaxis(np.asarray(train.numpy()), 1, -1)
    test = np.moveaxis(np.asarray(test.numpy()), 1, -1)

    _, _, kfn2 = cnn_limits.models.StraightConvNet()
    K2 = kfn2(train, train, get='nngp')
    K = kernel_fn(train, train, get='nngp')

    assert np.allclose(K[107], K2, rtol=1e-3)  # Ouch! Low precision.

    print("Calculating kernel")
    Kxx_ref = onp.asarray(np.stack(
        kernel_fn(np.moveaxis(np.asarray(train.numpy()), 1, -1),
                  np.moveaxis(np.asarray(train.numpy()), 1, -1), get='nngp')))
    Kxt_ref = onp.asarray(np.stack(
        kernel_fn(np.moveaxis(np.asarray(train.numpy()), 1, -1),
                  np.moveaxis(np.asarray(test.numpy()), 1, -1), get='nngp')))

    onp.save(base_dir()/"Kxx_ref.npy", Kxx_ref)
    onp.save(base_dir()/"Kxt_ref.npy", Kxt_ref)


    Kxx = onp.zeros(Kxx_ref.shape, dtype=onp.float64)
    Kxt = onp.zeros(Kxt_ref.shape, dtype=onp.float64)
    print_timings = PrintTimings(print_interval=print_interval)
    print_timings.data = [["x_atol", np.inf], ["t_atol", np.inf], ["x_meandev", np.inf], ["t_meandev", np.inf]]
    print("Entering l00p")
    tols = []
    for sample_i in print_timings(range(max_n_samples), total=max_n_samples):
        # _, params = init_fn(all_keys[sample_i])
        _, key = jax.random.split(key)
        _, params = init_fn(key)
        F_train = apply_fn(params, train)#[2::3, :]
        F_test = apply_fn(params, test)#[2::3, :]
        Kxx += np.expand_dims(F_train, -1) * np.expand_dims(F_train, -2) / max_n_samples
        Kxt += np.expand_dims(F_train, -1) * np.expand_dims(F_test, -2) / max_n_samples

        if sample_i % 100 == 0:
            Kxx_sub = Kxx*(max_n_samples/(sample_i+1)) - Kxx_ref
            print_timings.data[0][1] = np.abs(Kxx_sub).max()
            print_timings.data[2][1] = np.mean(Kxx_sub).max()
            Kxt_sub = Kxt*(max_n_samples/(sample_i+1)) - Kxt_ref
            print_timings.data[1][1] = np.abs(Kxt_sub).max()
            print_timings.data[3][1] = np.mean(Kxt_sub)
            tols.append([print_timings.data[0][1], print_timings.data[1][1],
                         print_timings.data[2][1], print_timings.data[3][1]])
    onp.save(base_dir()/"Kxx.npy", Kxx)
    onp.save(base_dir()/"Kxt.npy", Kxt)
    onp.save(base_dir()/"tols.npy", np.asarray(tols))


@experiment.automain
def main(batch_size, model, max_n_samples, n_channels, print_interval, _seed, N_reps):
    train_set, test_set = load_sorted_dataset()
    input_shape = train_set.dataset.data.shape

    _orig_init_fn, _orig_apply_fn, _ = getattr(cnn_limits.models, model)(n_channels, N_reps)
    init_fn = jax.jit(lambda rng: _orig_init_fn(rng, input_shape))
    @jax.jit
    def _apply_fn(params, x):
        x_nhwc = np.moveaxis(x, 1, -1)
        # dimensions: layer, readout, batch, 1
        out_l_b_r = _orig_apply_fn(params, x_nhwc)
        print([o.shape for o in out_l_b_r])
        out_lb_r = np.squeeze(np.stack(out_l_b_r, 0), -1)  # layer*readout, batch
        return out_lb_r

    def apply_fn(params, x):
        return _apply_fn(params, np.asarray(x.numpy()))

    all_keys = jax.random.split(jax.random.PRNGKey(_seed), max_n_samples)
    np.save(base_dir()/"random_keys.npy", all_keys)

    with h5py.File(base_dir()/"mc.h5", "w") as f:
        F_train = create_dataset(f, batch_size, "F_train", len(train_set))
        F_test = create_dataset(f, batch_size, "F_test", len(test_set))

        print_timings = PrintTimings(print_interval=print_interval)
        for sample_i in print_timings(range(max_n_samples), total=max_n_samples):
            _, params = init_fn(all_keys[sample_i])
            populate(F_train, sample_i, params, apply_fn, train_set)
            populate(F_test, sample_i, params, apply_fn, test_set)
