import timeit

import jax.numpy as np
from jax.config import config
from jax import jit, grad, random
from jax.experimental import optimizers
from neural_tangents import stax
from neural_tangents.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                  FanOut, Flatten, GeneralConv, Identity, Relu, MaxPool)
import neural_tangents as nt
import jax
import functools

import tqdm
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch

import matplotlib.pyplot as plt
import seaborn as sns


def benchmark(function, number, *args, **kwargs):
    t = timeit.timeit("function(*args, **kwargs)", number=number, globals=locals())/number
    units = [(1e-9, "ns"), (1e-6, "μs"), (1e-3, "ms"), (1.0, "s")]
    for i in range(1, len(units)+1):
        scale, unit = units[i-1]
        if i==len(units) or t < units[i][0]:
            print(f"Function {function} takes {t/scale:.2f}{unit}")
            return



if __name__ == "__main__":
    # rng_key = random.PRNGKey(0)

    # batch_size = 8
    # num_classes = 1001
    # input_shape = (batch_size, 32, 32, 3)
    # step_size = 0.1
    # num_steps = 10000

    # init_fun, predict_fun = ResNet50(num_classes)
    # _, init_params = init_fun(rng_key, input_shape)

    # def loss(params, batch):
    #     inputs, targets = batch
    #     logits = predict_fun(params, inputs)
    #     return -np.sum(logits * targets)

    # def accuracy(params, batch):
    #     inputs, targets = batch
    #     target_class = np.argmax(targets, axis=-1)
    #     predicted_class = np.argmax(predict_fun(params, inputs), axis=-1)
    #     return np.mean(predicted_class == target_class)

    # def synth_batches():
    #     rng = npr.RandomState(0)
    #     while True:
    #         images = rng.rand(*input_shape).astype('float32')
    #         labels = rng.randint(num_classes, size=(batch_size, 1))
    #         onehot_labels = labels == np.arange(num_classes)
    #         yield images, onehot_labels

    # opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)
    # batches = synth_batches()

    # @jit
    # def update(i, opt_state, batch):
    #     params = get_params(opt_state)
    #     return opt_update(i, grad(loss)(params, batch), opt_state)

    # opt_state = opt_init(init_params)
    # for i in tbx.PrintTimings("Iter", 5.)(range(num_steps)):
    #     opt_state = update(i, opt_state, next(batches))
    # trained_params = get_params(opt_state)



    # _, _, kernel_fn = stax.serial(
    #     # stax.Conv(20, (3, 3), (2, 2), padding='SAME'),
    #     # stax.Conv(20, (3, 3), (2, 2), padding='SAME'),
    #     stax.Conv(20, (3, 3), (2, 2), padding='SAME'),
    #     # stax.Flatten(),
    #     # stax.Dense(1),
    # )
    # kernel_fn = jit(kernel_fn, static_argnums=(2,))


    key, key1, key2 = random.split(random.PRNGKey(1243), 3)
    W, H = 9, 9
    x1 = random.normal(key1, (1, W, H, 1))
    x2 = random.normal(key2, (1, W, H, 1))

    stride = 2
    size = 4
    init_fn, apply_fn, k = stax.serial(
        # stax.Conv(1, (size, size), (1, 1), padding='SAME'),
        # stax.Conv(1, (size, size), (1, 1), padding='SAME'),
        # stax.Relu(),
        # stax.Conv(1, (size, size), (1, 1), padding='SAME'),
        # stax.Conv(1, (size, size), (stride, stride), padding='VALID'),
        # stax.Relu(),
        # stax.Conv(1, (size, size), (stride, stride), padding='VALID'),
        stax.Conv(1, (size, size), (stride, stride), padding='VALID'),
        stax.Relu(),
        )
    _, _, kernel_fn2 = stax.serial(
        (init_fn, apply_fn, k),
        stax.AvgPool((3, 3)),
    )
    kernel_fn2 = jit(kernel_fn2, static_argnums=(2,))
    K_orig = kernel_fn2(x1, x2, 'nngp')

    def kernel_fn2_blocking(x1, x2):
        return kernel_fn2(x1, x2, "nngp").block_until_ready()
    _ = kernel_fn2_blocking(x1, x2)

    # benchmark(kernel_fn2_blocking, 1000, x1, x2)

    neutral_params = (
        (np.ones((size, size, 1, 1)) / size, np.zeros(())),
        # (np.ones((size, size, 1, 1)) / size, np.zeros(())),
        # (),
        # (np.ones((size, size, 1, 1)) / size, np.zeros(())),
        # (np.ones((size, size, 1, 1)) / size, np.zeros(())),
        # (),
        # (np.ones((size, size, 1, 1)) / size, np.zeros(())),
        # (np.ones((size, size, 1, 1)) / size, np.zeros(())),
        # (),
        )
    apply_fn = jit(apply_fn)
    k_ = jit(k, (2, 3))
    def k(x1, x2, *args):
        return k_(x1, x2, "nngp", "auto")

    K = np.zeros((1))
    numel = 0
    stride = 2 #*2*2
    for i in range(-3, 4):  # Depends on stride and size
        for j in range(-3, 4):
            if i >= 0:
                i1 = slice(stride*i, x1.shape[1])
                i2 = slice(0, x2.shape[1] - stride*i)
            else:
                i1 = slice(0, x1.shape[1] + stride*i)
                i2 = slice(-stride*i, x2.shape[1])
            if j >= 0:
                j1 = slice(stride*j, x1.shape[1])
                j2 = slice(0, x2.shape[2] - stride*j)
            else:
                j1 = slice(0, x1.shape[2] + stride*j)
                j2 = slice(-stride*j, x2.shape[1])
            print(i1, j1, i2, j2)

            # kernel = apply_fn(neutral_params, (x1[:, i1, j1, :] * x2[:, i2, j2, :]).sum(axis=-1, keepdims=True))
            kernel = k(x1[:, i1, j1, :], x2[:, i2, j2, :], 'nngp')
            K += np.sum(kernel)
            numel += np.prod(kernel.shape)
    print(K/numel, K_orig)


    K = 0
    numel = 0
    stride = 1
    for i in range(0, 9):
        for j in range(0, 9):
            x2_ = np.roll(x2, (stride*i, stride*j), axis=(1, 2))
            kernel = k(x1, x2_, 'nngp')
            K += np.sum(kernel)
            numel += np.prod(kernel.shape)
    print(K/numel, K_orig)

