import jax.numpy as np
from jax.config import config
from jax import jit, grad, random, vmap
import jax
from jax.experimental import optimizers
from neural_tangents import stax
from neural_tangents.stax import (AvgPool, Conv, Dense, FanInSum,
                                  FanOut, Flatten, GeneralConv, Identity, Relu, GlobalAvgPool)
import tqdm
import neural_tangents as nt
import jax
import functools

import timeit


key, key1, key2 = random.split(random.PRNGKey(3243), 3)
W, H = 5, 5
# x1 = np.arange(W*H).reshape((1, W, H, 1)).astype(np.float32)
# x2 = np.arange(W*H, 2*W*H).reshape((1, W, H, 1)).astype(np.float32)
x1 = jax.random.normal(key1, (1, W, H, 1), dtype=np.float32)
x2 = jax.random.normal(key2, (3, W, H, 1), dtype=np.float32)


_, _, kernel_fn = Conv(1, (3, 3), (1, 1), 'SAME')
_, _, gap_kernel_fn = stax.serial(
    Conv(1, (3, 3), (1, 1), 'SAME'),
    GlobalAvgPool(),
    Flatten())

K_orig = gap_kernel_fn(x1, x2, get='nngp')

K = None
numel = 0
stride = 1

all_x1 = np.zeros((15*15, *x1.shape), x1.dtype)
all_x2 = np.zeros((15*15, *x2.shape), x2.dtype)

# x1 = np.pad(x1, [(0, 0), (0, 1), (0, 1), (0, 0)])
# x2 = np.pad(x2, [(0, 0), (0, 1), (0, 1), (0, 0)])

_idx = 0
for i in tqdm.trange(-4, 5):  # Depends on stride and size
    for j in range(-4, 5):
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

        # all_x1  = jax.ops.index_update(all_x1, (_idx, slice(None, None, None), i1, j1, slice(None, None, None)), x1[:, i1, j1, :])
        # all_x2  = jax.ops.index_update(all_x2, (_idx, slice(None, None, None), i2, j2, slice(None, None, None)), x2[:, i2, j2, :])
        var1 = stax._get_covariance(x1[:, i1, j1,  :], x1[:, i2, j2,  :], stax.M.OVER_PIXELS, 0, -1)
        var1 = np.diagonal(var1, axis1=0, axis2=1)
        var2 = stax._get_covariance(x2[:, i1, j1,  :], x2[:, i2, j2,  :], stax.M.OVER_PIXELS, 0, -1)
        var2 = np.diagonal(var2, axis1=0, axis2=1)
        nngp = stax._get_covariance(x1[:, i1, j1,  :], x2[:, i2, j2,  :], stax.M.OVER_PIXELS, 0, -1)
        # var1 = np.sum(x1[:, i1, j1, :] * x1[:, i2, j2, :], -1)
        # nngp = np.sum(x1[:, None, i1, j1, :] * x2[:, i2, j2, :], -1)
        # var2 = np.sum(x2[:, i1, j1, :] * x2[:, i2, j2, :], -1)
        is_gaussian = False
        is_height_width = True
        is_input = True
        x1_is_x2 = False
        marginal = stax.M.OVER_PIXELS
        cross = stax.M.OVER_PIXELS
        ntk = None

        inputs = stax.Kernel(
            var1, nngp, var2, ntk, is_gaussian, is_height_width, marginal,
            cross, x1.shape, x2.shape, x1_is_x2, is_input)

        kernel = kernel_fn(inputs)

        if K is None:
            K = np.sum(kernel.nngp, (-1, -2))
        else:
            K = K + np.sum(kernel.nngp, (-1, -2))

        numel += np.prod(kernel.nngp.shape)
        _idx += 1
print(numel)

# assert _idx == 15*15
# kernel = block_position_kfn(x1[None, :, i1, j1, :], x2[None, :, i2, j2, :])

print(K/numel, K_orig)
