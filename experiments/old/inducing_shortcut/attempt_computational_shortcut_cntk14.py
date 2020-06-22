import jax.numpy as np
import jax
import numpy as onp
from neural_tangents.stax import (AvgPool, Conv, Dense, FanInSum, FanOut,
                                  Flatten, GlobalAvgPool, Identity)
from neural_tangents import stax
from cnn_limits.models import Relu
from cnn_limits.layers import covariance_tensor, CorrelatedConv
from cnn_limits.sparse import patch_kernel_fn, gen_slices
import tqdm
import faulthandler
faulthandler.enable()
import gpytorch
import math

conv = Conv(1, (3, 3), (1, 1), padding='SAME')
relu = Relu()

gpytorch_kern = gpytorch.kernels.MaternKernel(nu=3/2)
gpytorch_kern.lengthscale = math.exp(1)

# pool_W_cov = covariance_tensor(5, 5, gpytorch_kern)
# pool = CorrelatedConv(1, (5, 5), (1, 1), padding='VALID',
#                       W_cov_tensor=pool_W_cov)
pool_W_cov = None
pool = GlobalAvgPool()


cntk14_nopool = stax.serial(*([conv, relu]*2))
cntk14 = stax.serial(cntk14_nopool, pool, stax.Flatten())

cntk14_nopool_kfn = cntk14_nopool[2]
cntk14_kfn = jax.jit(cntk14[2], static_argnums=(2,))

stride = 1
size = 32

cntk14_patch = patch_kernel_fn(cntk14_nopool_kfn, (stride, stride), pool_W_cov)

def quick_cntk14(x1, x2, get):
    numel = size**4

    res = None
    shapes = set()

    for i in tqdm.trange(-size+1, size):
        for j in range(-size+1, size):
            i1, i2 = gen_slices(stride, i)
            j1, j2 = gen_slices(stride, j)
            _, a, b, _ = x1[:, i1, j1, :].shape
            _, c, d, _ = x2[:, j2, j2, :].shape

            shapes.add((
                tuple(sorted((a, b))),
                tuple(sorted((c, d)))))

            # print(f"i={i} , j={j}")
            # kernel = cntk14_patch(i, j, x1, x2, get=('var1', 'nngp'))
            # if res is None:
            #     res = np.sum(kernel.nngp, (-1, -2))
            # else:
            #     res = res + np.sum(kernel.nngp, (-1, -2))
    return shapes, len(shapes)
#quick_cntk14 = jax.jit(quick_cntk14, static_argnums=(2,))


key1, key2, key = jax.random.split(jax.random.PRNGKey(3243), 3)
W, H = size, size
x1 = jax.random.normal(key1, (2, W, H, 3), dtype=np.float32)
x2 = jax.random.normal(key2, (3, W, H, 3), dtype=np.float32)

print(quick_cntk14(onp.ones((2, W, H, 3)), onp.ones((3, W, H, 3)), 'nngp'))
# print(cntk14_kfn(x1, None, 'var1'))
