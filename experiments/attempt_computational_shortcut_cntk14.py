import jax.numpy as np
import jax
from neural_tangents.stax import (AvgPool, Conv, Dense, FanInSum, FanOut,
                                  Flatten, GlobalAvgPool, Identity)
from neural_tangents import stax
from cnn_limits.models import Relu
from cnn_limits.layers import covariance_tensor, CorrelatedConv
from cnn_limits.sparse import patch_kernel_fn
import tqdm
import faulthandler
faulthandler.enable()
import gpytorch
import math

conv = Conv(1, (3, 3), (1, 1), padding='SAME')
relu = Relu()

gpytorch_kern = gpytorch.kernels.MaternKernel(nu=3/2)
gpytorch_kern.lengthscale = math.exp(1)

pool_W_cov = covariance_tensor(5, 5, gpytorch_kern)
pool = CorrelatedConv(1, (5, 5), (1, 1), padding='VALID',
                      W_cov_tensor=pool_W_cov)
# pool = GlobalAvgPool()

cntk14_nopool = stax.serial(*([conv, relu]*2))
cntk14 = stax.serial(cntk14_nopool, pool, stax.Flatten())

cntk14_nopool_kfn = cntk14_nopool[2]
cntk14_kfn = jax.jit(cntk14[2], static_argnums=(2,))

stride = 1
size = 5

cntk14_patch = patch_kernel_fn(cntk14_nopool_kfn, (stride, stride), pool_W_cov)

def quick_cntk14(x1, x2, get):
    numel = size**4

    res = None

    for i in tqdm.trange(-size+1, size):
        for j in range(-size+1, size):
            kernel = cntk14_patch(i, j, x1, x2, get=('var1', 'nngp'))
            if res is None:
                res = np.sum(kernel.nngp, (-1, -2))
            else:
                res = res + np.sum(kernel.nngp, (-1, -2))
    return res
#quick_cntk14 = jax.jit(quick_cntk14, static_argnums=(2,))


key1, key2, key = jax.random.split(jax.random.PRNGKey(3243), 3)
W, H = size, size
x1 = jax.random.normal(key1, (2, W, H, 3), dtype=np.float32)
x2 = jax.random.normal(key2, (3, W, H, 3), dtype=np.float32)

print(quick_cntk14(x1, x2, 'nngp'))
print(cntk14_kfn(x1, x2, 'nngp'))
