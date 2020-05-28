import jax.numpy as np
import jax
from neural_tangents.stax import (AvgPool, Conv, Dense, FanInSum, FanOut,
                                  Flatten, GlobalAvgPool, Identity)
from neural_tangents import stax
from cnn_limits.models import Relu
from cnn_limits.layers import covariance_tensor, CorrelatedConv
import tqdm
import gpytorch
import math

conv = Conv(1, (3, 3), (1, 1), padding='SAME')
relu = Relu()

gpytorch_kern = gpytorch.kernels.MaternKernel(nu=3/2)
gpytorch_kern.lengthscale = math.exp(1)

# pool_W_cov = covariance_tensor(32, 32, gpytorch_kern)
# pool = CorrelatedConv(1, (3, 3), (1, 1), padding='VALID',
#                       W_cov_tensor=pool_W_cov)
pool = GlobalAvgPool()

cntk14_nopool = stax.serial(*([conv, relu]*3))
cntk14 = stax.serial(cntk14_nopool, pool, stax.Flatten())

cntk14_nopool_kfn = cntk14_nopool[2]
cntk14_kfn = jax.jit(cntk14[2], static_argnums=(2,))

stride = 1
size = 5
def quick_cntk14(x1, x2, get):
    numel = size**4
    is_gaussian = False
    is_height_width = True
    is_input = True
    x1_is_x2 = False
    marginal = stax.M.OVER_PIXELS
    cross = stax.M.OVER_PIXELS
    ntk = (0. if 'ntk' in get else None)

    res = None

    for i in tqdm.trange(-size+1, size):
        if i >= 0:
            i1 = slice(stride*i, x1.shape[1])
            i2 = slice(0, x2.shape[1] - stride*i)
        else:
            i1 = slice(0, x1.shape[1] + stride*i)
            i2 = slice(-stride*i, x2.shape[1])

        for j in range(-size+1, size):
            if j >= 0:
                j1 = slice(stride*j, x1.shape[1])
                j2 = slice(0, x2.shape[2] - stride*j)
            else:
                j1 = slice(0, x1.shape[2] + stride*j)
                j2 = slice(-stride*j, x2.shape[1])

            # var1_cross = stax._get_covariance(x1[:, i1, j1,  :], x1[:, i2, j2,  :], marginal, 0, -1)
            # var1_cross = np.moveaxis(np.diagonal(var1_cross, axis1=0, axis2=1), -1, 0)
            var1 = stax._get_variance(x1, marginal, 0, -1)
            # var1 = np.concatenate([var1, var1_cross], axis=0)

            # var2_cross = stax._get_covariance(x2[:, i1, j1,  :], x2[:, i2, j2,  :], marginal, 0, -1)
            # var2_cross = np.moveaxis(np.diagonal(var2_cross, axis1=0, axis2=1), -1, 0)
            var2 = stax._get_variance(x2, marginal, 0, -1)
            # var2 = np.concatenate([var2, var2_cross], axis=0)

            nngp = stax._get_covariance(x1[:, i1, j1,  :], x2[:, i2, j2,  :], cross, 0, -1)

            var_slices = (i1, j1, i2, j2)
            inputs = stax.Kernel(
                var1, nngp, var2, ntk, is_gaussian, is_height_width, marginal,
                cross, x1.shape, x2.shape, x1_is_x2, is_input, var_slices)

            kernel = cntk14_nopool_kfn(inputs, None, None)
            # kernel = cntk14_nopool_kfn(x1[:, i1, j1, :], x2[:, i2, j2, :])

            if res is None:
                res = np.sum(kernel.nngp, (-1, -2))
                _var1 = np.sum(kernel.var1, (-1, -2))
                _var2 = np.sum(kernel.var2, (-1, -2))
            else:
                res = res + np.sum(kernel.nngp, (-1, -2))
                _var1 = _var1 + np.sum(kernel.var1, (-1, -2))
                _var2 = _var2 + np.sum(kernel.var2, (-1, -2))
    return res/numel, _var1/numel, _var2/numel
#quick_cntk14 = jax.jit(quick_cntk14, static_argnums=(2,))


key1, key2, key = jax.random.split(jax.random.PRNGKey(3243), 3)
W, H = size, size
x1 = jax.random.normal(key1, (2, W, H, 3), dtype=np.float32)
x2 = jax.random.normal(key2, (3, W, H, 3), dtype=np.float32)

print(quick_cntk14(x1, x2, 'nngp'))
print(cntk14_kfn(x1, x2, ('nngp', 'var1', 'var2')))

a = np.array([[10.042922, 10.258308 , 9.487773],
 [ 9.362908 ,10.365824  ,8.488091]])

b = np.array([[0.35157394 ,0.35285723 ,0.32514787],
 [0.3370959 , 0.34061882 ,0.31103492]])
