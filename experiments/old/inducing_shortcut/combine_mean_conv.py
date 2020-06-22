import matplotlib.pyplot as plt
from neural_tangents import stax
import jax.numpy as np
import numpy as onp
import gpytorch
from cnn_limits.layers import CorrelatedConv, covariance_tensor


conv_then_avgpool=False

if conv_then_avgpool:
    # H, H, W, W
    out = onp.zeros((4, 4, 4, 4))

    conv_sz = 3

    conv_kernel = onp.eye(conv_sz**2) # H*W, H*W
    conv_kernel = conv_kernel.reshape((conv_sz, conv_sz, conv_sz, conv_sz)).transpose((0, 2, 1, 3)) / conv_sz**2

    for meanpool_h1 in range(2):
        for meanpool_h2 in range(2):
            for meanpool_w1 in range(2):
                for meanpool_w2 in range(2):
                    out[meanpool_h1:meanpool_h1+conv_sz,
                        meanpool_h2:meanpool_h2+conv_sz,
                        meanpool_w1:meanpool_w1+conv_sz,
                        meanpool_w2:meanpool_w2+conv_sz] += conv_kernel / (2**2)**2
else:
    # H, H, W, W
    out = onp.zeros((6, 6, 6, 6))

    avgpool_kernel = np.ones((2, 2, 2, 2)) / 2**2
    avg_stride = 2
    avg_sz = 2

    for meanpool_h1 in range(0, 6, avg_stride):
        for meanpool_w1 in range(0, 6, avg_stride):
            out[meanpool_h1:meanpool_h1+avg_sz,
                meanpool_h1:meanpool_h1+avg_sz,
                meanpool_w1:meanpool_w1+avg_sz,
                meanpool_w1:meanpool_w1+avg_sz] += avgpool_kernel / 36


img = out.transpose((0, 2, 1, 3))
img_sz = int(img.size**.5)
img = img.reshape((img_sz, img_sz))

# plt.imshow(img)
# plt.colorbar()
# plt.show()

x1 = np.arange(6*6).astype(np.float64).reshape((1, 6, 6, 1))
x2 = x1 + 6*6


_, _, kfn1 = CorrelatedConv(1, (out.shape[0], out.shape[2]), strides=(2, 2), padding='VALID', W_cov_tensor=out)
# _, _, kfn2 = stax.Conv(1, (3, 3), strides=(1, 1))
conv = stax.Conv(1, (3, 3), strides=(1, 1))
avgpool = stax.AvgPool((2, 2), strides=(2, 2))
if conv_then_avgpool:
    _, _, kfn2 = stax.serial(conv, avgpool)
else:
    _, _, kfn2 = stax.serial(avgpool, conv)

K1 = kfn1(x1, x2, get='nngp', marginalization={'marginal': stax.M.NO, 'cross': stax.M.NO, 'spec': 'NHWC'})
K2, is_hw_K2 = kfn2(x1, x2, get=('nngp', 'is_height_width'), marginalization={'marginal': stax.M.NO, 'cross': stax.M.NO, 'spec': 'NHWC'})

if not is_hw_K2:
    K2 = K2.transpose((0, 1, 4, 5, 2, 3))

print(K1/K2)
print("allclose?",  np.allclose(K1, K2))




kern = gpytorch.kernels.MaternKernel(nu=1/2, lengthscale=2)
#kern = gpytorch.kernels.RBFKernel()
kern.lengthscale = 3.5

matern_cov = covariance_tensor(6, 6, kern)

matern_img = matern_cov.transpose((0, 2, 1, 3)).reshape(
    (np.prod(matern_cov.shape[:2]), np.prod(matern_cov.shape[2:])))

plt.imshow(matern_img)
plt.colorbar()
plt.show()
