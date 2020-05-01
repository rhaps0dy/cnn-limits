import jax.experimental.stax as ostax
import jax.numpy as np

import gpytorch
from neural_tangents import stax
from neural_tangents.stax import (AvgPool, Conv, Dense, FanInSum, FanOut,
                                  Flatten, GlobalAvgPool, Identity)

from .layers import (CorrelatedConv, CorrelationRelu, DenseSerialCheckpoint,
                     GaussianLayer, TickSerialCheckpoint, covariance_tensor)


def Relu():
    return stax.ABRelu(0, 2**.5)


def BasicBlock(out_chan, filter_shape=(3, 3), strides=(1, 1)):
    if strides == (1, 1):
        shortcut = Identity()
    else:
        shortcut = Conv(out_chan, (1, 1), strides, padding='SAME')
    main = stax.serial(
        # BatchNorm(),
        Relu(),
        Conv(out_chan, filter_shape, strides, padding='SAME'),
        # BatchNorm(),
        Relu(),
        Conv(out_chan, filter_shape, (1, 1), padding='SAME'),
    )
    return stax.serial(FanOut(2), stax.parallel(main, shortcut), FanInSum())


def BigBlock(BlockFn, channels, num_blocks, downsample=False):
    strides_first = (2, 2) if downsample else (1, 1)
    return stax.serial(
        BlockFn(channels, (3, 3), strides=strides_first),
        *[BlockFn(channels, (3, 3)) for _ in range(num_blocks-1)])


def PreResNetNoPooling(depth, channels=16):
    assert (depth - 2) % 6 == 0, "depth should be 6n+2"
    num_blocks_per_block = (depth - 2) // 6
    return stax.serial(
        Conv(channels, (3, 3), strides=(1, 1), padding='SAME'),
        BigBlock(BasicBlock,   channels, num_blocks_per_block, downsample=False),
        BigBlock(BasicBlock, 2*channels, num_blocks_per_block, downsample=True),
        BigBlock(BasicBlock, 4*channels, num_blocks_per_block, downsample=True),
        # BatchNorm(),
        )


def PreResNet(no_pooling_net, out_chan):
    _, _, f = no_pooling_net
    def kernel_fn(*args, **kwargs):
        kernel = f(*args, **kwargs)
        return np.mean(kernel, (-4, -3, -2, -1))
        # nngp = np.mean(kernel.nngp, (-4, -3, -2, -1))
        # var1 = np.mean(kernel.var1, (-4, -3, -2, -1))
        # var2 = np.mean(kernel.var2, (-4, -3, -2, -1))
        # return kernel._replace(var1=var1, var2=var2, nngp=nngp)
    #return None, None, kernel_fn

    return stax.serial(
        no_pooling_net,
        AvgPool((8, 8)),
        Flatten(),
        # Dense(1),
        # Relu(),
        # Dense(out_chan),
    )

def PreResNetEnd(no_pooling_net, out_chan):
    return stax.serial(
        no_pooling_net,
        Flatten(),
    )

def convrelu(channels, pool=None):
    if pool is None:
        avgpool = ()
    else:
        avgpool = (AvgPool(window_shape=pool, strides=pool),)
    strides = (1, 1)
    return stax.serial(
        *avgpool,
        Conv(channels, filter_shape=(3, 3), strides=strides, padding='SAME'),
        Relu(),
    )

def Myrtle1(channels=16):
    return stax.serial(
        Conv(channels, (3, 3), strides=(1, 1), padding='SAME'),
        Relu(),
        Conv(channels, (3, 3), strides=(2, 2), padding='SAME'),
        Relu(),
        Conv(channels, (3, 3), strides=(2, 2), padding='SAME'),
        Relu(),
        AvgPool((8, 8)),
    )

def Myrtle5(channels=16):
    conv = Conv(channels, (3, 3), strides=(1, 1), padding='SAME')
    relu = Relu()
    pool = AvgPool((2, 2), strides=(2, 2))

    return stax.serial(
        conv, relu,
        conv, relu, pool,
        conv, relu, pool,
        conv, relu, GlobalAvgPool(),
        CorrelationRelu())


def Myrtle10(channels=16):
    kern = gpytorch.kernels.MaternKernel(nu=3/2, lengthscale=2)
    Wcg = {}
    for sz in [32, 16, 8, 4, 2]:
        kern.lengthscale = sz/2
        Wcg[sz] = covariance_tensor(sz, sz, kern)

    return TickSerialCheckpoint(
        (*convrelu(channels), Wcg[32]),
        (*convrelu(channels), Wcg[32]),
        (*convrelu(channels), Wcg[32]),
        (*convrelu(channels, (2, 2)), Wcg[16]),
        (*convrelu(channels), Wcg[16]),
        (*convrelu(channels), Wcg[16]),
        (*convrelu(channels, (2, 2)), Wcg[8]),
        (*convrelu(channels), Wcg[8]),
        (*convrelu(channels), Wcg[8]),
        (*convrelu(channels, (2, 2)), Wcg[4]),
        (*convrelu(channels), Wcg[4]),
        (*convrelu(channels), Wcg[4]),
        (*convrelu(channels, (2, 2)), Wcg[2]),
        (*convrelu(channels), Wcg[2]),
        (*convrelu(channels), Wcg[2]),
    )

def convGaussian(channels, pool=None):
    if pool is None:
        avgpool = ()
    else:
        avgpool = (AvgPool(window_shape=pool, strides=pool),)
    strides = (1, 1)
    return stax.serial(
        *avgpool,
        Conv(channels, filter_shape=(3, 3), strides=strides, padding='SAME'),
        GaussianLayer(),
    )

def Myrtle10_Gaussian(channels=16):
    conv = Conv(channels, filter_shape=(3, 3), strides=(1, 1), padding='SAME')
    gaussian = GaussianLayer()
    avgpool2 = AvgPool(window_shape=(2, 2), strides=(2, 2))
    return stax.serial(
        conv, gaussian,
        conv, gaussian,
        conv, gaussian,
        avgpool2,
        conv, gaussian,
        conv, gaussian,
        conv, gaussian,
        avgpool2,
        conv, gaussian,
        conv, gaussian,
        conv, gaussian,
        GlobalAvgPool())

def NaiveConv(layers, channels=10):
    l = []
    for _ in range(layers):
        l.append(Conv(channels, (3, 3), strides=(1, 1), padding='SAME'))
        l.append(ABRelu(0, 2**.5))
    return stax.serial(
        *l,
        GlobalAvgPool(),
        Flatten(),
        Dense(1),
    )

def CorrelatedConvRelu(channels, W_cov, strides, W_cov_global):
    return (*stax.serial(
        CorrelatedConv(channels, (3, 3), strides, padding='SAME',
                       W_cov_tensor=W_cov),
        Relu(),
    ), W_cov_global)


def Myrtle5Correlated(channels=16):
    kern = gpytorch.kernels.MaternKernel(nu=3/2, lengthscale=2)
    W_cov = covariance_tensor(3, 3, kern)

    Wcg = {}
    for sz in [32, 16, 8, 4, 2]:
        kern.lengthscale = sz/2
        Wcg[sz] = covariance_tensor(sz, sz, kern)

    return TickSerialCheckpoint(
        CorrelatedConvRelu(channels, W_cov, (1, 1), Wcg[32]),
        CorrelatedConvRelu(channels, W_cov, (1, 1), Wcg[32]),
        CorrelatedConvRelu(channels, W_cov, (2, 2), Wcg[16]),
        CorrelatedConvRelu(channels, W_cov, (2, 2), Wcg[8]),
        CorrelatedConvRelu(channels, W_cov, (2, 2), Wcg[4]),
        CorrelatedConvRelu(channels, W_cov, (2, 2), Wcg[2]),
    )


def Myrtle5Uncorrelated(channels=16):
    return DenseSerialCheckpoint(
        stax.serial(Conv(channels, (3, 3), (1, 1), 'SAME'), Relu()),
        stax.serial(Conv(channels, (3, 3), (1, 1), 'SAME'), Relu()),
        stax.serial(Conv(channels, (3, 3), (2, 2), 'SAME'), Relu()),
        stax.serial(Conv(channels, (3, 3), (2, 2), 'SAME'), Relu()),
        stax.serial(Conv(channels, (3, 3), (2, 2), 'SAME'), Relu()),
        stax.serial(Conv(channels, (3, 3), (2, 2), 'SAME'), Relu()),
    )


def google_NNGP(channels, N_reps=1):
    block = stax.serial(Conv(channels, (3, 3), (1, 1), 'SAME'), Relu())
    return DenseSerialCheckpoint(*([block]*36))

def google_NNGP_sampling(channels, N_reps=1):
    kern = gpytorch.kernels.MaternKernel(nu=3/2)
    kern.lengthscale = 10
    W_cov = covariance_tensor(32, 32, kern)
    block = (*stax.serial(*([Conv(channels, (3, 3), (1, 1), 'SAME'), Relu()]*N_reps)), W_cov)
    return TickSerialCheckpoint(*([block]*(36//N_reps)))


def StraightConvNet(channels=16, N_reps=1):
    block = [Conv(channels, (3, 3), (1, 1), 'SAME'), Relu()]
    return stax.serial(
        *(block*36),
        Flatten(),
        Dense(1),
    )


def CNTK5(channels=16):
    conv = Conv(channels, filter_shape=(3, 3), strides=(1, 1), padding="SAME")
    relu = Relu()
    return stax.serial(
        *([conv, relu]*5),
        GlobalAvgPool(),
    )
