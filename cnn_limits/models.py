from neural_tangents import stax
from neural_tangents.stax import (AvgPool, Dense, FanInSum, FanOut,
                                  Flatten, Identity, Relu, ABRelu, Conv, GlobalAvgPool)
import jax.experimental.stax as ostax
import jax.numpy as np


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
    if True or pool is None:
        avgpool = ()
    else:
        avgpool = (AvgPool(pool),)
    strides = ((1, 1) if pool is None else pool)
    return stax.serial(
        # *avgpool,
        Conv(channels, (1, 1), strides=strides, padding='SAME'),
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
    return stax.serial(
        convrelu(channels),
        convrelu(channels),
        convrelu(channels, (2, 2)),
        convrelu(channels, (2, 2)),
        AvgPool((8, 8)),
    )


def Myrtle10(channels=16):
    return myrtle_checkpoint_serial(
        convrelu(channels),
        convrelu(channels),
        convrelu(channels),
        convrelu(channels, (2, 2)),
        convrelu(channels),
        convrelu(channels),
        convrelu(channels, (2, 2)),
        convrelu(channels),
        convrelu(channels),
    )

# @stax.layer
def myrtle_checkpoint_serial(*layers):
    init_fns, apply_fns, kernel_fns = zip(*layers)
    o_init_fn, o_apply_fn = ostax.serial(*zip(init_fns, apply_fns))

    n_outputs = 2*len(init_fns)

    def init_fn(rng, input_shape):
        output_shape, params = o_init_fn(rng, input_shape)
        return ([output_shape]*n_outputs,
                [params] + [()]*(n_outputs-1))

    def apply_fn(params, inputs, **kwargs):
        apply_out = o_apply_fn(params[0], inputs, **kwargs)
        return [apply_out] + [None]*(n_outputs-1)

    def kernel_fn(kernel):
        per_layer_kernels = []
        for f in kernel_fns:
            kernel = f(kernel)

            meanpool_cov1 = np.mean(kernel.cov1, (-4, -3, -2, -1))
            meanpool_cov2 = (None if kernel.cov2 is None else
                             np.mean(kernel.cov2, (-4, -3, -2, -1)))
            meanpool_nngp = np.mean(kernel.nngp, (-4, -3, -2, -1))

            tick_cov1 = np.mean(kernel.cov1, (-4, -3, -2, -1)) + 1
            tick_cov2 = (None if kernel.cov2 is None else
                         np.mean(kernel.cov2, (-4, -3, -2, -1)) + 1)
            tick_nngp = np.mean(kernel.nngp, (-4, -3, -2, -1)) + 1

            per_layer_kernels = per_layer_kernels + [
                kernel._replace(cov1=meanpool_cov1, cov2=meanpool_cov2, nngp=meanpool_nngp),
                kernel._replace(cov1=tick_cov1, cov2=tick_cov2, nngp=tick_nngp),
            ]
        return per_layer_kernels
    return init_fn, apply_fn, kernel_fn


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
