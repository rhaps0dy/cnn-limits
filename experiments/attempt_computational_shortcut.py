import jax.numpy as np
from jax.config import config
from jax import jit, grad, random, vmap
import jax
from jax.experimental import optimizers
from neural_tangents import stax
from neural_tangents.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                  FanOut, Flatten, GeneralConv, Identity, Relu, MaxPool)
import tqdm
import neural_tangents as nt
import jax
import functools

import timeit


def benchmark(function, number, *args, **kwargs):
    t = timeit.timeit("function(*args, **kwargs)", number=number, globals=locals())/number
    units = [(1e-9, "ns"), (1e-6, "μs"), (1e-3, "ms"), (1.0, "s")]
    for i in range(1, len(units)+1):
        scale, unit = units[i-1]
        if i==len(units) or t < units[i][0]:
            print(f"Function {function} takes {t/scale:.2f}{unit}")
            return


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
        Relu())


def PreResNet(no_pooling_net, out_chan):
    return stax.serial(
        no_pooling_net,
        AvgPool((4, 4)),
        Flatten(),
        Dense(out_chan))


key, key1, key2 = random.split(random.PRNGKey(3243), 3)
W, H = 16, 16
x1 = random.normal(key1, (1, W, H, 3))
x2 = random.normal(key2, (1, W, H, 3))

no_pooling_net = init_fn, apply_fn, kernel_fn = PreResNetNoPooling(20)
gap_init_fn, gap_apply_fn, gap_kernel_fn = PreResNet(no_pooling_net, 10)

kernel_fn = jit(kernel_fn, (2, 3))
gap_kernel_fn = jit(gap_kernel_fn, (2, 3))

K_orig = gap_kernel_fn(x1, x2, 'nngp', 'auto')

position_kfn = jit(vmap(lambda x, y: kernel_fn(x, y, 'nngp', 'auto'),
                        in_axes=(0, 0), out_axes=0))

if 'benchmarking':
    x1_ = random.normal(key1, (15*15, 40, W, H, 3))
    x2_ = random.normal(key2, (15*15, 40, W, H, 3))

    # ~1.5

    def block_gap_kernel_fn(x1, x2):
        return gap_kernel_fn(x1, x2, 'nngp', 'auto').block_until_ready()

    def block_position_kfn(*a, **k):
        return position_kfn(*a, **k).block_until_ready()

    block_gap_kernel_fn(x1, x2)
    block_position_kfn(x1_, x2_)

    benchmark(block_position_kfn, 100, x1_, x2_)
    benchmark(block_position_kfn, 100, x1_[:1], x2_[:1])
    benchmark(block_gap_kernel_fn, 100, x1_[0], x2_[0])


ffffs

K = 0.
numel = 0
stride = 2*2

all_x1 = np.zeros((15*15, *x1.shape), x1.dtype)
all_x2 = np.zeros((15*15, *x2.shape), x2.dtype)

# x1 = np.pad(x1, [(0, 0), (0, 1), (0, 1), (0, 0)])
# x2 = np.pad(x2, [(0, 0), (0, 1), (0, 1), (0, 0)])

_idx = 0
for i in tqdm.trange(-2, 3):  # Depends on stride and size
    for j in range(-2, 3):
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

        kernel = kernel_fn(x1[:, i1, j1, :], x2[:, i2, j2, :], 'nngp', 'auto')
        K += np.sum(kernel)
        numel += np.prod(kernel.shape)
        _idx += 1

# assert _idx == 15*15
# kernel = block_position_kfn(x1[None, :, i1, j1, :], x2[None, :, i2, j2, :])

print(K/numel, K_orig)
