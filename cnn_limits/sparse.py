from neural_tangents.stax import _INPUT_REQ, M, _get_variance, _get_covariance, Kernel
from neural_tangents.gather_rolled_idx import gather_rolled_idx
import functools
import warnings
import collections
import jax.numpy as np
import numpy as onp
import torch
from jax import lax

def gen_slices(stride, size, i):
    if i == 0:
        return slice(0, size), slice(0, size)
    i1 = slice(0, size-stride*abs(i))
    i2 = slice(stride*abs(i), size)
    return ((i1, i2) if i<0 else (i2, i1))


InducingPatches = collections.namedtuple("InducingPatches", ("Z", "i", "start_idx", "mask"))
def mask_and_start_idx(stride, size, indices, out_start_idx, out_mask):
    if out_start_idx is None:
        out_start_idx = onp.zeros((len(indices), 2), dtype=int)
    if out_mask is None:
        out_mask = onp.zeros((len(indices), size), dtype=bool)

    for i, idx in enumerate(indices):
        i1, i2 = gen_slices(stride, size, idx)
        out_start_idx[i, 0] = i1.start
        out_start_idx[i, 1] = i2.start
        out_mask[i, :(i1.stop - i1.start)] = 1
    return InducingPatches(None, indices, out_start_idx, out_mask)


def patch_kernel_fn(kernel_fn, strides, W_cov):
    input_req = getattr(kernel_fn, _INPUT_REQ)
    if (int(input_req['marginal']) > int(M.OVER_PIXELS) or
        int(input_req['cross']) > int(M.OVER_PIXELS)):
        warnings.warn(f"Kernel function {kernel_fn} requires marginalisation "
                      f"{input_req}, which will be slow.")
    is_gaussian = False
    is_height_width = True
    is_input = True
    x1_is_x2 = False
    marginal = M.OVER_PIXELS
    cross = M.OVER_PIXELS


    if W_cov is None:
        W_cov_diagonals = None
    else:
        sz1, sz2, _, _ = W_cov.shape
        W_cov_diagonals = onp.zeros((sz1*2-1, sz2*2-1, *W_cov.shape[2:]),
                                    dtype=onp.float32)
        W_cov = onp.asarray(W_cov).copy()
        W_cov *= (1/W_cov.sum())
        for i in range(-sz1+1, sz1):
            for j in range(-sz2+1, sz2):
                d = onp.diagonal(
                    onp.diagonal(W_cov, offset=j, axis1=2, axis2=3),
                    offset=i, axis1=0, axis2=1)
                W_cov_diagonals[i+sz1-1, j+sz2-1, :d.shape[0], :d.shape[1]] = d
        W_cov_diagonals = np.asarray(W_cov_diagonals)


    @functools.wraps(kernel_fn)
    def _patch_kernel_fn(z1, start_idx1, mask1, z2, start_idx2, mask2):
        var1 = _get_variance(z1, marginal, 0, -1)
        var2 = _get_variance(z2, marginal, 0, -1)

        cross_mask = (mask1[:, None, :, None] *
                      mask2[None, :, None, :])

        # test = onp.zeros((32, 32), dtype=bool)
        # for i in range(63):
        #     for j in range(63):
        #         i1, i2 = gen_slices(1, 32, i-31)
        #         j1, j2 = gen_slices(1, 32, j-31)
        #         assert cross_mask[i, j].sum() == (i1.stop - i1.start)*(j1.stop - j1.start)
        if start_idx1.shape[0] == 1 and z1.shape[0] != 1:
            start_idx1 = np.tile(start_idx1, (z1.shape[0], 1))
        if start_idx2.shape[0] == 1 and z2.shape[0] != 1:
            start_idx2 = np.tile(start_idx2, (z2.shape[0], 1))

        ij_1 = np.stack([
            np.tile(np.arange(len(z1))[None, :], (len(z2), 1)),
            np.tile(start_idx1[None, :, 0], (len(z2), 1)),
            np.tile(start_idx2[:, 0, None], (1, len(z1))),
        ], axis=2)
        z1_sliced = gather_rolled_idx(z1, ij_1)

        ij_2 = np.stack([
            np.tile(np.arange(len(z2))[None, :], (len(z1), 1)),
            np.tile(start_idx1[:, 1, None], (1, len(z2))),
            np.tile(start_idx2[None, :, 1], (len(z1), 1)),
        ], axis=2)
        z2_sliced = gather_rolled_idx(z2, ij_2)

        nngp = np.einsum("bahwc,abhwc->abhw", z1_sliced, z2_sliced)
        nngp /= z1.shape[-1]

        ntk = None

        inputs = Kernel(
            var1, nngp, var2, ntk, is_gaussian, is_height_width, marginal,
            cross, z1.shape, z2.shape, x1_is_x2, is_input,
            (ij_1, ij_2), cross_mask)
        outputs = kernel_fn(inputs, get=('nngp', 'is_height_width'))
        nngp = outputs.nngp

        if W_cov_diagonals is None:
            nngp *= outputs.var_mask
            _, _, H, W = nngp.shape
            nngp = nngp.sum((2, 3)) * (1/(H*H*W*W))
        else:
            assert sz1 == sz2, "not implemented"
            _out_ij_1, _out_ij_2 = outputs.var_start_idx
            idx = np.transpose(_out_ij_1[:, :, 1:], (1, 0, 2)) - _out_ij_2[:, :, 1:] + (sz1-1)

            W = np.squeeze(lax.gather(
                W_cov_diagonals, idx, lax.GatherDimensionNumbers(
                    offset_dims=(2, 3, 4, 5),
                    collapsed_slice_dims=(),
                    start_index_map=(0, 1)),
                slice_sizes=(1, 1, *W_cov_diagonals.shape[2:])), axis=(2, 3))

            W_idx = ('ji' if outputs.is_height_width else 'ij')
            nngp = np.einsum(f"abij,ab{W_idx}->ab", nngp, W)
        return nngp
    return _patch_kernel_fn


def patch_kernel_fn_torch(kernel_fn, strides, W_cov):
    def _patch_kernel_fn(i, j, x1, x2):
        i1, i2 = gen_slices(strides[0], i)
        j1, j2 = gen_slices(strides[1], j)
        out = kernel_fn(x1, (x1 if x2 is None else x2), same=(x2 is None),
                        diag=False, var_slices=(i1, j1, i2, j2))
        if W_cov is None:
            _, _, H, W = x1.shape
            nngp = out.sum((-1, -2)) * (1/(H*H*W*W))
        else:
            matching_W_cov = torch.diagonal(
                torch.diagonal(W_cov, offset=j, dim1=2, dim2=3),
                offset=i, dim1=0, dim2=1)
            nngp = (out * matching_W_cov).sum((-1, -2))
        return nngp
    return _patch_kernel_fn
