from neural_tangents.stax import _INPUT_REQ, M, _get_variance, _get_covariance, Kernel
import functools
import warnings
import jax.numpy as np
import torch

def gen_slices(stride, i):
    if i == 0:
        return slice(None), slice(None)
    i1 = slice(0, -stride*abs(i))
    i2 = slice(stride*abs(i), None)
    return ((i1, i2) if i<0 else (i2, i1))


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

    if W_cov is not None:
        W_cov = W_cov * (1/W_cov.sum())

    @functools.wraps(kernel_fn)
    def _patch_kernel_fn(i, j, x1, x2, get='nngp'):
        if 'var1' in get or 'var2' in get:
            warnings.warn("`var1` and `var2` won't have the correct value in the output")
        i1, i2 = gen_slices(strides[0], i)
        j1, j2 = gen_slices(strides[1], j)

        var1 = _get_variance(x1, marginal, 0, -1)
        var2 = _get_variance(x2, marginal, 0, -1)
        nngp = _get_covariance(x1[:, i1, j1, :], x2[:, i2, j2, :], cross, 0, -1)

        var_slices = (i1, j1, i2, j2)
        ntk = (0. if 'ntk' in get else None)

        inputs = Kernel(
            var1, nngp, var2, ntk, is_gaussian, is_height_width, marginal,
            cross, x1.shape, x2.shape, x1_is_x2, is_input, var_slices)
        if isinstance(get, tuple):
            get = (*get, 'is_height_width')
        else:
            get = (get, 'is_height_width')
        outputs = kernel_fn(inputs, get=get)

        if W_cov is None:
            _, H, W, _ = x1.shape
            nngp = outputs.nngp * (1/(H*H*W*W))
        else:
            matching_W_cov = np.diagonal(
                np.diagonal(W_cov, offset=j, axis1=2, axis2=3),
                offset=i, axis1=0, axis2=1)
            if outputs.is_height_width:
                nngp = outputs.nngp * matching_W_cov.T
            else:
                nngp = outputs.nngp * matching_W_cov
        return outputs._replace(nngp=nngp)

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
