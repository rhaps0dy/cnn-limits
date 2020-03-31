from neural_tangents.stax import _layer, Padding, _INPUT_REQ, _set_input_req_attr
import neural_tangents.stax as stax
from neural_tangents.utils.kernel import Marginalisation as M
import jax.experimental.stax as ostax

from jax import random, lax, ops
import jax.numpy as np
import itertools
import torch


@stax._layer
def CorrelatedConv(out_chan,
                   filter_shape,
                   strides=None,
                   padding=Padding.VALID.name,
                   W_std=None,
                   W_cov_tensor=None,
                   parameterization='ntk'):
    parameterization = parameterization.lower()
    if parameterization not in ['ntk', 'standard']:
        raise ValueError(f"Parameterization not supported: {parameterization}")
    padding = Padding(padding)
    if padding == Padding.CIRCULAR:
        raise NotImplementedError("no circular padding")
    lhs_spec = out_spec = "NHWC"
    rhs_spec = "HWIO"
    dimension_numbers = lhs_spec, rhs_spec, out_spec
    C_index = lhs_spec.index('C')

    filter_numel = np.prod(filter_shape)
    height, width = filter_shape
    def input_total_dim(input_shape):
        return input_shape[C_index] * filter_numel

    def W_init_scalar(key, kernel_shape, std):
        return std * random.normal(key, kernel_shape)
    def W_init_tensor(key, kernel_shape, std):
        x = random.normal(key, kernel_shape)
        # Naive einsum is already optimal, and uses BLAS TDOT
        return np.einsum("ghqw,hwio->gqio", std, x)

    if W_cov_tensor is None:
        if W_std is None:
                W_std = 1.0
        W_std = np.asarray(W_std)
        if W_std.shape == ():
            W_init = W_init_scalar
            W_cov_tensor = np.eye(filter_numel).reshape(
                (filter_shape[0], filter_shape[1], filter_shape[0], filter_shape[1])
            ).transpose([0, 2, 1, 3]) * (W_std**2 / filter_numel)
        else:
            W_init = W_init_tensor
            assert W_std.shape == (height, height, width, width)
            W_cov_tensor = np.einsum("ahcw,bhdw->abcd", W_std, W_std) / filter_numel
    else:
        W_init = W_init_tensor
        W_std = np.linalg.cholesky(
            W_cov_tensor.transpose((0, 2, 1, 3)).reshape((filter_numel, filter_numel))
        ).reshape((height, width, height, width)).transpose((0, 2, 1, 3))

    def init_fn(rng, input_shape):
        kernel_shape = (*filter_shape, input_shape[C_index], out_chan)
        output_shape = lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding.name, dimension_numbers)
        if parameterization == 'standard':
            W_std_init = W_std / np.sqrt(input_total_dim(input_shape))
        else:
            W_std_init = W_std
        return output_shape, W_init(rng, kernel_shape, W_std_init)

    def apply_fn(params, inputs, **kwargs):
        W = params
        if parameterization == 'ntk':
            W = W / np.sqrt(input_total_dim(inputs.shape))
        return lax.conv_general_dilated(inputs, W, strides, padding.name,
                                        dimension_numbers=dimension_numbers)

    def kernel_fn(kernels):
        var1, nngp, var2, ntk, is_height_width, marginal, cross = (
            kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
            kernels.is_height_width, kernels.marginal, kernels.cross)
        if cross not in [M.OVER_POINTS, M.NO]:
            raise ValueError("Only possible for `M.OVER_POINTS` and `M.NO`. "
                             f"Supplied {cross}")
        if is_height_width:
            strides_ = strides
            W_cov_tensor_ = W_cov_tensor
        else:
            strides_ = strides[::-1]
            W_cov_tensor_ = np.transpose(W_cov_tensor, (2, 3, 0, 1))

        var1 = conv4d_for_5or6d(var1, W_cov_tensor_, strides_, padding)
        if var2 is not None:
            var2 = conv4d_for_5or6d(var2, W_cov_tensor_, strides_, padding)
        nngp = conv4d_for_5or6d(nngp, W_cov_tensor_, strides_, padding)
        if parameterization == 'ntk':
            if ntk is not None:
                ntk = conv4d_for_5or6d(ntk, W_cov_tensor_, strides_, padding) + nngp
        else:
            raise NotImplementedError("Don't know how to do NTK in standard")
        return kernels._replace(var1=var1, nngp=nngp, var2=var2, ntk=ntk,
                                is_height_width=is_height_width,
                                is_gaussian=True, is_input=False)

    setattr(kernel_fn, _INPUT_REQ, {'marginal': M.OVER_POINTS,
                                    'cross': M.NO,
                                    'spec': "NHWC"})
    return init_fn, apply_fn, kernel_fn


def naive_conv4d_for_5or6d(mat, W_cov_tensor, strides, padding):
    data_dim, X, Y = mat.shape[:-4], mat.shape[-3], mat.shape[-1]
    strides_4d = (strides[0], strides[0], strides[1], strides[1])

    kernel = np.reshape(W_cov_tensor, (*W_cov_tensor.shape, 1, 1))
    mat = mat.reshape((-1, X, X, Y, Y, 1))
    mat = lax.conv_general_dilated(
        mat, kernel, strides_4d, padding.name,
        dimension_numbers=("NHGWQC", "HGWQIO", "NHGWQC"))
    return mat.reshape((*data_dim, *mat.shape[1:5]))

def conv4d_for_5or6d(mat, W_cov_tensor, strides, padding):
    data_dim, X, Y = mat.shape[:-4], mat.shape[-3], mat.shape[-1]
    strides_3d = (strides[0], strides[1], strides[1])
    kernel = np.reshape(W_cov_tensor, (*W_cov_tensor.shape, 1, 1))

    if kernel.shape[0] != 3:
        return naive_conv4d_for_5or6d(mat, W_cov_tensor, strides, padding)
    kwargs = dict(
        window_strides=strides_3d,
        padding=padding.name,
        dimension_numbers=("NHWQC", "HWQIO", "NHWQC"))
    _, new_X, new_Y, _, _ = lax.conv_general_shape_tuple(
        (1, X, Y, Y, 1), kernel.shape, **kwargs)

    """
    Reasoning:
    X + pad == 3 + (new_X-1) * strides[0]
    """
    s0 = strides[0]
    pad = kernel.shape[0] + (new_X-1)*s0 - X
    if pad <= 0:
        idx_in = [slice(0, -2, s0),
                  slice(1, -1, s0),
                  slice(2, None, s0)]
        idx_out = [slice(None), RuntimeError, slice(None)]
    elif pad == 1:
        idx_in = [slice(0, -1, s0),
                  slice(1, None, s0),
                  slice(2, None, s0)]
        idx_out = [slice(None), RuntimeError, slice(None, -1)]
    elif pad == 2:
        idx_in = [slice(s0-1, -1, s0),
                  slice(None, None, s0),
                  slice(1, None, s0)]
        idx_out = [slice(1, None), RuntimeError, slice(None, -1)]
    else:
        raise ValueError(f"Padding was too much: pad={pad}")
    conv_shape = (-1, X, Y, Y, 1)
    new_shape = (*data_dim, -1, new_X, new_Y, new_Y)

    out = [
        lax.conv_general_dilated(
            mat[..., idx_in[i], :, :, :].reshape(conv_shape),
            kernel[i],
            **kwargs).reshape(new_shape)
           for i in range(len(idx_in))]

    def f(a, i):
        b, j = out[i], idx_out[i]
        if j == slice(None):
            return a+b
        return ops.index_add(a, ops.index[..., j, :, :, :], b)

    return f(f(out[1], 2), 0)


@_layer
def TickSerialCheckpoint(*layers):
    init_fns, apply_fns, kernel_fns, W_covs_list = zip(*layers)
    o_init_fn, o_apply_fn = ostax.serial(*zip(init_fns, apply_fns))

    n_outputs = 2*len(init_fns)

    def init_fn(rng, input_shape):
        output_shape, params = o_init_fn(rng, input_shape)
        return ([output_shape]*n_outputs,
                [params] + [()]*(n_outputs-1))

    def apply_fn(params, inputs, **kwargs):
        apply_out = o_apply_fn(params[0], inputs, **kwargs)
        return [apply_out] + [None]*(n_outputs-1)

    W_covs_T = [W_cov.transpose((2, 3, 0, 1)).ravel()
                for W_cov in W_covs_list]
    W_covs = [W_cov.ravel() for W_cov in W_covs_list]

    def kernel_fn(kernel):
        per_layer_kernels = []
        for f, W_cov, W_cov_T in zip(kernel_fns, W_covs, W_covs_T):
            kernel = f(kernel)
            d1, d2, *_ = kernel.nngp.shape
            nngp = kernel.nngp.reshape((d1, d2, -1))

            meanpool_nngp = nngp.mean(-1)
            if kernel.is_height_width:
                tick_nngp = nngp @ W_cov
            else:
                tick_nngp = nngp @ W_cov_T

            per_layer_kernels = per_layer_kernels + [
                kernel._replace(nngp=meanpool_nngp, var1=None, var2=None, ntk=None),
                kernel._replace(nngp=tick_nngp, var1=None, var2=None, ntk=None),
            ]
        return per_layer_kernels
    setattr(kernel_fn, _INPUT_REQ, {'marginal': M.OVER_POINTS,
                                    'cross': M.NO,
                                    'spec': "NHWC"})
    return init_fn, apply_fn, kernel_fn



def covariance_tensor(height, width, kern):
    """
    Returns HxHxWxW tensor, whose elements are the kernel between the positions
    of the elements of two HxW and HxW arrays. Useful for making TICK kernels.
    """
    H = torch.arange(height)
    W = torch.arange(width)
    HW = torch.stack(torch.meshgrid(H, W), 2)
    HW = HW.reshape((height*width, 2)).to(next(kern.parameters()))
    with torch.no_grad():
        mat = kern(HW, HW).evaluate().reshape((height, width, height, width))
    return np.asarray(mat.transpose(1, 2).cpu().numpy())
