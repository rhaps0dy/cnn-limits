import jax
import jax.numpy as np
import torch
from jax import lax, ops, random
import warnings

import neural_tangents.stax as stax
from neural_tangents.stax import (_INPUT_REQ, Padding, _layer,
                                  _set_input_req_attr)
from neural_tangents.utils.kernel import Marginalisation as M


def W_init_scalar(std, key, kernel_shape):
    return std * random.normal(key, kernel_shape)
def W_init_tensor(std, key, kernel_shape):
    x = random.normal(key, kernel_shape)
    # Naive einsum is already optimal, and uses BLAS TDOT
    return np.einsum("ghqw,hwio->gqio", std, x)
def tensor_cholesky(W_cov_tensor):
    height, width = W_cov_tensor.shape[-3], W_cov_tensor.shape[-1]
    filter_numel = height*width
    a = W_cov_tensor.transpose((0, 2, 1, 3)).reshape((filter_numel, filter_numel))
    a = np.linalg.cholesky(a)
    return a.reshape((height, width, height, width)).transpose((0, 2, 1, 3))

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

    if W_cov_tensor is None:
        if W_std is None:
                W_std = 1.0
        W_std = np.asarray(W_std)
        if W_std.shape == ():
            W_init = W_init_scalar
            W_cov_tensor = np.eye(filter_numel).reshape(
                (filter_shape[0], filter_shape[1], filter_shape[0], filter_shape[1])
                # Has only the diagonal so we divide by filter_numel
            ).transpose([0, 2, 1, 3]) * (W_std**2 / filter_numel)
        else:
            W_init = W_init_tensor
            assert W_std.shape == (height, height, width, width)
            W_cov_tensor = np.einsum("ahcw,bhdw->abcd", W_std, W_std)
            # Has the whole tensor so we divide by filter_numel**2
            W_cov_tensor = W_cov_tensor * (1/W_cov_tensor.sum())
    else:
        W_cov_tensor = W_cov_tensor * (1/W_cov_tensor.sum())
        W_init = W_init_tensor
        W_std = tensor_cholesky(W_cov_tensor)

    def init_fn(rng, input_shape):
        kernel_shape = (*filter_shape, input_shape[C_index], out_chan)
        output_shape = lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding.name, dimension_numbers)
        if parameterization == 'standard':
            W_std_init = W_std / np.sqrt(input_total_dim(input_shape))
        else:
            W_std_init = W_std
        return output_shape, W_init(W_std_init, rng, kernel_shape)

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

    kwargs = dict(
        window_strides=strides_3d,
        padding=padding.name,
        dimension_numbers=("NHWQC", "HWQIO", "NHWQC"))
    _, new_X, new_Y, _, _ = lax.conv_general_shape_tuple(
        (1, X, Y, Y, 1), kernel.shape, **kwargs)
    if new_X <= 0 or new_Y <= 0:
        raise ValueError(
            f"Filter shape {kernel.shape} does not fit in image shape {X}, {Y}")
    if any(s > 5 for s in kernel.shape):
        warnings.warn(f"Kernel shape {kernel.shape} larger than tested (up to 5)")

    """
    Reasoning:
    X + pad == 3 + (new_X-1) * strides[0]
    """
    s0 = strides[0]
    pad = max(0, kernel.shape[0] + (new_X-1)*s0 - X)
    assert pad < kernel.shape[0]
    ax_sz = mat.shape[-4]
    # We have to add in total "pad" padding. Padding first gets added in the
    # end, then in the beginning.
    pad_begin = pad//2
    pad_end = pad-pad_begin

    # This means that the floor-rounded half point of the filter always has no
    # padding.
    no_padding_point = (kernel.shape[0]-1)//2

    # Calculate: where does the `no_padding_point` fall, at the beginning and at the end of the convolution?
    base_idx_in = slice(no_padding_point-pad_begin,
                        (ax_sz+pad_end) - kernel.shape[0] + no_padding_point + 1,
                        s0)

    # idx_out is not used for the base index
    idxes = [(no_padding_point, base_idx_in, RuntimeError)]
    # Now move left:
    for i in range(1, no_padding_point+1):
        kernel_i = no_padding_point-i
        start = base_idx_in.start - i
        out_start = 0
        while start < 0:
            start += s0  # Take start modulo stride, plus as many strides needed to make it positive
            out_start += 1
        idx_in = slice(start, base_idx_in.stop-i, s0)
        if out_start == 0:
            idx_out = slice(None)
        else:
            idx_out = slice(out_start, new_X)
        idxes.append((kernel_i, idx_in, idx_out))

    # Now move right:
    for i in range(1, kernel.shape[0]-no_padding_point):
        kernel_i = no_padding_point+i
        idx_in = slice(base_idx_in.start+i, min(base_idx_in.stop+i, ax_sz), s0)
        n_elem_out = (idx_in.stop - idx_in.start -1)//s0 + 1
        if n_elem_out == new_X:
            idx_out = slice(None)
        else:
            idx_out = slice(0, n_elem_out)
        idxes.append((kernel_i, idx_in, idx_out))

    conv_shape = (-1, X, Y, Y, 1)
    new_shape = (*data_dim, -1, new_X, new_Y, new_Y)

    out = [
        lax.conv_general_dilated(
            mat[..., idx_in, :, :, :].reshape(conv_shape),
            kernel[kernel_i],
            **kwargs).reshape(new_shape)
        for (kernel_i, idx_in, _) in idxes]

    res = out[0]
    for (_, _, j), b in zip(idxes[1:], out[1:]):
        if j == slice(None):
            res = res+b
        else:
            res = ops.index_add(res, ops.index[..., j, :, :, :], b)
    return res


def _serial_init_fn(init_fns, readout_init_fns, init_intermediate, rng, input_shape):
    output_shape = []
    all_params = []
    rngs = jax.random.split(rng, 2*len(init_fns)).reshape((len(init_fns), 2, -1))
    for i, (fn, readout_fn) in enumerate(zip(init_fns, readout_init_fns)):
        input_shape, params = fn(rngs[i, 0], input_shape)
        if init_intermediate:
            shapes, params_readout = readout_fn(rngs[i, 1], input_shape)
        else:
            raise NotImplementedError
        output_shape = output_shape + shapes
        all_params.append((params, params_readout))
    return output_shape, all_params

def _serial_apply_fn(apply_fns, readout_apply_fns, params, inputs, **kwargs):
    outputs = []
    for fn, fn_readout, (p, p_readout) in zip(apply_fns, readout_apply_fns, params):
        inputs = fn(p, inputs)
        if p_readout is None:
            outputs.append(None)
        else:
            out = fn_readout(p_readout, inputs)
            if isinstance(out, list):
                outputs = outputs + out
            else:
                outputs.append(out)
    return outputs


@_layer
def TickSerialCheckpoint(*layers, intermediate_outputs=True):
    init_fns, apply_fns, kernel_fns, W_covs_list = zip(*layers)
    dense_init, dense_apply, dense_kernel = stax.serial(stax.Flatten(), stax.Dense(1))

    # Assume input_shape[-3:-1] are spatial dimensions
    def readout_init(W_std, rng, input_shape):
        rng = jax.random.split(rng, 3)
        out_dense, params_meanpool_dense = dense_init(rng[0], (*input_shape[:-3], input_shape[-1]))
        params_tick = W_init_tensor(W_std, rng[1], (*input_shape[-3:], 1))
        _, params_dense = dense_init(rng[2], input_shape)
        return [out_dense]*3, (params_meanpool_dense, params_tick, params_dense)
    init_fn = jax.partial(
        _serial_init_fn, init_fns,
        [jax.partial(readout_init, tensor_cholesky(W_cov))
         for W_cov in W_covs_list],
        intermediate_outputs)

    def readout_apply(params, inputs):
        # Assume that inputs[-1] are the channels, and inputs[-3:-1] the image
        # spatial dimensions.
        params_meanpool_dense, params_tick, params_dense = params
        mean_pool = dense_apply(params_meanpool_dense, np.mean(inputs, axis=(-3, -2)))
        tick = np.einsum("...hwi,hwio->...o", inputs, params_tick) / (
            np.sqrt(np.prod(inputs.shape[-3:])))
        dense = dense_apply(params_dense, inputs)
        return [mean_pool, tick, dense]

    apply_fn = jax.partial(
        _serial_apply_fn, apply_fns, [readout_apply]*len(apply_fns))

    def f(W_cov):
        a = W_cov.ravel()
        return a * (1/a.sum())
    W_covs_T = [f(W_cov.transpose((2, 3, 0, 1)))
                for W_cov in W_covs_list]
    W_covs = [f(W_cov) for W_cov in W_covs_list]

    def kernel_fn(kernel):
        per_layer_kernels = []
        for f, W_cov, W_cov_T in zip(kernel_fns, W_covs, W_covs_T):
            kernel = f(kernel)
            d1, d2, h, _, w, _ = kernel.nngp.shape
            nngp = kernel.nngp.reshape((d1, d2, -1))
            var1 = kernel.var1.reshape((d1, -1))
            var2 = (None if kernel.var2 is None else kernel.var2.reshape((d2, -1)))

            meanpool_nngp = nngp.mean(-1)
            meanpool_var1 = var1.mean(-1)
            meanpool_var2 = (None if var2 is None else var2.mean(-1))
            if kernel.is_height_width:
                tick_nngp = nngp @ W_cov
                tick_var1 = var1 @ W_cov
                tick_var2 = (None if var2 is None else var2@W_cov)
            else:
                tick_nngp = nngp @ W_cov_T
                tick_var1 = var1 @ W_cov_T
                tick_var2 = (None if var2 is None else var2@W_cov_T)
            this_layer_kernels = [
                kernel._replace(nngp=meanpool_nngp, var1=meanpool_var1, var2=meanpool_var2, ntk=None),
                kernel._replace(nngp=tick_nngp, var1=tick_var1, var2=tick_var2, ntk=None),
                dense_kernel(kernel),
            ]
            per_layer_kernels = per_layer_kernels + this_layer_kernels
        return per_layer_kernels
    setattr(kernel_fn, _INPUT_REQ, {'marginal': M.OVER_POINTS,
                                    'cross': M.NO,
                                    'spec': "NHWC"})
    return init_fn, apply_fn, kernel_fn

@_layer
def DenseSerialCheckpoint(*layers, intermediate_outputs=True):
    init_fns, apply_fns, kernel_fns = zip(*layers)

    _readout_init_fn, readout_apply_fn, readout_kernel_fn = stax.serial(
        stax.Flatten(), stax.Dense(1))
    def readout_init_fn(rng, input_shape):
        s, params = _readout_init_fn(rng, input_shape)
        return [s], params
    init_fn = jax.partial(_serial_init_fn,
                           init_fns,  [readout_init_fn ]*len(init_fns),
                           intermediate_outputs)
    apply_fn = jax.partial(_serial_apply_fn,
                           apply_fns, [readout_apply_fn]*len(init_fns))

    def kernel_fn(kernel):
        per_layer_kernels = []
        for f in kernel_fns:
            kernel = f(kernel)
            per_layer_kernels.append(readout_kernel_fn(kernel))
        return per_layer_kernels
    _set_input_req_attr(kernel_fn, kernel_fns + (readout_kernel_fn,))
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


def elementwise_init_fn(rng, input_shape):
    return input_shape, ()


def elementwise_apply_fn(params, inputs):
    raise NotImplementedError


def _gaussian_kernel(ker_mat, prod, do_backprop):
    sqrt_prod = stax._safe_sqrt(prod)
    cosines = ker_mat / sqrt_prod
    return sqrt_prod * np.exp(cosines - 1)


@_layer
def GaussianLayer(do_backprop=False):
    "similar to relu but less acute"
    def kernel_fn(kernels):
        var1, nngp, var2, ntk, marginal = \
            kernels.var1, kernels.nngp, kernels.var2, kernels.ntk, kernels.marginal
        if ntk is not None:
            raise NotImplementedError
        prod11, prod12, prod22 = stax._get_normalising_prod(var1, var2, marginal, kernels.var_slices)
        nngp = _gaussian_kernel(nngp, prod12, do_backprop)
        var1 = _gaussian_kernel(var1, prod11, do_backprop)
        if var2 is not None:
            var2 = _gaussian_kernel(var2, prod22, do_backprop)

        return kernels._replace(
            var1=var1, nngp=nngp, var2=var2, ntk=ntk, is_gaussian=False,
            marginal=marginal)

    return elementwise_init_fn, elementwise_apply_fn, kernel_fn


def proj_relu_kernel(ker_mat, prod, do_backprop):
    cosines = ker_mat / stax._safe_sqrt(prod)
    angles = stax._arccos(cosines, do_backprop)
    dot_sigma = (1 - angles/np.pi)
    ker_mat = (stax._sqrt(1 - cosines**2, do_backprop) / np.pi
               + dot_sigma*cosines)
    return ker_mat


@_layer
def CorrelationRelu(do_backprop=False):
    "Returns the correlation, not the covariance, from a ReLU"
    def kernel_fn(kernels):
        var1, nngp, var2, ntk, marginal = \
            kernels.var1, kernels.nngp, kernels.var2, kernels.ntk, kernels.marginal
        if ntk is not None:
            raise NotImplementedError
        prod11, prod12, prod22 = stax._get_normalising_prod(var1, var2, marginal, kernels.var_slices)
        nngp = proj_relu_kernel(nngp, prod12, do_backprop)
        return kernels._replace(
            var1=var1, nngp=nngp, var2=var2, ntk=ntk, is_gaussian=False,
            marginal=marginal)
    return elementwise_init_fn, elementwise_apply_fn, kernel_fn


@_layer
def TickSweep(model, W_covs_list):
    orig_init_fn, orig_apply_fn, orig_kernel_fn = model
    dense_init, dense_apply, dense_kernel = stax.serial(stax.Flatten(), stax.Dense(1))

    def init_fn(rng, input_shape):
        rng = jax.random.split(rng, 2)
        out_shape, params = orig_init_fn(rng[0], input_shape)
        out_dense, _ = dense_init(rng[1], out_shape)
        return [out_dense]*(len(W_covs_list) + 2), params

    def apply_fn(*args, **kwargs):
        raise NotImplementedError

    def f(W_cov):
        a = W_cov.ravel()
        return a * (1/a.sum())
    W_covs_T = [f(W_cov.transpose((2, 3, 0, 1)))
                for W_cov in W_covs_list]
    W_covs = [f(W_cov) for W_cov in W_covs_list]

    def kernel_fn(kernel):
        kernel = orig_kernel_fn(kernel)
        d1, d2, h, _, w, _ = kernel.nngp.shape

        nngp = kernel.nngp.reshape((d1, d2, -1))
        var1 = kernel.var1.reshape((d1, -1))
        var2 = (None if kernel.var2 is None else kernel.var2.reshape((d2, -1)))

        meanpool_nngp = nngp.mean(-1)
        meanpool_var1 = var1.mean(-1)
        meanpool_var2 = (None if var2 is None else var2.mean(-1))
        if kernel.is_height_width:
            tick_nngp = [nngp @ W_cov_T for W_cov_T in W_covs_T]
            tick_var1 = [var1 @ W_cov_T for W_cov_T in W_covs_T]
            tick_var2 = [(None if var2 is None else var2 @ W_cov_T)
                         for W_cov_T in W_covs_T]
        else:
            tick_nngp = [nngp @ W_cov for W_cov in W_covs]
            tick_var1 = [var1 @ W_cov for W_cov in W_covs]
            tick_var2 = [(None if var2 is None else var2 @ W_cov)
                         for W_cov in W_covs]
        return [
            dense_kernel(kernel),
            *[kernel._replace(nngp=n, var1=v1, var2=v2, ntk=None)
              for (n, v1, v2) in zip(tick_nngp, tick_var1, tick_var2)],
            kernel._replace(nngp=meanpool_nngp, var1=meanpool_var1, var2=meanpool_var2, ntk=None),
        ]
    setattr(kernel_fn, _INPUT_REQ, {'marginal': M.OVER_POINTS,
                                    'cross': M.NO,
                                    'spec': "NHWC"})
    return init_fn, apply_fn, kernel_fn
