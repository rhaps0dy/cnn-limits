import unittest
import jax.numpy as np
import jax

from neural_tangents import stax
from neural_tangents.stax import Padding
from neural_tangents.utils.kernel import Marginalisation as M
from cnn_limits.layers import CorrelatedConv, conv4d_for_5or6d, naive_conv4d_for_5or6d, covariance_tensor, TickSerialCheckpoint
import gpytorch

NO_MARGINAL_ARG = {'marginal': M.OVER_POINTS, 'cross': M.NO, 'spec': "NHWC"}

def covariance_test(v_apply_fn, params, kernel_fn, x1, x2):
    y1 = v_apply_fn(params, x1)
    y2 = v_apply_fn(params, x2)
    N_samples = len(y1)
    assert np.allclose(y1.mean(0), np.zeros_like(y1[1:]), atol=0.1)
    assert np.allclose(y2.mean(0), np.zeros_like(y2[1:]), atol=0.1)

    mom2 = np.einsum("bnhwc,bmgqc->nmhgwq", y1, y2) / N_samples
    mom1 = np.einsum("nhwc,mgqc->nmhgwq", y1.mean(0), y2.mean(0))
    empirical_K = mom2 - mom1
    analytic_K = kernel_fn(x1, x2, 'nngp', NO_MARGINAL_ARG)
    assert np.allclose(empirical_K, analytic_K, atol=0.1)


class CorrelatedConvTest(unittest.TestCase):
    def setUp(self):
        self.key, self.x1, self.x2 = self.random_inputs((5, 5))
    def random_inputs(self, x_shape):
        key = jax.random.PRNGKey(1234)
        key, k1, k2 = jax.random.split(key, 3)
        x1 = jax.random.normal(k1, (2, *x_shape, 1))
        x2 = jax.random.normal(k2, (1, *x_shape, 1))
        return key, x1, x2

    def test_uncorrelated_with_neural_tangents(self):
        shape = (5, 5)
        x1 = np.reshape(np.arange(1*np.prod(shape)).astype(np.float32)*3, (1, *shape, 1))
        x2 = np.reshape(np.arange(1, 1*np.prod(shape)+1).astype(np.float32)*3, (1, *shape, 1))

        kwargs = dict(out_chan=1, filter_shape=(3, 3), strides=(1, 1),
                      padding='SAME', W_std=1.3)
        init_uncorr, _, kernel_uncorr = stax.Conv(**kwargs)
        init_corr, _, kernel_corr = CorrelatedConv(**kwargs)
        umat, is_height_width = kernel_uncorr(x1, x2, ('nngp', 'is_height_width'),
                                              NO_MARGINAL_ARG)
        if not is_height_width:
            # .reshape((2, 1, 3, 3, 3, 3))
            umat = umat.transpose((0, 1, 4, 5, 2, 3))

        cmat, is_height_width = kernel_corr(x1, x2, ('nngp', 'is_height_width'))
        assert is_height_width
        assert np.allclose(umat, cmat)

    def test_neuraltangents_empirical(self, N_samples=100000, filter_shape=(5, 5)):
        kparams = jax.random.split(self.key, N_samples)
        init_fn, apply_fn, kernel_fn = stax.Conv(
            1, filter_shape, (1, 1), 'VALID', W_std=1.2, parameterization='ntk')
        _, params = jax.vmap(init_fn, (0, None), (None, 0))(kparams, self.x1.shape)
        v_apply_fn = jax.vmap(apply_fn, ((0, 0), None))
        covariance_test(v_apply_fn, params, kernel_fn, self.x1, self.x2)


    def test_correlated(self, N_samples=400000, filter_shape=(2, 2)):
        _, kW, kparams = jax.random.split(self.key, 3)
        filter_numel = np.prod(filter_shape)
        W_std = jax.random.normal(self.key, (filter_numel, filter_numel))
        W_cov = W_std @ W_std.T
        W_cov_reshaped = W_cov.reshape((*filter_shape, *filter_shape)).transpose((0, 2, 1, 3))

        W_std_tensor = W_std.reshape((*filter_shape, *filter_shape)).transpose((0, 2, 1, 3))
        W_cov_tensor = np.einsum("ahcw,bhdw->abcd", W_std_tensor, W_std_tensor)
        assert np.allclose(W_cov_reshaped, W_cov_tensor)

        init_fn, apply_fn, kernel_fn = CorrelatedConv(
            1, filter_shape, (1, 1), 'VALID', W_std_tensor, parameterization='ntk')

        _, params_from_init_fn = init_fn(kparams, self.x1.shape)
        params_from_init_fn = np.reshape(params_from_init_fn, filter_shape)

        params_from_raw = jax.random.normal(kparams, (filter_numel,)) @ W_std.T
        params_from_raw = np.reshape(params_from_raw, filter_shape)
        assert np.allclose(params_from_init_fn, params_from_raw)

        params = jax.random.normal(kparams, (N_samples, filter_numel)) @ W_std.T
        params = np.reshape(params, (N_samples, *filter_shape, 1, 1))
        v_apply_fn = jax.vmap(apply_fn, (0, None))
        covariance_test(v_apply_fn, params, kernel_fn, self.x1, self.x2)


def make_tensor_with_shape(shape):
    n = np.prod(shape)
    return np.arange(n).reshape(shape).astype(np.float32)


class Conv4dTest(unittest.TestCase):
    @staticmethod
    def one_shape_test(x_size, strides, padding, cov_shape=(3, 3)):
        mat = make_tensor_with_shape(
            (1, 2, x_size[0], x_size[0], x_size[1], x_size[1]))
        W_cov_tensor = make_tensor_with_shape((cov_shape[0], cov_shape[0], cov_shape[1], cov_shape[1])) + 5
        res1 = naive_conv4d_for_5or6d(mat, W_cov_tensor, strides, padding)
        res2 = conv4d_for_5or6d(mat, W_cov_tensor, strides, padding)
        assert np.allclose(res1, res2)

    def test_conv_same(self):
        for sz in [(4, 4), (8, 8), (7, 4), (6, 4)]:
            for st in [(1, 1), (2, 2), (3, 1)]:
                self.one_shape_test(sz, st, Padding.SAME)

    def test_conv_valid(self):
        for sz in [(4, 4), (8, 8), (7, 4), (6, 4)]:
            for st in [(1, 1), (2, 2), (3, 1)]:
                self.one_shape_test(sz, st, Padding.VALID)

    def test_conv_same_4(self):
        for sz in [(4, 4), (8, 8), (7, 4), (6, 4)]:
            for st in [(1, 1), (2, 2), (3, 1)]:
                self.one_shape_test(sz, st, Padding.SAME, cov_shape=(4, 4))

    def test_conv_valid_4(self):
        for sz in [(4, 4), (8, 8), (7, 4), (6, 4)]:
            for st in [(1, 1), (2, 2), (3, 1)]:
                self.one_shape_test(sz, st, Padding.VALID, cov_shape=(4, 4))

    def test_conv_same_5(self):
        for sz in [(4, 4), (8, 8), (7, 4), (6, 5)]:
            for st in [(1, 1), (2, 2), (3, 1)]:
                self.one_shape_test(sz, st, Padding.SAME, cov_shape=(5, 5))

    def test_conv_valid_5(self):
        for sz in [(5, 5), (8, 8), (7, 5), (6, 5)]:
            for st in [(1, 1), (2, 2), (3, 1)]:
                self.one_shape_test(sz, st, Padding.VALID, cov_shape=(5, 5))


class TickSerialCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.x1 = make_tensor_with_shape((2, 5, 5, 1))
        self.x2 = make_tensor_with_shape((1, 5, 5, 1)) + self.x1.max()

        kern = gpytorch.kernels.MaternKernel(nu=3/2)
        Wcg = []
        for l in (1, 2, 3):
            kern.lengthscale = l
            Wcg.append(covariance_tensor(5, 5, kern))
        self.Wcg = Wcg

        self.init_fns, self.apply_fns, self.kernel_fns = zip(
            stax.serial(stax.Conv(3, (3, 3), padding='SAME'), stax.Relu()),
            stax.serial(stax.Conv(3, (3, 3), padding='SAME'), stax.Relu()),
            stax.serial(stax.Conv(3, (3, 3), padding='SAME'), stax.Relu()),
        )
        self.checkpoint_init, self.checkpoint_apply, self.checkpoint_kern = TickSerialCheckpoint(
            *zip(self.init_fns, self.apply_fns, self.kernel_fns, self.Wcg))

    def test_equal_to_separate(self):
        x1, x2 = self.x1, self.x2
        init_fns, apply_fns, kernel_fns = self.init_fns, self.apply_fns, self.kernel_fns

        kernels_checkpointed = self.checkpoint_kern(x1, x2, get='nngp')
        kernels_avg_checkpointed = kernels_checkpointed[::3]
        kernels_tick_checkpointed = kernels_checkpointed[1::3]
        kernels_dense_checkpointed = kernels_checkpointed[2::3]

        for i, k_checkpointed in enumerate(kernels_avg_checkpointed):
            _, _, kfn = stax.serial(
                *zip(init_fns[:i+1], apply_fns[:i+1], kernel_fns[:i+1]),
                stax.GlobalAvgPool())
            k = kfn(x1, x2, get='nngp')
            assert np.allclose(k_checkpointed, k)

        for i, (k_checkpointed, W) in enumerate(zip(kernels_tick_checkpointed, self.Wcg)):
            _, _, kfn = stax.serial(
                *zip(init_fns[:i+1], apply_fns[:i+1], kernel_fns[:i+1]))
            k = kfn(x1, x2, get='nngp', marginalization=NO_MARGINAL_ARG)
            k = (W * k).sum((-4, -3, -2, -1))
            assert np.allclose(k_checkpointed, k)

        for i, k_checkpointed in enumerate(kernels_dense_checkpointed):
            _, _, kfn = stax.serial(
                *zip(init_fns[:i+1], apply_fns[:i+1], kernel_fns[:i+1]),
                stax.Flatten())
            k = kfn(x1, x2, get='nngp')
            assert np.allclose(k_checkpointed, k)


    def test_init(self, N=2):
        output_shape, _ = self.checkpoint_init(jax.random.PRNGKey(123), (N, 5, 5, 2))
        assert output_shape == [(3, N, 1)]*3

    def test_apply(self):
        all_shapes, all_params = self.checkpoint_init(jax.random.PRNGKey(123), self.x1.shape)
        out_checkpointed = self.checkpoint_apply(all_params, self.x1)
        _, dense_apply, _ = stax.Dense(1)

        params, readout_params = zip(*all_params)

        for i, out_ck in enumerate(out_checkpointed):
            out_ck_avg, out_ck_tick, out_ck_dense = out_ck[0, ...], out_ck[1, ...], out_ck[2, ...]
            assert out_ck.shape == all_shapes[i]

            _, afn, _ = stax.serial(*zip(self.init_fns[:i+1], self.apply_fns[:i+1], self.kernel_fns[:i+1]))
            out = afn(params[:i+1], self.x1)

            out_avg = dense_apply(readout_params[i][0][1], out.mean((-3, -2)))
            assert np.allclose(out_ck_avg, out_avg)

            out_tick = (out[..., None] * readout_params[i][1]).sum((-4, -3, -2))
            assert np.allclose(out_ck_tick, out_tick)

            out_dense = dense_apply(readout_params[i][2][1], out.reshape(out.shape[0], -1))
            assert np.allclose(out_ck_dense, out_dense)





