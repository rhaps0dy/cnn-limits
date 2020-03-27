import unittest
import jax.numpy as np
import jax

from neural_tangents import stax
from neural_tangents.utils.kernel import Marginalisation as M
from cnn_limits.layers import CorrelatedConv

NO_MARGINAL_ARG = {'marginal': M.NO, 'cross': M.NO, 'spec': "NHWC"}

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
            1, filter_shape, (1, 1), 'VALID', W_std_tensor, 'ntk')

        _, params_from_init_fn = init_fn(kparams, self.x1.shape)
        params_from_init_fn = np.reshape(params_from_init_fn, filter_shape)

        params_from_raw = jax.random.normal(kparams, (filter_numel,)) @ W_std.T
        params_from_raw = np.reshape(params_from_raw, filter_shape)
        assert np.allclose(params_from_init_fn, params_from_raw)

        params = jax.random.normal(kparams, (N_samples, filter_numel)) @ W_std.T
        params = np.reshape(params, (N_samples, *filter_shape, 1, 1))
        v_apply_fn = jax.vmap(apply_fn, (0, None))
        covariance_test(v_apply_fn, params, kernel_fn, self.x1, self.x2)




