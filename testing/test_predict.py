import logging
import unittest

import jax
import jax.numpy as np
import torch

from experiments.predict import (jax_linear_lik, likelihood_cholesky,
                                 likelihood_cholesky_kernel)
from gpytorch.distributions import MultivariateNormal

torch.set_default_dtype(torch.float64)


class LikTest(unittest.TestCase):
    def test_likelihood_kernelised(self, N=10, E=4, D=3):
        F = torch.randn(N, E)
        Y = torch.randn(N, D)
        FF = F.t() @ F
        FY = F.t() @ Y
        y = Y.pow(2).sum()

        jax_FF, jax_FY, jax_y = (np.asarray(a.numpy()) for a in (FF, FY, y))
        lik_fn = jax.value_and_grad(
            jax.partial(jax_linear_lik, jax_FF, jax_FY, jax_y, N))

        log_sigy = 0.8
        value, grad = lik_fn(np.array([log_sigy]))

        torch_log_sigy = torch.tensor([log_sigy], requires_grad=True)
        K = F@F.t() + torch.eye(F.shape[0]) * torch_log_sigy.exp()
        dim_log_lik = MultivariateNormal(torch.zeros(F.shape[0]), K).log_prob(Y.t())
        torch_value = dim_log_lik.sum()
        # A = torch.triangular_solve(Y, torch.cholesky(K), upper=False).solution
        # torch_value = -.5*A.pow(2).sum()
        # torch_value = -.5 * K.det().log()
        # assert torch.allclose(
        #     torch_value,
        #     -.5*(F.t()@F
        #          + torch.eye(F.shape[1])*torch_log_sigy.exp()).det().log() -.5*(F.shape[0]-F.shape[1])*torch_log_sigy)
        torch_value.backward()

        value2, grad2 = (np.asarray(a.detach().numpy()) for a in (torch_value, torch_log_sigy.grad))

        assert np.allclose(value, value2)
        assert np.allclose(grad, grad2)

    def test_max_likelihood(self, N=10, E=4, D=3):
        F = torch.randn(N, E)
        Y = torch.randn(N, D)
        FF = F.t() @ F
        FY = F.t() @ Y
        sigy, lik, FtL, Ly, (grid_x, likelihoods) = likelihood_cholesky(FF, FY, F.t(), Y)
        assert isinstance(FtL, np.ndarray)
        assert isinstance(Ly, np.ndarray)
        assert isinstance(grid_x, np.ndarray)
        assert isinstance(likelihoods, np.ndarray)
        print(sigy, lik)
        import matplotlib.pyplot as plt
        plt.plot(grid_x, likelihoods)
        plt.show()

    def test_likelihood_cholesky_kernel(self, N=10, D=19):
        F = torch.randn(2*N - 5, 2*N)
        Y = torch.randn(N, D)
        Kxx_notril = F[:N] @ F[:N].t()
        Kxx = torch.tril(Kxx_notril)
        Kxt = F[:N] @ F[N:].t()

        sigy, lik, FtL, Ly, (grid, likelihoods) = likelihood_cholesky_kernel(
            Kxx.numpy(), Kxt.numpy(), Y.numpy(), 100, logging)

        with torch.no_grad():
            dist = MultivariateNormal(torch.zeros(N), Kxx_notril + sigy*torch.eye(N))
            lik_torch = dist.log_prob(Y.t()).sum()
            L = dist.lazy_covariance_matrix.cholesky().evaluate()
            FtL_torch = torch.triangular_solve(Kxt, L, upper=False).solution.t()
            Ly_torch = torch.triangular_solve(Y, L, upper=False).solution
        assert np.allclose(lik_torch.numpy(), lik)
        assert np.allclose(FtL_torch.numpy(), FtL)
        assert np.allclose(Ly_torch.numpy(), Ly)
