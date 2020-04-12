import logging
import unittest

import numpy as np
import torch

from experiments.predict import likelihood_cholesky_kernel
from gpytorch.distributions import MultivariateNormal

torch.set_default_dtype(torch.float64)


class LikTest(unittest.TestCase):
    def test_lck_dual(self, N=10, E=4, D=3):
        F = torch.randn(N, E)
        F_test = torch.randn(N+1, E)
        Y = torch.randn(N, D)
        FF = F.t() @ F
        FY = F.t() @ Y

        Kxx = F@F.t()
        Kxt = F @ F_test.t()
        sigy, lik, FtL, Ly, (grid, likelihoods) = likelihood_cholesky_kernel(
            FF.numpy(), F_test.t().numpy(), Y.numpy(), 100, logging, FY=FY.numpy())
        with torch.no_grad():
            dist = MultivariateNormal(torch.zeros(N), Kxx + sigy*torch.eye(N))
            lik_torch = dist.log_prob(Y.t()).sum()
            L = dist.lazy_covariance_matrix.cholesky().evaluate()
            FtL_torch = torch.triangular_solve(Kxt, L, upper=False).solution
            Ly_torch = torch.triangular_solve(Y, L, upper=False).solution

            assert np.allclose(lik_torch.numpy(), lik)
            assert np.allclose(FtL @ Ly,
                                (FtL_torch.t() @ Ly_torch).numpy())

    def test_lck_primal(self, N=10, D=19):
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

    def test_lck_consistent(self, N=10, E=4, D=3):
        F = np.random.randn(N, E)
        F_test = np.random.randn(N+1, E)
        Y = np.random.randn(N, D)
        FF = F.T @ F
        FY = F.T @ Y
        Kxx = F@F.T
        Kxt = F @ F_test.T

        sigy1, lik1, FtL1, Ly1, (grid1, likelihoods1) = likelihood_cholesky_kernel(
            FF, F_test.T, Y, 5000, logging, FY=FY)
        sigy2, lik2, FtL2, Ly2, (grid2, likelihoods2) = likelihood_cholesky_kernel(
            Kxx, Kxt, Y, 5000, logging, FY=None)

        assert np.allclose(FtL1@Ly1, FtL2@Ly2)
        assert np.allclose(sigy1, sigy2)
        assert np.allclose(lik1, lik2)
