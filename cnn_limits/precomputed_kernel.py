import torch
import math
import gpytorch
from gpytorch.lazy import CholLazyTensor, NonLazyTensor

class PrecomputedKernel(gpytorch.kernels.Kernel):
    def __init__(self, Kxx, Kxt, Kx_diag, Kt_diag, jitter=0.0):
        super().__init__()

        n_train, n_test = Kxt.shape
        train_x = torch.arange(n_train)[:, None].to(Kxx.dtype)
        test_x = torch.arange(n_train, n_train+n_test)[:, None].to(train_x)

        if torch.any(torch.isnan(Kxx)):
            mask = torch.triu(torch.ones(Kxx.shape, dtype=torch.bool))
            Kxx[mask] = Kxx.t()[mask]

        self.register_buffer("train_x", train_x)
        self.register_buffer("test_x", test_x)
        self.register_buffer("Kxx", Kxx)
        self.register_buffer("Kxx_chol", NonLazyTensor(Kxx).add_jitter(jitter).cholesky().evaluate())
        self.register_buffer("Kxt", Kxt)
        self.register_buffer("Kx_diag", Kx_diag)
        self.register_buffer("Kt_diag", Kt_diag)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False):
        if diag:
            if torch.equal(x1, self.train_x):
                return self.Kx_diag
            elif torch.equal(x1, self.test_x):
                return self.Kt_diag
        else:
            if torch.equal(x1, self.train_x):
                if torch.equal(x2, self.train_x):
                    return CholLazyTensor(self.Kxx_chol)
                elif torch.equal(x2, self.test_x):
                    return self.Kxt
            elif torch.equal(x1, self.test_x):
                if torch.equal(x2, self.train_x):
                    return self.Kxt.t()
                elif torch.equal(x1, self.test_x):
                    sz = self.Kt_diag.size()
                    K = torch.empty((*sz[:-1], sz[-1], sz[-1]))
                    K[...] = math.nan
                    K.diagonal(dim1=-2, dim2=-1).copy_(self.Kt_diag)
                    return K
            elif torch.equal(x1, torch.cat((self.train_x, self.test_x), 0))\
                    and torch.equal(x1, x2):
                n_train = self.train_x.size(0)
                n_test = self.test_x.size(0)
                n_total = n_train + n_test
                K = torch.empty((n_total, n_total), dtype=self.Kxx.dtype, device=self.Kxx.device)
                K[:n_train, :n_train] = self.Kxx
                K[:n_train, n_train:] = self.Kxt
                K[n_train:, :n_train] = self.Kxt.t()
                K[n_train:, n_train:] = math.nan
                K[n_train:, n_train:].diagonal(dim1=-2, dim2=-1).copy_(self.Kt_diag)
                return K
        raise NotImplementedError
