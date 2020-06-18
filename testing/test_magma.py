import unittest

import numpy as np

from cnn_limits.magma import syevd


class MagmaTest(unittest.TestCase):
    def one_test_syevd(self, dtype, N=1000):
        A = np.random.randn(N, N)
        A_orig = (A@A.T).astype(dtype)
        np_vals, np_vecs = np.linalg.eigh(A_orig)
        rtol = (1e-3 if dtype==np.float32 else 1e-8)

        A = np.tril(A_orig)
        res = syevd(A, vectors=False, lower=True)
        assert np.allclose(res.vals, np_vals, rtol=rtol)

        A = np.triu(A_orig)
        res = syevd(A, vectors=True, lower=False)
        assert np.allclose(res.vals, np_vals, rtol=rtol)
        for i in range(N):
            assert (np.allclose(res.vecs[:, i], np_vecs[:, i], rtol=rtol)
                    or np.allclose(res.vecs[:, i], -np_vecs[:, i], rtol=rtol))

    def test_syevd(self):
        for dtype in [np.float64, np.float32]:
            self.one_test_syevd(dtype)


    def test_syevd_negative(self, N=1000):
        Q, _ = np.linalg.qr(np.random.randn(N, N))
        eig = np.array(sorted(np.random.randn(N)))
        A = (Q * eig) @ Q.T
        # A is a symmetric matrix with negative eigenvalues.
        res = syevd(A, vectors=True)
        assert np.allclose(eig, res.vals)
        for i in range(N):
            assert (np.allclose(res.vecs[:, i], Q[:, i])
                    or np.allclose(res.vecs[:, i], -Q[:, i]))
