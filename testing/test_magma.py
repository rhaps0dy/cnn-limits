import unittest

import numpy as np

from cnn_limits.magma import eigh


class MagmaTest(unittest.TestCase):
    def one_test_eigh(self, dtype, N=10):
        A = np.random.randn(N, N)
        A_orig = (A@A.T).astype(dtype)
        np_vals, np_vecs = np.linalg.eigh(A_orig)
        rtol = (1e-3 if dtype==np.float32 else 1e-8)

        A = np.tril(A_orig)
        res = eigh(A, vectors=False, lower=True)
        assert np.allclose(res.vals, np_vals, rtol=rtol)

        A = np.triu(A_orig)
        res = eigh(A, vectors=True, lower=False)
        assert np.allclose(res.vals, np_vals, rtol=rtol)
        for i in range(N):
            assert (np.allclose(res.vecs[:, i], np_vecs[:, i], rtol=rtol)
                    or np.allclose(res.vecs[:, i], -np_vecs[:, i], rtol=rtol))

    def test_eigh(self):
        for dtype in [np.float64, np.float32]:
            self.one_test_eigh(dtype)
