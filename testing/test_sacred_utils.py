import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import cnn_limits.sacred_utils as SU


class ZCATest(unittest.TestCase):
    def test_do_transforms(self):
        train, test = SU._load_dataset("CIFAR10", "/scratch/ag919/datasets")
        # train = Subset(train, range(32*32*3+100))
        # test = Subset(test, range(32*32*3+100))

        X_orig, y_orig = SU.whole_dset(train)
        _, yt_orig = SU.whole_dset(test)

        train, test = SU.do_transforms(train, test, ZCA=True)
        X, y = SU.whole_dset(train)
        _, yt = SU.whole_dset(test)

        assert torch.equal(y_orig, y)
        assert torch.equal(yt_orig, yt)

        np.save("X.npy", X[0].transpose(0, -1).numpy())
        np.save("X_orig.npy", X_orig[0].transpose(0, -1).numpy())

        X_flat = X.reshape((len(X), -1)).to(torch.float64)
        should_eye = X_flat.t() @ X_flat
        eye = torch.eye(X_flat.size(1), dtype=torch.float64)

        assert torch.allclose(should_eye, eye)
