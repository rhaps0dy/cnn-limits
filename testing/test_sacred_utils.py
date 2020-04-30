import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

import cnn_limits.sacred_utils as SU


def preprocess(train, test, zca_bias=0.0001, return_weights=True):
    origTrainShape = train.shape
    origTestShape = test.shape

    train = np.ascontiguousarray(train, dtype=np.float32).reshape(train.shape[0], -1).astype('float64')
    test = np.ascontiguousarray(test, dtype=np.float32).reshape(test.shape[0], -1).astype('float64')


    nTrain = train.shape[0]

    # Zero mean every feature
    train = train - np.mean(train, axis=1)[:,np.newaxis]
    test = test - np.mean(test, axis=1)[:,np.newaxis]

    # Normalize
    # train_norms = np.linalg.norm(train, axis=1)
    # test_norms = np.linalg.norm(test, axis=1)
    train_norms = np.sqrt(np.sum(np.square(train), 1) + 1e-16)
    test_norms = np.sqrt(np.sum(np.square(test), 1) + 1e-16)

    # Make features unit norm
    train = train/train_norms[:,np.newaxis]
    test = test/test_norms[:,np.newaxis]

    data_means = np.mean(train, axis=1)


    trainCovMat = 1.0/nTrain * train.T.dot(train)

    (E,V) = np.linalg.eig(trainCovMat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E)
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)

    train = (train).dot(global_ZCA)
    test = (test).dot(global_ZCA)
    if return_weights:
        return (train.reshape(origTrainShape).astype('float64'), test.reshape(origTestShape).astype('float64')), global_ZCA
    else:
        return (train.reshape(origTrainShape).astype('float64'), test.reshape(origTestShape).astype('float64'))



class ZCATest(unittest.TestCase):
    def test_do_transforms(self):
        train, test = SU.load_dataset("CIFAR10", "/scratch/ag919/datasets")
        train = Subset(train, range(32*32*3+100))
        test = Subset(test, range(32*32*3+100))

        X_orig, y_orig = SU.whole_dset(train)
        Xt_orig, yt_orig = SU.whole_dset(test)
        X_orig = X_orig[:, :, :10, :10]
        Xt_orig = Xt_orig[:, :, :10, :10]

        train = TensorDataset(X_orig, y_orig)
        test = TensorDataset(Xt_orig, yt_orig)

        train, test, W = SU.do_transforms(train, test, ZCA=True, ZCA_bias=1e-5)
        X, y = SU.whole_dset(train)
        Xt, yt = SU.whole_dset(test)

        assert torch.equal(y_orig, y)
        assert torch.equal(yt_orig, yt)

        (X_ref, Xt_ref), W_ref = preprocess(X_orig.numpy(), Xt_orig.numpy(), zca_bias=1e-5)
        # train_ref, test_ref, W_ref = SU.do_transforms(train, test, ZCA=True, ZCA_bias=1e-5, device='cpu')
        # X_ref, _ = SU.whole_dset(train)
        # Xt_ref, _ = SU.whole_dset(test)
        # assert torch.allclose(W, W_ref)
        # assert torch.allclose(X, X_ref)
        # assert torch.allclose(Xt, Xt_ref)

        assert torch.allclose(W, torch.from_numpy(W_ref), atol=5e-3)
        assert torch.allclose(X, torch.from_numpy(X_ref), atol=5e-3)
        assert torch.allclose(Xt, torch.from_numpy(Xt_ref), atol=5e-3)
