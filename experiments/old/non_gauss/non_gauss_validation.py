import math
import itertools
import warnings
from pathlib import Path

import h5py
import gc
import numpy as np
import pandas as pd
import sacred
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cnn_limits.sacred_utils as SU
import gpytorch
import nigp
from cnn_limits.layers import proj_relu_kernel
from cnn_limits.natural_lbfgs import LBFGSNat
from cnn_limits.precomputed_kernel import PrecomputedKernel
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from nigp.natural_variational_distribution import \
    NaturalVariationalDistribution, TrilNaturalVariationalDistribution
from nigp.torch_lbfgs import LBFGSScipy

# from experiments.predict_cv_acc import accuracy_eig

experiment = sacred.Experiment("non_gauss_validation", [SU.ingredient])
if __name__ == '__main__':
    SU.add_file_observer(experiment)


@experiment.config
def config():
    kernel_matrix_path = "/scratch/ag919/logs/save_new/166"

    N_classes = 10
    N_quadrature_points = 20

    multiply_var = False
    apply_relu = False
    training_iter=2000
    likelihood_type = "robustmax"
    lr = 0.05
    variational_dist_type = "trilnat"
    momentum = 0.9

    use_cuda = True
    sigy = 1.36301994e-05


def dataset_full(dset):
    X, y = next(iter(DataLoader(dset, batch_size=len(dset))))
    return X, y

def dataset_targets(dset):
    return dataset_full(dset)[1]


class NaturalClassifier(gpytorch.models.ApproximateGP):
    @experiment.capture
    def __init__(self, inducing_points, num_classes, base_kernel, variational_dist_type):
        if variational_dist_type == "nat":
            klass = NaturalVariationalDistribution
        elif variational_dist_type == "trilnat":
            klass = TrilNaturalVariationalDistribution
        elif variational_dist_type == "chol":
            klass = gpytorch.variational.CholeskyVariationalDistribution
        else:
            raise ValueError(f"variational_dist_type={variational_dist_type}")
        v_dist = klass(inducing_points.size(0), torch.Size([num_classes]))
        v_strat = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.WhitenedVariationalStrategy(
                self, inducing_points, v_dist, learn_inducing_locations=False),
            num_tasks=num_classes)

        super().__init__(v_strat)
        self.mean_module = gpytorch.means.ZeroMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.covar_module = base_kernel

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x),
                                  self.covar_module(x))


@experiment.capture
def build_likelihood(N_classes, N_quadrature_points, likelihood_type, sigy):
    if likelihood_type == "robustmax":
        return gpytorch.likelihoods.RobustmaxLikelihood(
            N_classes, epsilon=1e-3, num_quadrature_points=N_quadrature_points)
    elif likelihood_type == "softmax":
        return gpytorch.likelihoods.SoftmaxLikelihood(
            N_classes, N_classes, False)
    elif likelihood_type == "gaussian":
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            N_classes, noise_constraint=gpytorch.constraints.GreaterThan(0.0))
        likelihood.noise = sigy
        likelihood.noise_covar.noise = 0.
        return likelihood
    else:
        raise ValueError(f"likelihood_type={likelihood_type}")


@experiment.capture
def optimize(closure, state_closure, parameters, num_data, save_fname, training_iter, lr, momentum, _log):
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    for step in range(training_iter):
        optimizer.zero_grad()
        loss = closure()
        loss.backward()
        grad_norm = 0.
        for group in optimizer.param_groups:
            for p in group['params']:
                _flat_grad = p.grad.view(-1)
                grad_norm += (_flat_grad@_flat_grad).item()
        grad_norm = math.sqrt(grad_norm)
        # for group in optimizer.param_groups:
        #     group['lr'] = lr/min(1., grad_norm)
        _log.info(f"ELBO={-loss.item()}, grad_norm={grad_norm}")

        torch.save(state_closure(), save_fname)

        optimizer.step()


@experiment.capture
def do_one_N(Kxx, Kxt, Kx_diag, Kt_diag, train_Y, test_Y, save_fname, _log, N_classes, N_quadrature_points, training_iter, use_cuda, likelihood_type, lr, optimizer_type):
    # Use the SVGP. in GPytorch, it automatically does not recompute the
    # covariance matrix if you pass it the inducing points.
    n_train, n_test = Kxt.shape

    kernel = PrecomputedKernel(
        *map(torch.from_numpy, (Kxx, Kxt, Kx_diag, Kt_diag)))
    likelihood = build_likelihood()
    model = NaturalClassifier(kernel.train_x, N_classes, kernel)
    model.train()
    likelihood.train()

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, n_train)

    if use_cuda:
        model = model.cuda()
        likelihood = likelihood.cuda()
        mll = mll.cuda()
        train_Y = train_Y.cuda()
    test_Y = test_Y.cpu().numpy()
    del kernel
    kernel = model.covar_module

    if likelihood_type == "gaussian":
        _target_train_Y = torch.nn.functional.one_hot(train_Y).to(kernel.Kxx)
    else:
        _target_train_Y = train_Y

    optimize(lambda: -mll(model(kernel.train_x), _target_train_Y).sum(),
             lambda: (model.state_dict(), likelihood.state_dict()),
             model.variational_strategy.parameters(),
             len(kernel.train_x), save_fname)
    with torch.no_grad():
        elbo = mll(model(kernel.train_x), _target_train_Y).sum()
        _log.info(f"Final model ELBO: {elbo.item()}")
        # model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.data.copy_(F.one_hot(train_Y).t())
        # print("model ELBO after mean set: ", mll(model(kernel.train_x), train_Y))

    model = model.eval().cpu()
    likelihood = likelihood.eval().cpu()
    with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
        preds = model(kernel.test_x).mean.numpy()
        acc = (preds.argmax(-1) == test_Y).mean()
    return None, ((likelihood.noise.item() if likelihood_type == "gaussian" else ()), acc)


@experiment.command
def debug():
    train_set, test_set = SU.load_sorted_dataset(N_train=50, N_test=10)
    X, y = dataset_full(train_set)
    Xt, yt = dataset_full(train_set)

    X = X.reshape((X.size(0), -1))
    Xt = Xt.reshape((Xt.size(0), -1))

    rbf_k = gpytorch.kernels.RBFKernel()
    rbf_k.lengthscale = 20

    Kx_diag = rbf_k(X, diag=True)
    Kt_diag = rbf_k(Xt, diag=True)
    Kxx = rbf_k(X).evaluate()
    Kxt = rbf_k(X, Xt).evaluate()

    acc = do_one_N(*(a.detach().numpy() for a in [Kxx, Kxt, Kx_diag, Kt_diag]),
                   y, yt, "/tmp/shitt.pt", use_cuda=False)
    print("Accuracy: ", acc)


@experiment.automain
def main(kernel_matrix_path, multiply_var, _log, apply_relu, sigy):
    kernel_matrix_path = Path(kernel_matrix_path)
    train_set, test_set = SU.load_sorted_dataset(
        dataset_treatment="load_train_idx",
        train_idx_path=kernel_matrix_path)

    train_Y = dataset_targets(train_set)
    test_Y = dataset_targets(test_set)

    with h5py.File(kernel_matrix_path/"kernels.h5", "r") as f:
        N_layers, N_total, _ = f['Kxx'].shape

        all_N = list(itertools.takewhile(
            lambda a: a <= N_total,
            (2**i * 10 for i in itertools.count(0))))
        data = pd.DataFrame(index=range(N_layers), columns=all_N)
        accuracy = pd.DataFrame(index=range(N_layers), columns=all_N)

        for layer in reversed(data.index):
            Kxx = f['Kxx'][layer].astype(np.float64)
            mask = np.triu(np.ones(Kxx.shape, dtype=np.bool), k=1)
            Kxx[mask] = Kxx.T[mask]
            assert np.array_equal(Kxx, Kxx.T)
            assert np.all(np.isfinite(Kxx))

            Kxt = f['Kxt'][layer].astype(np.float64)
            try:
                Kx_diag = f['Kx_diag'][layer].astype(np.float64)
                Kt_diag = f['Kt_diag'][layer].astype(np.float64)
            except KeyError:
                Kx_diag = np.diag(Kxx)

            if multiply_var:
                assert np.allclose(np.diag(Kxx), 1.)
                Kxx *= np.sqrt(Kx_diag[:, None]*Kx_diag)
                Kxt *= np.sqrt(Kx_diag[:, None]*Kt_diag)
            else:
                assert not np.allclose(np.diag(Kxx), 1.)

            if apply_relu:
                prod12 = Kx_diag[:, None] * Kx_diag
                Kxx = np.asarray(proj_relu_kernel(Kxx, prod12, False))
                Kxx = Kxx * prod12    # Use covariance, not correlation matrix
                prod12_t = Kx_diag[:, None] * Kt_diag
                Kxt = np.asarray(proj_relu_kernel(Kxt, prod12_t, False))
                Kxt = Kxt * prod12_t

            for N in reversed(data.columns):
                # Made a mistake, the same label are all contiguous in the training set.
                train_idx = slice(0, N_total, N_total//N)
                with gpytorch.settings.diagonal_jitter(0.0):
                    data.loc[layer, N], accuracy.loc[layer, N] = do_one_N(
                        Kxx[train_idx, train_idx],
                        Kxt[train_idx],
                        Kx_diag[train_idx], Kt_diag,
                        train_Y[train_idx], test_Y,
                        f"_{{step}}_layer_{layer}_N_{N}.pt")
                (sigy, acc) = map(np.squeeze, accuracy.loc[layer, N])
                _log.info(f"For layer={layer}, N={N}, sigy={sigy}; accuracy={acc}")

                # print(accuracy_eig(
                #     Kxx[train_idx, train_idx],
                #     Kxt[train_idx], np.eye(10)[train_Y[train_idx]],
                #     test_Y, [sigy]))

                pd.to_pickle(data, SU.base_dir()/"grid_acc.pkl.gz")
                pd.to_pickle(accuracy, SU.base_dir()/"accuracy.pkl.gz")
