import math
import itertools
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sacred
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cnn_limits.sacred_utils as SU
import gpytorch
from cnn_limits.precomputed_kernel import PrecomputedKernel
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.variational import \
    NaturalVariationalDistribution, TrilNaturalVariationalDistribution
from nigp.torch_lbfgs import LBFGSScipy

from cnn_limits.classify_utils import (dataset_targets, dataset_full,
                                       centered_one_hot, fold_idx,
                                       balanced_data_indices)


experiment = sacred.Experiment("non_gauss_lik", [SU.ingredient])
if __name__ == '__main__':
    SU.add_file_observer(experiment)


@experiment.config
def config():
    kernel_matrix_path = "/scratch/ag919/logs/save_new/166"

    N_classes = 10
    N_quadrature_points = 20

    multiply_var = False
    apply_relu = False
    training_iter=20
    likelihood_type = "robustmax"
    lr = 25.6
    optimizer_type = "sgd"
    line_search_fn=None
    variational_dist_type = "trilnat"
    variational_strategy_type = "unwhitened"
    momentum = 0.9
    load_file = None

    use_cuda = True
    sigy = 1.36301994e-05


class NaturalClassifier(gpytorch.models.ApproximateGP):
    @experiment.capture
    def __init__(self, inducing_points, num_classes, base_kernel, variational_dist_type, variational_strategy_type):
        if variational_dist_type == "nat":
            klass = NaturalVariationalDistribution
        elif variational_dist_type == "trilnat":
            klass = TrilNaturalVariationalDistribution
        elif variational_dist_type == "chol":
            klass = gpytorch.variational.CholeskyVariationalDistribution
        else:
            raise ValueError(f"variational_dist_type={variational_dist_type}")
        v_dist = klass(inducing_points.size(0), torch.Size([num_classes]))
        if variational_strategy_type == "unwhitened":
            v_strat_type = gpytorch.variational.UnwhitenedVariationalStrategy
        elif variational_strategy_type == "whitened":
            v_strat_type = gpytorch.variational.WhitenedVariationalStrategy
        else:
            raise ValueError(f"variational_strategy_type={variational_strategy_type}")

        v_strat = gpytorch.variational.MultitaskVariationalStrategy(
            v_strat_type(
                self, inducing_points, v_dist, learn_inducing_locations=False),
            num_tasks=num_classes)

        super().__init__(v_strat)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
        self.covar_module.outputscale = 4.
        # self.covar_module = base_kernel

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x),
                                  self.covar_module(x))


class ExactRegressor(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        return (
            MultivariateNormal(self.mean_module(x),
                               self.covar_module(x)))

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
def optimize(closure, parameters, model, likelihood, n_train, test_Y, optimizer_type, training_iter, lr, momentum, line_search_fn, _log, load_file):
    if load_file is not None:
        model_sd, lik_sd = torch.load(load_file)
        model.load_state_dict(model_sd)
        likelihood.load_state_dict(lik_sd)
        _log.info(f"Loaded from {load_file}")

    if optimizer_type == "lbfgsscipy":
        one_step = True
        optimizer = LBFGSScipy(parameters, max_iter=training_iter, history_size=20)
    elif optimizer_type == "lbfgs":
        one_step = True
        optimizer = torch.optim.LBFGS(
            parameters,
            max_iter=training_iter,
            lr=lr,
            history_size=20,
            line_search_fn=line_search_fn)
    elif optimizer_type == "sgd":
        one_step = False
        gamma = 1.2
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    elif optimizer_type == "asgd":
        one_step = False
        optimizer = torch.optim.ASGD(parameters, lr=lr)
    elif optimizer_type == "adam":
        one_step = False
        gamma = 1.1
        optimizer = torch.optim.Adam(parameters, lr=lr, amsgrad=False)
    elif optimizer_type == "amsgrad":
        one_step = False
        optimizer = torch.optim.Adam(parameters, lr=lr, amsgrad=True)
    else:
        raise ValueError(f"optimizer_type={optimizer_type}")

    # optimizer2 = torch.optim.Adam(likelihood.parameters(), lr=1.0)
    def _closure(i):
        optimizer.zero_grad()
        # optimizer2.zero_grad()
        loss = closure()
        loss.backward()
        grad_norm = 0.
        for group in optimizer.param_groups:
            lr = group['lr']
            for p in group['params']:
                _flat_grad = p.grad.view(-1)
                grad_norm += (_flat_grad@_flat_grad).item()
        grad_norm = math.sqrt(grad_norm)
        # for group in optimizer.param_groups:
        #     group['lr'] = lr/min(1., grad_norm)
        try:
            epsilon = likelihood.epsilon
        except AttributeError:
            epsilon = 0.
        _log.info(f"i={i}, ELBO={-loss.item()}, grad_norm={grad_norm}, epsilon={epsilon}, lr={lr}")
        return loss

    if one_step:
        optimizer.step(_closure)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma, last_epoch=-1)
        for i in range(training_iter):
            _closure(i)
            optimizer.step()
            # optimizer2.step()
            try:
                # For NaturalVariationalDistribution only
                model.variational_strategy.base_variational_strategy._variational_distribution.reparameterise()
            except AttributeError:
                pass
            scheduler.step()
            for group in optimizer.param_groups:
                group['lr'] = min(model.covar_module.base_kernel.train_x.size(0), group['lr'])
            if i % 10 == 0:
                device = next(iter(model.parameters())).device
                model.cpu().eval()
                with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
                    try:
                        del model.variational_strategy.base_variational_strategy._mean_cache
                    except AttributeError as e:
                        print("AttributeError when deleting:", e)
                    preds = model(model.covar_module.test_x).mean.cpu().numpy()
                    acc = (preds.argmax(-1) == test_Y).mean()
                _log.info(f"Accuracy at {i}: {acc}")
                torch.save((model.state_dict(), likelihood.state_dict()), SU.base_dir()/f"model_{i}.pt")
                model.to(device=device).train()


@experiment.capture
def do_one_N(Kxx, Kxt, Kx_diag, Kt_diag, train_Y, test_Y, save_fname, jitter, _log, N_classes, N_quadrature_points, training_iter, use_cuda, likelihood_type, lr, optimizer_type):
    # Use the SVGP. in GPytorch, it automatically does not recompute the
    # covariance matrix if you pass it the inducing points.
    n_train, n_test = Kxt.shape

    kernel = PrecomputedKernel(
        *map(torch.from_numpy, (Kxx, Kxt, Kx_diag, Kt_diag)), jitter=jitter)
    likelihood = build_likelihood()
    model = NaturalClassifier(kernel.train_x, N_classes, kernel)
    # model = ExactRegressor(kernel.train_x, torch.nn.functional.one_hot(train_Y).to(kernel.Kxx).t(), likelihood, kernel)
    model.train()
    likelihood.train()

    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, n_train)

    train_Y = torch.from_numpy(train_Y)
    if use_cuda:
        model = model.cuda()
        likelihood = likelihood.cuda()
        mll = mll.cuda()
        train_Y = train_Y.cuda()
    del kernel
    kernel = model.covar_module

    if likelihood_type == "gaussian":
        _target_train_Y = torch.nn.functional.one_hot(train_Y).to(kernel.Kxx)
    else:
        _target_train_Y = train_Y

    optimize(lambda: -mll(model(kernel.train_x), _target_train_Y).sum(),
             model.variational_strategy.parameters(),
             model=model, likelihood=likelihood, n_train=n_train, test_Y=test_Y)
    with torch.no_grad():
        elbo = mll(model(kernel.train_x), _target_train_Y).sum()
        _log.info(f"Final model ELBO: {elbo.item()}")
        # model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.data.copy_(F.one_hot(train_Y).t())
        # print("model ELBO after mean set: ", mll(model(kernel.train_x), train_Y))

    torch.save((model.state_dict(), likelihood.state_dict()), save_fname)

    model = model.eval().cpu()
    likelihood = likelihood.eval().cpu()
    with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
        preds = model(kernel.test_x).mean.numpy()
        acc = (preds.argmax(-1) == test_Y).mean()
    import pdb; pdb.set_trace()
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

    acc = do_one_N(*(a.detach().numpy() for a in [Kxx, Kxt, Kx_diag, Kt_diag, y, yt]),
                   "/tmp/shitt.pt", use_cuda=False)
    print("Accuracy: ", acc)


@experiment.automain
def main(kernel_matrix_path, _log):
    kernel_matrix_path = Path(kernel_matrix_path)
    train_set, test_set = SU.load_sorted_dataset(
        dataset_treatment="load_train_idx",
        train_idx_path=kernel_matrix_path)

    train_Y = dataset_targets(train_set)
    test_Y = dataset_targets(test_set)

    with open(kernel_matrix_path/"config.json", "r") as src,\
         open(SU.base_dir()/"old_config.json", "w") as dst:
        dst.write(src.read())

    with h5py.File(kernel_matrix_path/"kernels.h5", "r") as f:
        N_layers, N_total, _ = f['Kxx'].shape

        all_N = [2560]
        # all_N = list(itertools.takewhile(
        #     lambda a: a <= N_total,
        #     (2**i * 10 for i in itertools.count(0))))
        data = pd.DataFrame(index=[40], columns=all_N)
        accuracy = pd.DataFrame(index=data.index, columns=all_N)

        for layer in reversed(data.index):
            _log.info("Reading Kxx...")
            Kxx = f['Kxx'][layer].astype(np.float64)
            # mask = np.triu(np.ones(Kxx.shape, dtype=np.bool), k=1)
            mask = np.isnan(Kxx)
            Kxx[mask] = Kxx.T[mask]
            assert np.allclose(Kxx, Kxx.T)

            _log.info("Reading Kxt...")
            Kxt = f['Kxt'][layer].astype(np.float64)
            try:
                Kx_diag = f['Kx_diag'][layer].astype(np.float64)
                Kt_diag = f['Kt_diag'][layer].astype(np.float64)
            except KeyError:
                Kx_diag = np.diag(Kxx)

            for N in reversed(data.columns):
                this_N_permutation = balanced_data_indices(train_Y)
                train_idx = this_N_permutation[:N]
                this_Kxx = Kxx[train_idx, :][:, train_idx]

                min_eigval = float(np.linalg.eigvalsh(this_Kxx).min())
                if min_eigval < 0:
                    jitter = (1 + 1/128) * abs(min_eigval)
                else:
                    jitter = 0.
                print(f"min_eigval={min_eigval}, jitter={jitter}")

                with gpytorch.settings.diagonal_jitter(0.):
                    data.loc[layer, N], accuracy.loc[layer, N] = do_one_N(
                        this_Kxx,
                        Kxt[train_idx],
                        Kx_diag[train_idx], Kt_diag,
                        train_Y[train_idx], test_Y,
                        f"_{{step}}_layer_{layer}_N_{N}.pt",
                        jitter=jitter)

                (sigy, acc) = map(np.squeeze, accuracy.loc[layer, N])
                _log.info(f"For layer={layer}, N={N}, sigy={sigy}; accuracy={acc}")

                # print(accuracy_eig(
                #     Kxx[train_idx, train_idx],
                #     Kxt[train_idx], np.eye(10)[train_Y[train_idx]],
                #     test_Y, [sigy]))

                pd.to_pickle(data, SU.base_dir()/"grid_acc.pkl.gz")
                pd.to_pickle(accuracy, SU.base_dir()/"accuracy.pkl.gz")
