import torchvision
import torch
from torch.utils.data import DataLoader
import os

import gpytorch
import pickle_utils as pu
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal

import math
import sacred
from nigp import artifacts, tbx, plot
import nigp.variational, nigp.models

experiment = sacred.Experiment('tick_gp', [artifacts.ingredient, tbx.ingredient,])
if __name__ == '__main__':
   experiment.observers.append(sacred.observers.FileStorageObserver("logs"))


@experiment.pre_run_hook
def hook(num_likelihood_samples, default_dtype):
    gpytorch.settings.num_likelihood_samples._set_value(num_likelihood_samples)
    gpytorch.settings.max_cholesky_size._set_value(100000)  # disable CG, it makes eigenvalues negative :(
    torch.set_default_dtype(getattr(torch, default_dtype))


@experiment.post_run_hook
def hook(_run):
    print(f"This was run {_run._id}")


@experiment.config
def _config():
    dataset_name = "MNIST"
    dataset_base_path = "/scratch/ag919/datasets/"
    num_inducing = 1000
    patch_size = (5, 5)
    lr_init = 0.01
    lr_time_decay = -0.2
    loc_lengthscale_init = 3.

    epochs = 100
    learn_inducing_locations = True
    batch_size = 128

    num_likelihood_samples = 5
    default_dtype = 'float32'
    use_cuda = True

    model_type = "tick"

    load_model_from = None



@experiment.capture
def dataset(dataset_name, dataset_base_path):
    if dataset_name == "CIFAR10":
        dataset_base_path = os.path.join(dataset_base_path, dataset_name)
        trans = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
        ])
    else:
        trans = torchvision.transforms.ToTensor()

    _dset = getattr(torchvision.datasets, dataset_name)
    train = _dset(dataset_base_path, train=True, download=True, transform=trans)
    test = _dset(dataset_base_path, train=False, transform=trans)
    return train, test


def init_inducing_patches(train_set, N, patch_size):
    "Select patches from the training set uniformly at random"
    idx = torch.randint(low=0, high=len(train_set), size=(N,))
    train_x = torch.stack([train_set[i][0] for i in idx], dim=0)

    patches = torch.nn.functional.unfold(train_x, patch_size)
    patch_idx = torch.randint(low=0, high=patches.size(-1), size=(N, 1, 1))
    Z = torch.gather(patches, dim=2, index=patch_idx.repeat(1, patch_size.numel(), 1))

    pos_h_max = train_x.size(-2) - patch_size[-2] + 1
    pos_w_max = train_x.size(-1) - patch_size[-1] + 1
    "Select patch positions for them uniformly at random"
    Z_pos = torch.rand(N, 2) * torch.tensor([pos_h_max, pos_w_max])
    return torch.cat([Z_pos, Z.squeeze(2)], dim=1)


class ConvDomainVStrat(nigp.variational.TrueWhitenedVariationalStrategy):
    def __init__(self, model, image_size, patch_size, inducing_patches,
                 variational_distribution, learn_inducing_locations=False):
        super().__init__(model, inducing_patches, variational_distribution,
                         learn_inducing_locations=False)
        self.patch_size = patch_size
        self.image_size = image_size

        n_patches_h = image_size[-2] - patch_size[-2] + 1
        n_patches_w = image_size[-1] - patch_size[-1] + 1

        pos_h = torch.arange(n_patches_h).to(torch.get_default_dtype())
        pos_h = pos_h.unsqueeze(1).repeat(1, n_patches_w)
        pos_w = torch.arange(n_patches_w).to(torch.get_default_dtype())
        pos_w = pos_w.unsqueeze(0).repeat(n_patches_h, 1)
        img_pos = torch.stack([pos_h, pos_w], dim=2).view(-1, 2).contiguous()
        self.register_buffer('img_pos', img_pos)

    @property
    @gpytorch.utils.memoize.cached(name="prior_distribution_memo")
    def nonwhite_prior(self):
        z = (self.inducing_points[..., :2], self.inducing_points[..., 2:])
        return self.model.forward(z)

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None):
        Lzz = self.Kzz.cholesky().evaluate()
        assert not torch.equal(x, inducing_points), "This is inter-domain!"

        x_patches = (torch.nn.functional.unfold(x, self.patch_size)
                     .transpose(-1, -2))  # B,n_patches,patch_size
        _x = (self.img_pos, x_patches)
        _z = (self.inducing_points[..., :2], self.inducing_points[..., 2:])

        K_xpatch = self.model.covar_module(_x)
        K_xdiag = K_xpatch.evaluate().sum((-1, -2))   # Double sum over patches
        # The real distribution isn't diagonal but for VI purposes this is sufficient
        p_f = MultivariateNormal(torch.zeros_like(K_xdiag), gpytorch.lazy.DiagLazyTensor(K_xdiag))

        K_xz = self.model.covar_module(_z, _x).sum(-1)  # Sum over patches
        Kzx = gpytorch.lazy.NonLazyTensor(K_xz.transpose(-1, -2))
        batch_mvn = nigp.variational.white_conditional(
            p_f, Lzz, Kzx, inducing_values, variational_inducing_covar)
        return batch_mvn

    def __call__(self, x, prior=False):
        "copied from _VariationalStrategy.__call__ mostly"
        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x)

        # Delete previously cached items from the training distribution
        if self.training:
            if hasattr(self, "_memoize_cache"):
                delattr(self, "_memoize_cache")
                self._memoize_cache = dict()
        # (Maybe) initialize variational distribution
        if not self.variational_params_initialized.item():
            prior_dist = self.prior_distribution
            self._variational_distribution.initialize_variational_distribution(prior_dist)
            self.variational_params_initialized.fill_(1)

        # DO NOT Ensure inducing_points and x are the same size
        inducing_points = self.inducing_points

        # Get p(u)/q(u)
        variational_dist_u = self.variational_distribution

        # Get q(f)
        if isinstance(variational_dist_u, MultivariateNormal):
            return gpytorch.module.Module.__call__(self,
                x,
                inducing_points,
                inducing_values=variational_dist_u.mean,
                variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
            )
        elif isinstance(variational_dist_u, gpytorch.distributions.Delta):
            return gpytorch.module.Module.__call__(self,
                x, inducing_points, inducing_values=variational_dist_u.mean, variational_inducing_covar=None
            )
        else:
            raise RuntimeError(
                f"Invalid variational distribuition ({type(variational_dist_u)}). "
                "Expected a multivariate normal or a delta distribution."
            )


class ConvSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, image_size, num_classes, patch_size, inducing_patches, learn_inducing_locations):
        batch_shape = torch.Size([num_classes])
        v_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_patches.size(-2), batch_shape=batch_shape)
        v_strat = ConvDomainVStrat(self, image_size, patch_size,
                                   inducing_patches, v_dist,
                                   learn_inducing_locations=learn_inducing_locations)
        super().__init__(v_strat)

        kernel_batch_shape = torch.Size([])

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module_loc = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=3/2, batch_shape=kernel_batch_shape, ard_num_dims=2, lengthscale=3.),
            batch_shape=kernel_batch_shape)
        self.covar_module_f = gpytorch.kernels.RBFKernel(
            ard_num_dims=patch_size.numel(), batch_shape=kernel_batch_shape)

    def covar_module(self, x1, x2=None):
        x_loc, x_f = x1
        if x2 is None:
            z_loc = z_f = None
        else:
            z_loc, z_f = x2
        return self.covar_module_loc(x_loc, z_loc) * self.covar_module_f(x_f, z_f)

    def forward(self, x):
        x_loc, _ = x
        return MultivariateNormal(self.mean_module(x_loc), self.covar_module(x))


@experiment.automain
def main(patch_size, num_inducing, use_cuda, lr_init, epochs,
         learn_inducing_locations, model_type, load_model_from, batch_size,
         lr_time_decay):
    train_set, test_set = dataset()
    num_classes = len(train_set.classes)

    loader = DataLoader(train_set, batch_size=1)
    train_x, _ = next(iter(loader))

    patch_size = torch.Size(patch_size)
    image_size = train_x.size()

    if model_type == "tick":
        inducing_patches = init_inducing_patches(train_set, num_inducing, patch_size)
        model = ConvSVGP(image_size=image_size,
                         num_classes=num_classes,
                         patch_size=patch_size,
                         inducing_patches=inducing_patches,
                         learn_inducing_locations=learn_inducing_locations)
    elif model_type == "rbf":
        idx = torch.randint(low=0, high=len(train_set), size=(num_inducing,))
        inducing_points = torch.stack([train_set[i][0].view(-1) for i in idx], dim=0)
        model = nigp.models.SVGP(train_x.view(-1, image_size.numel()), torch.Size([num_classes]), inducing_points)

    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
        num_features=num_classes, num_classes=num_classes,
        mixing_weights=False)

    if load_model_from is not None:
        model_sd, lik_sd = pu.load(load_model_from)
        model.load_state_dict(model_sd)
        likelihood.load_state_dict(lik_sd)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, len(train_set))

    if use_cuda:
        model = model.cuda()
        likelihood = likelihood.cuda()

    params = [*model.parameters(), *likelihood.parameters()]
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(params, lr=lr_init)

    for n, p in [*model.named_parameters(), *likelihood.named_parameters()]:
        print(n, p.shape, p.requires_grad)

    print_timings = tbx.PrintTimings("Iter", 5.)
    print_timings.data.append(["loss", "nan"])

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=use_cuda, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=use_cuda, drop_last=False)

    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        next_range = range(len(loader)*(epoch-1)+1, len(loader)*epoch + 1)

        model.train()
        likelihood.train()
        for step, (train_x, train_y) in zip(next_range, print_timings(loader)):
            for g in optimizer.param_groups:
                g['lr'] = lr_init * (step**lr_time_decay)  # Amend learning rate as 1/t

            if use_cuda:
                train_x = train_x.cuda()
                train_y = train_y.cuda()

            if model_type == "rbf":
                train_x = train_x.view(-1, image_size.numel())

            optimizer.zero_grad()

            loss = -mll(MultitaskMultivariateNormal.from_batch_mvn(model(train_x)), train_y).sum()
            loss.backward()
            print_timings.data[0][1] = loss.item()

            if step % 94 == 0:
                tbx.add_scalar("train/elbo", -loss.item(), step)
            optimizer.step()

        artifacts.pkl_dump((model.state_dict(), likelihood.state_dict()),
                           f"model_and_lik_{epoch:03d}.pkl.gz")

        model.eval()
        likelihood.eval()

        lpp = 0.
        acc_top1 = 0
        acc_top2 = 0
        acc_top3 = 0
        for test_x, test_y in test_loader:
            if use_cuda:
                test_x = test_x.cuda()
                test_y = test_y.cuda()
            if model_type == "rbf":
                test_x = test_x.view(-1, image_size.numel())

            with torch.no_grad():
                test_f = MultitaskMultivariateNormal.from_batch_mvn(model(test_x))
                lpp += likelihood.log_marginal(test_y, test_f).sum(0).item()

                topk = test_f.mean.topk(k=3, dim=1, largest=True, sorted=True).indices
                ntop1, ntop2, ntop3 = (topk == test_y.unsqueeze(1)).sum(0)
                acc_top1 += ntop1.item()
                acc_top2 += ntop2.item()
                acc_top3 += ntop3.item()

        acc_top3 = (acc_top1 + acc_top2 + acc_top3)/len(test_set)
        acc_top2 = (acc_top1 + acc_top2)/len(test_set)
        acc_top1 = acc_top1/len(test_set)

        tbx.add_scalar("test/lpp", lpp, step)
        tbx.add_scalar("test/acc_top1", acc_top1, step)
        tbx.add_scalar("test/acc_top2", acc_top2, step)
        tbx.add_scalar("test/acc_top3", acc_top3, step)



