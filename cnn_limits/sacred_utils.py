import gpytorch
import torch
import torchvision
import os

__all__ = ["gpytorch_pre_run_hook", "load_dataset"]

def gpytorch_pre_run_hook(num_likelihood_samples, default_dtype):
    gpytorch.settings.num_likelihood_samples._set_value(num_likelihood_samples)
    gpytorch.settings.max_cholesky_size._set_value(100000)  # disable CG, it makes eigenvalues negative :(
    torch.set_default_dtype(getattr(torch, default_dtype))

def load_dataset(dataset_name, dataset_base_path, additional_transforms=[]):
    if dataset_name == "CIFAR10":
        dataset_base_path = os.path.join(dataset_base_path, dataset_name)
    if len(additional_transforms) == 0:
        trans = torchvision.transforms.ToTensor()
    else:
        trans = torchvision.transforms.Compose([
            *additional_transforms,
            torchvision.transforms.ToTensor(),
        ])

    _dset = getattr(torchvision.datasets, dataset_name)
    train = _dset(dataset_base_path, train=True, download=True, transform=trans)
    test = _dset(dataset_base_path, train=False, transform=trans)
    return train, test
