import gpytorch
import torch
import torchvision
import os

__all__ = ["gpytorch_pre_run_hook", "load_dataset", "interlaced_argsort"]

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

def interlaced_argsort(dset):
    y = torch.tensor(dset.targets)
    starting_for_class = [None]* len(dset.classes)
    argsort_y = torch.argsort(y)
    for i, idx in enumerate(argsort_y):
        if starting_for_class[y[idx]] is None:
            starting_for_class[y[idx]] = i

    for i in range(len(starting_for_class)-1):
        assert starting_for_class[i] < starting_for_class[i+1]
    assert starting_for_class[0] == 0

    init_starting = [a for a in starting_for_class] + [len(dset)]

    res = []
    while len(res) < len(dset):
        for i in range(len(starting_for_class)):
            if starting_for_class[i] < init_starting[i+1]:
                res.append(argsort_y[starting_for_class[i]].item())
                starting_for_class[i] += 1
    assert len(set(res)) == len(dset)
    assert len(res) == len(dset)
    return res
