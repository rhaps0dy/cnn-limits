import contextlib
import os
import pickle
from pathlib import Path

import sacred
import torch
import torchvision
from torch.utils.data import Subset

import gpytorch

__all__ = ["gpytorch_pre_run_hook", "load_dataset", "interlaced_argsort", "base_dir", "def_new_file", "def_load_sorted_dataset"]

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


def add_file_observer(experiment, name):
    if name == "__main__":
        log_dir = (os.environ['LOG_DIR'] if 'LOG_DIR' in os.environ
                   else f"/scratch/ag919/logs/{experiment.path}")
        print("Resulting log dir: ", log_dir)
        experiment.observers.append(
            sacred.observers.FileStorageObserver(log_dir))

def base_dir(_run, _log):
    try:
        return Path(_run.observers[0].dir)
    except IndexError:
        _log.warning("This run has no associated directory, using `/tmp`")
        return Path("/tmp")


def def_new_file(base_dir):
    @contextlib.contextmanager
    def new_file(relative_path, mode="wb"):
        full_path = os.path.join(base_dir(), relative_path)
        with open(full_path, mode) as f:
            yield f
    return new_file

def def_load_sorted_dataset(experiment, _load_dataset):
    @experiment.capture
    def load_sorted_dataset(sorted_dataset_path, N_train, N_test):
        with experiment.open_resource(os.path.join(sorted_dataset_path, "train.pkl"), "rb") as f:
            train_idx = pickle.load(f)
        with experiment.open_resource(os.path.join(sorted_dataset_path, "test.pkl"), "rb") as f:
            test_idx = pickle.load(f)
        train_set, test_set = _load_dataset()
        return (Subset(train_set, train_idx[:N_train]),
                Subset(test_set, test_idx[:N_test]))
    return load_sorted_dataset

# class Utils:
#     @classmethod
#     def define_functions(self, experiment):
#         self.load_dataset = experiment.capture(load_dataset)
#         self.base_dir = experiment.capture(base_dir)
#         self.new_file = def_new_file(experiment)
#         self.load_sorted_dataset = def_load_sorted_dataset(experiment)
