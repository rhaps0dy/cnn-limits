import sys

import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

from cnn_gp import ProductIterator
import torch
from torch.utils.data import TensorDataset

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} dest_file [source_file1 source_file2 ...]")
    sys.exit(1)

_, dest_path, *src_paths = sys.argv
print(f"copying {dest_path} <- {src_paths}")
dest_path = Path(dest_path)
src_paths = tuple(map(Path, src_paths))


def workers_from(path):
    with open(path/"config.json", "r") as f:
        c = json.load(f)
    return (c['worker_rank'], c['n_workers'], c['batch_size'])

def write_to_dest(dest_data, dest_info, dataset_name, src_path, data1, data2):
    this_wr, this_nw, this_bs = workers_from(src_path)
    _, _dest_nw, _dest_bs = dest_info
    assert this_nw == _dest_nw
    assert this_bs == _dest_bs

    print(f"For dset={dataset_name}, src={src_path}, worker_rank={this_wr}, n_workers={this_nw}")
    with h5py.File(src_path/"kernels.h5", "r") as src_f:
        print("reading dest_data")
        src_data = src_f[dataset_name]
        in_nan = False
        for _, (i, (x1,)), (j, (x2,)) in tqdm(ProductIterator(this_bs, data1, data2, worker_rank=this_wr, n_workers=this_nw)):
            sl1 = slice(i, i+x1.shape[0])
            sl2 = slice(j, j+x2.shape[0])
            chunk = src_data[:, sl1, sl2]
            if np.isnan(chunk[0, 0, 0]):
                if not in_nan:
                    print(f"Skipping chunk ({i}, {j}) because it is NaN")
                    in_nan = True
            else:
                if in_nan:
                    print(f"Writing chunk ({i}, {j}), not NaN!!")
                    in_nan = False
                dest_data[:, sl1, sl2] = chunk


try:
    dest_info = _src_wr, _src_nw, _src_bs = workers_from(dest_path)
    print(f"Dest worker_rank={_src_wr}, n_workers={_src_nw}, batch_size={_src_bs}")
    with h5py.File(dest_path/"kernels.h5", "r") as dest_f:
        depth, len_X, len_X2 = dest_f["Kxt"].shape

except FileNotFoundError:
    with open(src_paths[0]/"config.json", "r") as f:
        c = json.load(f)
    dest_info = (c['worker_rank'], c['n_workers'], c['batch_size'])
    with h5py.File(src_paths[0]/"kernels.h5", "r") as src_f:
        depth, len_X, len_X2 = src_f["Kxt"].shape
    print(f"Creating data sets using {depth}, {len_X}, {len_X2}")

    assert not os.path.exists(dest_path/"kernels.h5")
    with h5py.File(dest_path/"kernels.h5", "w") as dest_f:
        _s = (len_X, len_X2)
        dest_f.create_dataset("Kxt", shape=(depth, *_s), dtype=np.float64,
                              chunks=(1, *_s), maxshape=(None, *_s))
        _s = (len_X, len_X)
        dest_f.create_dataset("Kxx", shape=(depth, *_s), dtype=np.float64,
                              chunks=(1, *_s), maxshape=(None, *_s))

train_set = TensorDataset(torch.arange(len_X))
test_set = TensorDataset(torch.arange(len_X2))

print(f"Opening dest {dest_path}")
with h5py.File(dest_path/"kernels.h5", "r+") as dest_f:
    print("Kxx")
    dest_data = dest_f["Kxx"][...]
    for _file_i, src_path in enumerate(src_paths):
        write_to_dest(dest_data, dest_info, "Kxx", src_path, data1=train_set, data2=None)
with h5py.File(dest_path/"kernels.h5", "r+") as dest_f:
    print("Committing dest_data")
    for i in tqdm(range(depth)):
        dest_f["Kxx"].write_direct(dest_data, source_sel=np.s_[i], dest_sel=np.s_[i])
del dest_data

with h5py.File(dest_path/"kernels.h5", "r+") as dest_f:
    print("Kxt")
    dest_data = dest_f["Kxt"][...]
    for _file_i, src_path in enumerate(src_paths):
        write_to_dest(dest_data, dest_info, "Kxt", src_path, data1=train_set, data2=test_set)
with h5py.File(dest_path/"kernels.h5", "r+") as dest_f:
    print("Committing dest_data")
    for i in tqdm(range(depth)):
        dest_f["Kxt"].write_direct(dest_data, source_sel=np.s_[i], dest_sel=np.s_[i])
