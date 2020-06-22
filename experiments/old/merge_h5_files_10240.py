import sys

import h5py
import os
import numpy as np
from tqdm import tqdm
from cnn_gp import create_h5py_dataset

if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} dest_file source_file_xt source_file")
    sys.exit(1)

_, dest_file, src_file_xt, src_file = sys.argv

if os.path.exists(dest_file):
    ans = input(f"File `{dest_file}` exists. Are you sure you want to overwrite? (type 'yes')")
    if ans != "yes":
        sys.exit(1)

with h5py.File(dest_file, "w") as dest_f,\
     h5py.File(src_file_xt, "r") as src_f_xt,\
     h5py.File(src_file, "r") as src_f:
    assert np.allclose(src_f_xt["Kt_diag"][...], src_f["Kx_diag"][...]), "train and test correspond well"

    src_shape_0 = src_f["Kxx"].shape[0]
    SIZE = src_f_xt["Kt_diag"].shape[1] + src_f["Kx_diag"].shape[1]

    dest_f.create_dataset("Kt_diag", shape=(src_shape_0, 1), dtype=np.float32, fillvalue=1.)
    dest_f.create_dataset("Kxt", shape=(src_shape_0, SIZE, 1), dtype=np.float32, fillvalue=0.)
    Kx_diag = dest_f.create_dataset("Kx_diag", shape=(src_shape_0, SIZE), dtype=np.float32, fillvalue=np.nan)
    Kx_diag[:, :SIZE//2] = src_f["Kx_diag"]
    Kx_diag[:, SIZE//2:] = src_f_xt["Kx_diag"]

    dest_data = create_h5py_dataset(
        dest_f, batch_size=256, name="Kxx",
        diag=False, N=SIZE, N2=SIZE)
    dest_data.resize(src_shape_0, axis=0)

    for i in tqdm(range(len(dest_data))):
        dest_data[i, :SIZE//2, :SIZE//2] = src_f["Kxx"][i, ...]
        dest_data[i, SIZE//2:, SIZE//2:] = src_f_xt["Kxx"][i, ...]
        dest_data[i, SIZE//2:, :SIZE//2] = src_f_xt["Kxt"][i, ...]
