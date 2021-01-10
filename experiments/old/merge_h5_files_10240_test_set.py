import sys

import h5py
import numpy as np
from tqdm import tqdm

dest_file = "w00_kernels.h5"

step = 10240 // 32

with h5py.File(dest_file, "a") as dest_f:
    assert not np.any(np.isnan(dest_f["Kxt"][:, :step, :]))
    for i in range(0, 32):
        if not (i==19 or i==29):
            print(f"Checking {i}")
            assert not np.any(np.isnan(dest_f["Kxt"][:, i*step:(i+1)*step, :]))
            continue

        path = f"w{i:02d}_kernels.h5"
        print(f"Loading file {path}")
        with h5py.File(path, "r") as src_f:
            data = src_f["Kxt"][:, i*step:(i+1)*step, :]
            if np.any(np.isnan(data)):
                print(f"File {path} has NaNs. Skipping.")
            else:
                dest_f["Kxt"][:, i*step:(i+1)*step, :] = data


