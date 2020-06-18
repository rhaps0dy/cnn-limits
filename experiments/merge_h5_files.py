import sys

import h5py
import numpy as np
from tqdm import tqdm

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} dest_file [source_file1 source_file2 ...]")
    sys.exit(1)

_, dest_file, *src_files = sys.argv
print(f"copying {dest_file} <- {src_files}")

with h5py.File(dest_file, "a") as dest_f:
    for _file_i, path in enumerate(src_files):
        print(f"Opening file {path} ({_file_i}/{len(src_files)})")
        with h5py.File(path, "r") as src_f:
            valid_keys = [k
                          for k in dest_f.keys()
                          if k in src_f.keys()]
            for k in tqdm(src_f.keys()):
                # if k == "Kxt":
                #     print("skiping kxt")
                #     continue
                print(f"Doing data set {k}")
                try:
                    dest_data = dest_f[k]
                except KeyError:
                    dest_data = dest_f.create_dataset(
                        k, shape=src_f[k].shape, dtype=src_f[k].dtype,
                        fillvalue=np.nan, chunks=(1, *src_f[k].shape[1:]),
                        maxshape=(None, *src_f[k].shape[1:]))

                src_data = src_f[k]
                assert dest_data.shape == src_data.shape, f"Not the same for dest={dest_file}, src={path}, k={k}"
                for i in tqdm(range(len(dest_data))):
                    src = src_data[i, ...]
                    dest = dest_data[i, ...]
                    dest_is_nan = np.isnan(dest)
                    dest[dest_is_nan] = src[dest_is_nan]
                    dest_data[i, ...] = dest
