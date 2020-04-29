import contextlib
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paramiko

LOGS = "/scratch/ag919/logs"


class ExperimentResults:
    _host_and_path = re.compile(r"^([a-zA-Z0-9.]+):(.*)$")

    def __init__(self, path, notes):
        self.host, fpath = self._host_and_path.match(path).groups()
        self.path = Path(fpath)
        self.notes = notes

    @contextlib.contextmanager
    def open(self, fname, mode="rb"):
        if self.host == os.uname().nodename:
            with open(self.path/fname, mode) as f:
                yield f
        else:
            with contextlib.closing(paramiko.SSHClient()) as ssh:
                ssh.load_system_host_keys()
                ssh.connect(self.host)
                with contextlib.closing(ssh.open_sftp()) as sftp, \
                     contextlib.closing(sftp.open(str(self.path/fname))) as f:
                    yield f

    def read_pickle(self, fname):
        with self.open(fname, "rb") as f:
            compression = ("gzip" if fname[-2:] == "gz" else None)
            return pd.read_pickle(f, compression=compression)

    def __str__(self):
        return f"ExperimentResults(\"{self.host}:{self.path}\", \"\"\"{self.notes}\"\"\")"


el = dict(
    # Closed form kernels, no likelihood optimization
    exp_nngp_google_v1=ExperimentResults(f"huygens:/scratch/ag919/logs/predict/90", """
Classify CIFAR-10 using a 32-layer rectangular ReLU network, no residual layers.
This is the architecture used in the 2019 Google NNGP paper.
The kernel is calculated in closed form.

The kernel has been inverted using the minimum possible noise that doesn't give NaNs.
This version has a variance that vanishes with depth, so the results are unreliable after layer 22.
"""),
    exp_nngp_google_v2=ExperimentResults("huygens:/scratch/ag919/logs/predict/92", """
Classify CIFAR-10 using a 32-layer rectangular ReLU network, no residual layers.
This is the architecture used in the 2019 Google NNGP paper.
The kernel is calculated in closed form.

The kernel has been inverted using the minimum possible noise that doesn't give NaNs.
This version has constant variance with depth (has a √2 correction factor for ReLU std).
"""),

    # Closed form kernel, likelihood optimization
    exp_nngp_google_v3=ExperimentResults("huygens:/scratch/ag919/logs/predict/96", """
Optimized the likelihood before doing predictions
"""),

    # Random feature kernels, no likelihood optimization
    exp_mc_nn_double_descent=ExperimentResults("ulam:/scratch/ag919/logs/predict/5", """
Classify CIFAR-10 using a 32-layer rectangular ReLU network, no residual layers. Arch from 2019 Google NNGP paper.
Kernel calculated by drawing a bunch of random NNs, each with 16 channels.
"""),

    # Random feature kernels with 10k examples, likelihood optimization
    exp_optl_mcnn_old=ExperimentResults("ulam:/scratch/ag919/logs/predict/46", """
Classify CIFAR-10 using a 32-layer rectangular ReLU network, no residual layers. Arch from 2019 Google NNGP paper.
Kernel calculated by drawing 138000 random NNs, each with 16 channels.

The maximum likelihood value of jitter is found, for N=50000. Smaller N use the same jitter. 
"""),

    # Plot likelihood and accuracy vs. sigma_y^2 f
    exp_sigy_grid=ExperimentResults("ulam:/scratch/ag919/logs/predict_lik_vs_acc/9", """
    Plot likelihood and accuracy vs. sigma_y^2. Calculated using GPU.
    Arch from 2019 Google NNGP paper. Kernel calculated by drawing 138000 random NNs, each with 16 channels.
"""),

    exp_sigy_grid_k=ExperimentResults("huygens:/scratch/ag919/logs/predict_lik_vs_acc/14", """
    Plot likelihood and accuracy vs. sigma_y^2. Calculated using GPU.
    Arch from 2019 Google NNGP paper. Kernel calculated analytically.
"""),

    exp_sigy_grid_v2=ExperimentResults("ulam:/scratch/ag919/logs/predict_lik_vs_acc/11", "Now with spectrum"),
    exp_sigy_grid_k_v2=ExperimentResults("huygens:/scratch/ag919/logs/predict_lik_vs_acc/16", "Now with spectrum"),
    exp_sigy_grid_v3=ExperimentResults("ulam:/scratch/ag919/logs/predict_lik_vs_acc/15", "Normalising the random feature kernels"),

#     exp_sigy_grid_num_features=ExperimentResults("1", """
# Like the previous experiments, but each "layer" corresponds to the a given number of features for the mean_pool
# """),
    exp_sigy_grid_num_features="lost in previous notebook revisiosn",
    exp_sigy_grid_num_features_small=ExperimentResults("ulam:/scratch/ag919/logs/predict_lik_vs_acc/30", """
Like exp_sigy_grid_num_features. Just a continuation into smaller numbers of features and training data.
"""),

    exp_sigy_grid_num_features_32=ExperimentResults("huygens:/scratch/ag919/logs/predict_lik_vs_acc/24", """
Like exp_sigy_grid_num_features_small, but the NN has width 32.
"""),

    exp_sigy_grid_myrtle=ExperimentResults("huygens:/scratch/ag919/logs/predict_lik_vs_acc/18", """
Like the previous, but doing it for every layer of the Myrtle network,
and various numbers of training examples up to 5000.
"""),
)


def plot_df(ax, df, things, cmap_name='viridis'):
    colors = plt.get_cmap(cmap_name)(np.linspace(0., 1., len(df)))

    for (_, row), color in zip(df.iterrows(), colors):
        ax.plot(row.index, row.values, label=f"{row.name} {things}", color=color)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("Accuracy")
