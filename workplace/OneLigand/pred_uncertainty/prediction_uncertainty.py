import glob
import os.path

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from lsal.alearn import SingleLigandPrediction
from lsal.utils import pkl_load

_npy_files = """learning_SL0519.npy
learning_AL1026.npy
learning_AL1213.npy
learning_AL1222.npy
learning_AL0130.npy
learning_AL0303.npy
learning_AL0331.npy
learning_AL0503.npy"""


def collect_uncertainties_from_learning_data(pred_dir):
    """
    save pred std array to a npy file, this is a 3d array with
    a[<ligand>][<concentration>][<pred_std>]
    """
    iter_name = os.path.basename(pred_dir)
    assert iter_name.startswith("learning_")
    npy_array_file = f"{iter_name}.npy"
    if not os.path.isfile(npy_array_file):
        pred_chunks = glob.glob(f"../../../workplace_data/OneLigand/{iter_name}/prediction/prediction_chunk_*.pkl")
        pred_chunks = sorted(pred_chunks)
        stds = []
        for pred in tqdm.tqdm(pred_chunks):
            slps = pkl_load(pred, print_timing=False)
            slps: list[SingleLigandPrediction]
            for slp in slps:
                stds.append(slp.pred_std)
        stds = np.array(stds)
        np.save(npy_array_file, stds)


def collect_uncertainties():
    for pred_dir in sorted(glob.glob("../../../workplace_data/OneLigand/learning_*")):
        collect_uncertainties_from_learning_data(pred_dir)


def plot_pred_uncertainty(aggregate=False):
    data = {}
    for npy_array_file in _npy_files.split():
        stds = np.load(npy_array_file)
        data[npy_array_file.replace(".npy", "")] = stds

    if aggregate:
        n_cols = 3
        n_figs = len(data)
        n_rows = n_figs // n_cols + 1
        iter_names = list(data.keys())
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows))
        for i in range(n_rows * n_cols):
            irow = i // n_cols
            icol = i % n_cols
            ax = axs[irow][icol]
            if i >= n_figs:
                ax.visible = False
                ax.axis('off')
                continue
            iter_name = iter_names[i]
            stds = data[iter_name]
            xs = list(range(stds.shape[1]))
            ys = stds.mean(axis=0)
            yerr = stds.std(axis=0)

            ax.errorbar(xs, ys, yerr=yerr, fmt='o', color='k')
            ax.set_title(iter_name)
            ax.set_ylabel("Uncertainty")
            ax.set_xlabel("Concentration")
            ax.set_ylim([0, 0.4])

        fig.savefig("pred_uncertainty_agg.png", dpi=600)
    else:
        xs = []
        ys = []
        for iter_name, stds in data.items():
            xs.append(iter_name.replace("learning_", ""))
            ys.append(stds.mean())
        fig, ax = plt.subplots()
        ax.set_ylabel("Averaged Prediction Uncertainty")
        ax.set_xlabel("Experiment Campaign")
        ax.plot(xs, ys)
        fig.savefig("pred_uncertainty_averaged.png", dpi=600)


if __name__ == '__main__':
    plot_pred_uncertainty(aggregate=False)
    plot_pred_uncertainty(aggregate=True)
