import logging

import matplotlib.pyplot as plt
import numpy as np
import umap

from lsal.utils import SEED


def scatter2d(data_2d: np.ndarray) -> plt.Figure:
    plot_kwds = {'alpha': 0.5, 's': 80, 'linewidth': 0}
    fig, ax = plt.subplots()
    ax.scatter(data_2d.T[0], data_2d.T[1], color='gray', **plot_kwds)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def umap_run(dmat, nn, md, wdir="./"):
    saveas = "nn{}-md{}".format(nn, md)
    logging.warning("working on: {}".format(saveas))
    transformer = umap.UMAP(
        n_neighbors=nn, min_dist=md, metric="precomputed", random_state=SEED)
    data_2d = transformer.fit_transform(dmat)
    fig = scatter2d(data_2d)
    fig.savefig("{}/{}.png".format(wdir, saveas))
    return data_2d


def tune_umap(
        dmat,
        n_neighbors_values=(3, 5, 7,),
        min_dist_values=(0.1, 0.2, 0.3,),
        wdir="./",
):
    for nn in n_neighbors_values:
        for md in min_dist_values:
            umap_run(dmat, nn, md, wdir)
