from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from lsal.campaign.loader import ReactionCollection, LigandExchangeReaction, get_od, get_sumod, \
    get_target_data
from lsal.utils import json_load, get_basename


def mpl_plot_per_ligand_results(
        reaction_collection: ReactionCollection,
        get_function: Callable[[LigandExchangeReaction, ...], float] = get_sumod,
        ylim=(2e5, 5.5e5),
        ylabel="sum/OD",
        logx=True,
):
    ligand_to_results = get_target_data(reaction_collection, get_function)

    ligands = sorted(ligand_to_results.keys())
    ncols = 5
    nrows = len(ligands) // ncols + 1
    fig, total_axes = plt.subplots(nrows=nrows, ncols=ncols,
                                   figsize=(4 * nrows, 4 * ncols,))
    for i in range(nrows):
        for j in range(ncols):
            total_axes[i][j].set_axis_off()

    for iax, ligand in enumerate(ligands):
        ax = total_axes[iax // ncols][iax % ncols]
        ax.set_axis_on()
        data = ligand_to_results[ligand]
        xs = data["amount"]
        ys = data["values"]
        y_ref = np.mean(data["ref_values"])
        y_ref_err = np.std(data["ref_values"])
        # y_ref_err = max(data["ref_OD"]) - min(data["ref_OD"])
        y_ref = np.array([y_ref, ] * len(xs))
        ax.scatter(xs, ys, marker="x", c="k", label="Experimental")
        ax.fill_between(sorted(xs), y_ref - 3 * y_ref_err, y_ref + 3 * y_ref_err, alpha=0.2, label=r"ref $3\delta$")
        ax.set_title(ligand.label)
        if logx:
            ax.set_xscale("log")
        ax.set_ylim(ylim)
        ax.set_xlabel("amount (uL*uM)")
        ax.set_ylabel(ylabel)
        if iax == 0:
            ax.legend()
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    reactions = json_load("../data/collect_reactions_SL_0519.json")

    fig_sumod = mpl_plot_per_ligand_results(reactions, )
    fig_sumod.savefig(get_basename(__file__) + "_sumod.png")

    fig_od = mpl_plot_per_ligand_results(reactions, get_od, (-0.2, 2.3), "OD", True)
    fig_od.savefig(get_basename(__file__) + "_od.png")
