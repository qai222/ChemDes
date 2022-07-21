import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from lsal.utils import pkl_load, get_basename

sns.set_style("whitegrid")

AvailableHistoryData = sorted(glob.glob("visdata_obo/*.pkl"))
AvailableHistoryData = {get_basename(f): pkl_load(f) for f in AvailableHistoryData}

SWF_names = sorted(AvailableHistoryData.keys())


def get_xys(visdata_dict, avg_over_unlearned=True):
    xs = []
    ys_mae = []
    ys_uncertainty = []
    ys_uncertainty_top2 = []
    for key, visdata in visdata_dict.items():
        ligands = sorted(visdata.keys())
        unlearned_ligands = [lig for lig in ligands if not visdata[lig]["is_learned"]]
        if len(unlearned_ligands) == 0:
            break
        xs.append(len(ligands) - len(unlearned_ligands))
        maes_wrt_real_values = []
        overall_unceratinty_values = []
        overall_unceratinty_top2_values = []
        if avg_over_unlearned:
            avg_ligands = unlearned_ligands
        else:
            avg_ligands = ligands
        for lig in avg_ligands:
            maes_wrt_real_values.append(visdata[lig]["mae_wrt_real"])
            overall_unceratinty_top2_values.append(visdata[lig]["uncertainty_top2"])
            overall_unceratinty_values.append(visdata[lig]["uncertainty"])
        ys_mae.append(np.mean(maes_wrt_real_values))
        ys_uncertainty.append(np.mean(overall_unceratinty_values))
        ys_uncertainty_top2.append(np.mean(overall_unceratinty_top2_values))
    return xs, ys_mae, ys_uncertainty, ys_uncertainty_top2


if __name__ == '__main__':

    fig = plt.figure(constrained_layout=True, figsize=[12, 6 * len(SWF_names)])
    fig.suptitle('Twin-RF Learning curve')

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=len(SWF_names), ncols=1, )
    for irow, subfig in enumerate(subfigs):
        swf_name = SWF_names[irow]
        fom_type, metric, _, _ = swf_name.split("--")
        visdata_dictionary = AvailableHistoryData[swf_name]
        same_row_ys = []
        subfig.suptitle(f'FOM: {fom_type} Suggestion: {metric}', fontsize='x-large')
        # create 1x2 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=2)
        for average_over_unlearned, ylabel, ax in zip((False, True),
                                                      ("Average over all ligands", "Average over unlearned ligands"),
                                                      axs):
            xs, mae, uncert, uncert_top2 = get_xys(visdata_dictionary, avg_over_unlearned=average_over_unlearned)
            same_row_ys += mae
            same_row_ys += uncert
            same_row_ys += uncert_top2
            ax.plot(xs, mae, "ko-", label="MAE")
            ax.plot(xs, uncert, "bx-.", label="Uncertainty")
            ax.plot(xs, uncert_top2, "rv:", label="Uncertainty-top2%")
            ax.set_xlabel("Number of learned ligands")
            ax.set_ylabel("{}: {}".format(fom_type, ylabel))
            ax.set_xticks(xs)
            ax.legend()
        ylim = (min(same_row_ys), max(same_row_ys))
        delta_ylim = ylim[1] - ylim[0]
        ylim = (ylim[0] - 0.05 * delta_ylim, ylim[1] + 0.05 * delta_ylim)
        for ax in axs:
            ax.set_ylim(ylim)
    # plt.tight_layout()
    plt.savefig("learning_curves.png")
