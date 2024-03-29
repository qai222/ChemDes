import glob
import os.path

import numpy as np
import pandas as pd
import tqdm
from loguru import logger

from lsal.alearn.one_ligand import SingleLigandPrediction
from lsal.utils import json_load, json_dump
from lsal.utils import pkl_load, pkl_dump, truncate_distribution


def select_cfs(res1: str, res2: str) -> list[dict]:
    selected = []
    unique_pairs = []
    exported_cfs = json_load("../export_cfs.json.gz")
    for cf in exported_cfs:
        if cf['residual1'] == res1 and cf['residual2'] == res2:
            selected.append(cf)
            unique_pairs.append((cf['smiles1'], cf['smiles2']))
    assert len(set(unique_pairs)) == len(selected)
    logger.info(f"selected cfs: {len(selected)}")
    return selected


def dump_pred_from_selected_cfs(res1: str, res2: str):
    dump_file = f"selected_predictions_{res1}-{res2}.pkl"
    selected_cfs = select_cfs(res1, res2)
    if os.path.isfile(dump_file):
        return pkl_load(dump_file), selected_cfs
    unique_smiles = []
    for cf in selected_cfs:
        unique_smiles.append(cf['smiles1'])
        unique_smiles.append(cf['smiles2'])
    unique_smiles = sorted(set(unique_smiles))
    pred_ranges = []
    selected_preds = []
    for pred_file in tqdm.tqdm(
            sorted(glob.glob("E:\workplace_data\OneLigand\learning_SL0519\prediction\prediction_*.pkl"))):
        preds = pkl_load(pred_file, print_timing=False)
        for pred in preds:
            pred: SingleLigandPrediction
            if pred.ligand.smiles in unique_smiles:
                selected_preds.append(pred)
    pkl_dump(selected_preds, dump_file)
    return selected_preds, selected_cfs


def get_max_conc(pred: SingleLigandPrediction):
    idx = truncate_distribution(pred.pred_mu, "top", 0.02, True)
    return np.mean(pred.amounts[idx])


def main_analysis(preds: list[SingleLigandPrediction], cfs: list[dict]):
    pred_dict = {p.ligand.smiles: p for p in preds}
    pred_dict: dict[str, SingleLigandPrediction]
    delta_fom = []
    delta_conc = []
    for cf in cfs:
        smi1 = cf['smiles1']
        smi2 = cf['smiles2']
        pred1 = pred_dict[smi1]
        pred2 = pred_dict[smi2]
        delta_max_conc = get_max_conc(pred2) - get_max_conc(pred1)
        delta_fom.append(cf['delta'])
        delta_conc.append(delta_max_conc)
    return delta_conc, delta_fom


def main_plot(delta_conc: list[float], delta_fom: list[float]):
    data = pd.DataFrame({"dconc": delta_conc, "dfom": delta_fom})
    import seaborn as sns
    fig = sns.jointplot(
        data, x="dconc", y="dfom",
        marker="+", s=100, marginal_kws=dict(bins=25, fill=False),
    )
    fig.ax_joint.set_xlabel("$\Delta \log c_{max}$")
    fig.ax_joint.set_ylabel("$\Delta FOM_{max}$")
    return fig


def main(res1: str, res2: str):
    predictions, selected_cfs = dump_pred_from_selected_cfs(res1, res2)
    logger.info(f"selected predictions: {len(predictions)}")
    delta_conc, delta_fom = main_analysis(predictions, selected_cfs)
    json_dump({"delta_conc": delta_conc, "delta_fom": delta_fom}, f"dconc_{res1}-{res2}.json")
    fig = main_plot(delta_conc, delta_fom)
    fig.savefig(f"dconc_{res1}-{res2}.png")


if __name__ == '__main__':
    main("N", "Br")
    main("O", "Br")
    main("O", "CN")
    main("C", "O")
    main("CO", "F")
    main("CO", "N")
