import glob
from typing import Union

import pandas as pd
from pqdm.processes import pqdm

from lsal.campaign.single_learner import SingleLigandPredictions, _known_metric
from lsal.tasks.complexity import calculate_complexities
from lsal.utils import pkl_load, FilePath, get_basename, smi2imagestr, pkl_dump


def get_vis_record(slp: SingleLigandPredictions) -> dict[str, Union[float, str]]:
    ranks = dict()
    smi = slp.ligand.smiles
    ranks["smiles"] = smi
    for m in _known_metric:
        ranks[m] = slp.rank_metric(m)
    return ranks


def get_vis_record_from_pkl(prediction_pkl: FilePath) -> dict[str, Union[float, str]]:
    slp = pkl_load(prediction_pkl, print_timing=False)
    return get_vis_record(slp)


def get_ligand_descriptions(smi: str):
    r = calculate_complexities(smi)
    r["img"] = smi2imagestr(smi)
    return r


if __name__ == '__main__':
    available_foms = ["fom1", "fom2", "fom3"]
    pool_prediction_folders = ["../predictions_{}".format(fom) for fom in available_foms]
    learned_prediction_results = pkl_load("../predict_for_learned.pkl")

    smis = []
    for folder in pool_prediction_folders:
        prediction_pkls = sorted(glob.glob(f"{folder}/*.pkl"))
        # parallel
        records = pqdm(prediction_pkls, get_vis_record_from_pkl, n_jobs=8)
        df = pd.DataFrame.from_records(records)
        df.to_csv(f"{get_basename(folder)}.csv", index=False)
        smis += df["smiles"].tolist()

    # also output vis records for learned ligands
    learned_predictions = pkl_load("../predict_for_learned.pkl")
    for fom in available_foms:
        slps = learned_predictions[fom]
        slps: list[SingleLigandPredictions]
        records = [get_vis_record(slp) for slp in slps]
        df = pd.DataFrame.from_records(records)
        df.to_csv(f"predictions_{fom}_learned.csv", index=False)
        smis += df["smiles"].tolist()

    # calculate complexities and their images
    unique_smis = sorted(set(smis))
    records = pqdm(smis, get_ligand_descriptions, n_jobs=8)
    data = {r["smiles"]: r for r in records}
    pkl_dump(data, f"{get_basename(__file__)}_ligands.pkl")
