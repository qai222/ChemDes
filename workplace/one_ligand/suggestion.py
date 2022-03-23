import itertools
import logging
import random

import numpy as np
import pandas as pd

from chemdes.schema import Molecule
from chemdes.tasks.preprocess import load_ligand_to_des_record, preprocess_descriptor_df, load_descriptors_and_fom
from chemdes.twinsk.estimator import TwinRegressor, upper_confidence_interval


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def truncate_distribution(x: list[float] or np.ndarray, position="top", fraction=0.1):
    """ keep top or bottom x% of the population """
    if isinstance(x, list):
        x = np.array(x)
    assert x.ndim == 1
    if position == "top":
        return np.array(sorted(x, reverse=True)[:int(len(x) * fraction)])
    else:
        return np.array(sorted(x)[:int(len(x) * fraction)])


def ligands_to_df_X(ligands: [Molecule], ligand_to_des_record: dict, cmin: float, cmax: float, nfake=1000):
    final_cols = set()
    records = []
    for ligand in ligands:
        fake_amounts = np.linspace(cmin, cmax, nfake)
        des_record = ligand_to_des_record[ligand]
        for fa in fake_amounts:
            record = {"ligand_inchi": ligand.inchi, "ligand_iupac_name": ligand.iupac_name, "ligand_amount": fa}
            record.update(des_record)
            if len(final_cols) == 0:
                final_cols.update(set(record.keys()))
            records.append(record)
    df = pd.DataFrame.from_records(records, columns=sorted(final_cols))
    return df


def get_predictions(reg: TwinRegressor, ligands: [Molecule], ligand_to_des_record: dict, cmin: float, cmax: float,
                    nfake=1000, y_real=None, df_X_real=None) -> pd.DataFrame:
    if df_X_real is not None:
        df_X = df_X_real
    else:
        df_X = ligands_to_df_X(ligands, ligand_to_des_record, cmin, cmax, nfake)
    ligand_iupac_name = df_X["ligand_iupac_name"]
    df_X = preprocess_descriptor_df(df_X)
    y_distribution = reg.twin_predict_distribution(df_X.values)
    mu = y_distribution.mean(axis=1)
    std = y_distribution.std(axis=1)
    uci = np.apply_along_axis(upper_confidence_interval, 1, y_distribution)
    if y_real is None:
        y_real = np.empty(df_X.shape[0])
        y_real.fill(np.nan)
        y_real = y_real
    assert len(mu) == len(std) == len(uci) == len(y_real) == df_X.shape[0]

    values = np.vstack([df_X["ligand_amount"].values, mu, std, uci, y_real])
    values = values.T
    df_outcome = pd.DataFrame(data=values, columns=["amount", "mean", "std", "uci", "y_real"])
    df_outcome["ligand"] = ligand_iupac_name
    return df_outcome


def get_data_for_comparison(outcome_df: pd.DataFrame):
    data = dict()
    ligands = sorted(set(outcome_df["ligand"]))
    labelled_ligands = []
    unlabelled_ligands = []
    for ligand in ligands:
        logging.warning("Ligand: {}".format(ligand))
        df_ligand = outcome_df.loc[outcome_df['ligand'] == ligand]
        if df_ligand["y_real"].isnull().all():
            unlabelled_ligands.append(ligand)
        elif not df_ligand["y_real"].isnull().any():
            labelled_ligands.append(ligand)
        else:
            logging.critical("This ligand is partially labelled: {}".format(ligand))
        data[ligand] = {
            k: truncate_distribution(df_ligand[k].values) for k in ["mean", "std", "uci"]
        }
        logging.warning("# of predictions after truncation: {}".format(len(data[ligand]["mean"])))
    return data, labelled_ligands, unlabelled_ligands


def compare_two_ligand(data, l1, l2, key="mean", seed=42, n=1001):
    """ data[ligand][key] is a list of predictions using fake amounts """
    pred1 = data[l1][key]
    pred2 = data[l2][key]
    assert len(pred1) == len(pred2)
    joint_populations = list(itertools.product(range(len(pred1)), range(len(pred2))))
    random.seed(seed)
    joint_samples = random.choices(joint_populations, k=n)
    l1_better = 0
    l2_better = 0
    for i, j in joint_samples:
        if pred1[i] >= pred2[j]:
            l1_better += 1
        else:
            l2_better += 1
    return l1_better > l2_better


def suggest_k(k, data, ligands, key="mean", higher_is_better=True):
    data = {l: data[l] for l in ligands}
    suggestions = []
    while len(suggestions) < k:
        for suggestion in suggestions:
            data.pop(suggestion, None)
        suggestion = None
        for l in data:
            if suggestion is None:
                suggestion = l
            else:
                if higher_is_better:
                    if compare_two_ligand(data, l, suggestion, key, seed=42, n=1001):
                        suggestion = l
                else:
                    if compare_two_ligand(data, suggestion, l, key, seed=42, n=1001):
                        suggestion = l
        assert not suggestion is None
        suggestions.append(suggestion)
    return suggestions


if __name__ == '__main__':
    # load descriptors
    LigandToDesRecord = load_ligand_to_des_record("../ligand_descriptors/molecular_descriptors_2022_03_21.csv")
    # load predictor
    from joblib import load

    reg = load("output/tuned.joblib")

    # load labelled data
    labelled_ligands, df_X_labelled, df_y_labelled = load_descriptors_and_fom(
        mdes_csv="../ligand_descriptors/molecular_descriptors_2022_03_21.csv",
        reactions_json="output/2022_0304_LS001_MK003_reaction_data.json",
    )

    # range of amounts
    amounts = df_X_labelled["ligand_amount"].tolist()
    cmin, cmax = min(amounts), max(amounts)

    # predict
    unlabelled_ligands = [l for l in LigandToDesRecord if l not in labelled_ligands]
    df_outcome_unlabelled = get_predictions(reg, unlabelled_ligands, LigandToDesRecord, cmin, cmax, 1000, y_real=None)
    df_outcome_labelled = get_predictions(reg, labelled_ligands, LigandToDesRecord, cmin, cmax, 100,
                                          y_real=df_y_labelled.values, df_X_real=df_X_labelled)
    df_outcome = pd.concat([df_outcome_labelled, df_outcome_unlabelled], axis=0)
    df_outcome.to_csv("output/suggestions_pred.csv", index=False)

    # prepare data for MC comparison
    data, labelled_ligands_names, unlabelled_ligands_names = get_data_for_comparison(df_outcome)

    # suggestions from unlabelled ligands
    suggestions = pd.DataFrame()
    for key in ["mean", "std", "uci"]:
        suggestions["best_5_{}".format(key)] = suggest_k(k=5, data=data, ligands=unlabelled_ligands_names, key=key,
                                                         higher_is_better=True)
        suggestions["worst_5_{}".format(key)] = suggest_k(k=5, data=data, ligands=unlabelled_ligands_names, key=key,
                                                          higher_is_better=False)
    suggestions.to_csv("output/suggestions.csv", index=False)
