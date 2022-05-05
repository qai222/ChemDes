import itertools
import logging
import random
from typing import Tuple

import numpy as np
import pandas as pd

from lsal.schema import Molecule
from lsal.tasks.io import ligand_to_ml_input
from lsal.twinsk.estimator import TwinRegressor, upper_confidence_interval
from lsal.utils import FilePath, SEED


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


def get_predictions(
        reg: TwinRegressor, ligands: list[Molecule], ligand_inventory: list[Molecule],
        descriptor_csv: FilePath,
        cmin: float, cmax: float,
        nfake=1000, y_real=None, df_X_real=None, df_ligands=None
) -> pd.DataFrame:
    if df_X_real is not None:
        df_X = df_X_real
        assert df_ligands is not None
    else:
        fake_amounts = [np.linspace(cmin, cmax, nfake) for _ in range(len(ligands))]
        ligand_to_fake_amounts = dict(zip(ligands, fake_amounts))
        df_ligands, df_X, _ = ligand_to_ml_input(
            ligand_to_data=ligand_to_fake_amounts, data_type="amount", ligand_inventory=ligand_inventory,
            descriptor_csv=descriptor_csv,
        )
    y_distribution = reg.twin_predict_distribution(df_X.values)
    mu = y_distribution.mean(axis=1)
    std = y_distribution.std(axis=1)
    uci = np.apply_along_axis(upper_confidence_interval, 1, y_distribution)
    if y_real is None:
        y_real = np.empty(df_X.shape[0])
        y_real.fill(np.nan)
    assert len(mu) == len(std) == len(uci) == len(y_real) == df_X.shape[0]

    values = np.vstack([df_X["ligand_amount"].values, mu, std, uci, y_real])
    values = values.T
    df_outcome = pd.DataFrame(data=values, columns=["amount", "mean", "std", "uci", "y_real"])
    df_outcome["ligand_label"] = [l.label for l in df_ligands]
    return df_outcome


def get_data_for_comparison(outcome_df: pd.DataFrame) -> Tuple[
    dict[Molecule, dict[str, list[float]]], list[str], list[str]]:
    data = dict()
    ligands = sorted(set(outcome_df["ligand_label"]))
    labelled_ligands = []
    unlabelled_ligands = []
    for ligand in ligands:
        logging.warning("ligand_label: {}".format(ligand))
        df_ligand = outcome_df.loc[outcome_df['ligand_label'] == ligand]
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


def compare_two_ligand(data, l1, l2, key="mean", seed=SEED, n=1001):
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
                    if compare_two_ligand(data, l, suggestion, key, seed=SEED, n=1001):
                        suggestion = l
                else:
                    if compare_two_ligand(data, suggestion, l, key, seed=SEED, n=1001):
                        suggestion = l
        assert suggestion is not None
        suggestions.append(suggestion)
    return suggestions


def suggestion_using_tuned_model(
        tuned_regressor, ligand_inventory: list[Molecule], df_X_known, df_y_known, df_ligands_known,
        descriptor_csv: FilePath,
):
    # range of amounts
    amounts = df_X_known["ligand_amount"].tolist()
    cmin, cmax = min(amounts), max(amounts)

    # what to predict
    known_ligands = sorted(set(df_ligands_known))
    unknown_ligands = [l for l in ligand_inventory if l not in known_ligands]

    # predict
    df_outcome_unknown = get_predictions(
        reg=tuned_regressor, ligands=unknown_ligands, ligand_inventory=ligand_inventory, descriptor_csv=descriptor_csv,
        cmin=cmin, cmax=cmax, nfake=1000, y_real=None, df_X_real=None, df_ligands=None
    )
    df_outcome_known = get_predictions(
        reg=tuned_regressor, ligands=known_ligands, ligand_inventory=ligand_inventory, descriptor_csv=descriptor_csv,
        cmin=cmin, cmax=cmax, nfake=1000, y_real=df_y_known, df_X_real=df_X_known, df_ligands=df_ligands_known
    )
    df_outcome = pd.concat([df_outcome_known, df_outcome_unknown], axis=0)

    # prepare data for MC comparison
    data, known_ligands_labels, unknown_ligands_labels = get_data_for_comparison(df_outcome)

    # suggestions from unlabelled ligands
    k = 6
    suggestions = pd.DataFrame()
    for key in ["mean", "std", "uci"]:
        suggestions["best_{}_{}".format(k, key)] = suggest_k(k=k, data=data, ligands=unknown_ligands_labels, key=key,
                                                             higher_is_better=True)
        suggestions["worst_{}_{}".format(k, key)] = suggest_k(k=k, data=data, ligands=unknown_ligands_labels, key=key,
                                                              higher_is_better=False)
    return df_outcome, suggestions
