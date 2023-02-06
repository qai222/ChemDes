"""
fit the initial dataset one ligand by another
note: this costs a lot space...
"""
import glob
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from loguru import logger

from lsal.alearn.one_ligand_worker import SingleLigandPrediction, SingleLigandLearner
from lsal.schema import L1XReactionCollection, Molecule
from lsal.utils import get_basename, get_workplace_data_folder, get_folder
from lsal.utils import json_load, json_dump, createdir

_work_folder = get_workplace_data_folder(__file__)
_code_folder = get_folder(__file__)
_basename = get_basename(__file__)
createdir(_work_folder)

RC = json_load("../OneLigand/learning_SL0519/reaction_collection_train_SL0519.json.gz")
RC: L1XReactionCollection
LigPool = RC.unique_ligands
LigPool = LigPool
L2Rs = RC.ligand_to_reactions_mapping()


def seed_one_by_one(
        init_lig: Molecule, rank_method="rank_average_pred_std", use_al=True,
):
    if use_al:
        wdir = f"{_work_folder}/al/init__{init_lig.label}"
    else:
        wdir = f"{_work_folder}/rand/init__{init_lig.label}"

    createdir(wdir)

    taught_ligs = [init_lig, ]
    learner = SingleLigandLearner.init_trfr('FigureOfMerit', wdir=wdir)
    learner.teach_reactions(L1XReactionCollection(L2Rs[init_lig]), f"{wdir}/t__{len(taught_ligs)}.pkl")
    ligand_amounts = learner.latest_teaching_record.reaction_ids.amount_geo_space(200)
    obo_data = []
    while len(taught_ligs) < len(LigPool):
        slps = learner.predict(LigPool, ligand_amounts)
        slps: list[SingleLigandPrediction]

        df_all = SingleLigandPrediction.calculate_ranking(LigPool, slps)
        not_taught = [lig for lig in LigPool if
                      lig not in learner.latest_teaching_record.reaction_ids.unique_ligands]
        eval_result_all = learner.eval_against_reactions(RC)
        rc_not_taught = []
        for li in L2Rs:
            if li in not_taught:
                rc_not_taught += L2Rs[li]
        rc_not_taught = L1XReactionCollection(rc_not_taught)
        eval_result_not_taught = learner.eval_against_reactions(rc_not_taught)

        obo_data.append(
            [
                [lig for lig in taught_ligs],
                [lig for lig in not_taught],
                df_all,
                eval_result_not_taught,
                eval_result_all,
            ]
        )

        df = SingleLigandPrediction.calculate_ranking(not_taught, slps)
        i = df[rank_method].argmax()
        suggest_lig = not_taught[i]
        logger.critical(f"actual suggest: {suggest_lig.label}")
        if not use_al:
            suggest_lig = random.sample(not_taught, 1)[0]
        taught_ligs.append(suggest_lig)
        rs = []
        for lig in taught_ligs:
            rs += L2Rs[lig]
        logger.info(f"suggest: {suggest_lig.label}, now teaching: nligands={len(taught_ligs)} nrs={len(rs)}")
        learner.teach_reactions(L1XReactionCollection(rs), f"{wdir}/t__{len(taught_ligs)}.pkl")
    return obo_data


def run_obo(use_al=True):
    random.seed(28931)
    ligs = random.sample(LigPool, 10)

    for lig in tqdm.tqdm(ligs):
        data = seed_one_by_one(lig, use_al=use_al)
        json_dump(data, f"obo_{lig.label}.json.gz", gz=True)


def get_obo_eval_xy_2d(use_al=True, average_over_untaught=True):
    rank_method = "rank_average_pred_std"
    if use_al:
        files = "./al/obo_LIGAND-*.json.gz"
    else:
        files = "./rand/obo_LIGAND-*.json.gz"
    xs_2d = []
    ys_confidence_2d = []
    ys_mae_2d = []
    records = []
    for json_file in glob.glob(files):
        init_lig_lab = get_basename(json_file).replace("obo_", "").replace(".json", "")
        d = json_load(json_file)
        for taught_ligs, untaught_ligs, df, _, res_all in d:
            x = len(taught_ligs)
            n_all = len(taught_ligs) + len(untaught_ligs)
            n_untaught = len(untaught_ligs)

            if average_over_untaught:
                untaught_labs = [lig.label for lig in untaught_ligs]
                df_untaught = df[df['ligand_label'].isin(untaught_labs)]
                y_conf = sum(df_untaught[rank_method].tolist()) / n_untaught
                res_not_taught = [r for r in res_all if r['ligand'] in untaught_labs]
                # df_untaught = df[df['ligand_label'].isin(untaught_labs)]
                y_mae = np.mean([abs(r['y_pred_mu'] - r['y']) for r in res_not_taught])
            else:
                y_conf = sum(df[rank_method].tolist()) / n_all
                y_mae = np.mean([abs(r['y_pred_mu'] - r['y']) for r in res_all])
            record = dict(
                x=x,
                init_lig_lab=init_lig_lab,
                y_conf=y_conf,
                y_mae=y_mae
            )
            records.append(record)

    return pd.DataFrame.from_records(records)


def plot_obo():
    ebar = ("ci", 95)
    df_al = get_obo_eval_xy_2d(use_al=True)
    df_rand = get_obo_eval_xy_2d(use_al=False)
    fig, ax = plt.subplots()
    g1 = sns.pointplot(
        df_al, x="x", y="y_mae",
        errorbar=ebar,
        capsize=.4, join=False, ax=ax,
        color="r", label="AL"
    )
    g2 = sns.pointplot(
        df_rand, x="x", y="y_mae",
        errorbar=ebar,
        capsize=.4, join=False, ax=ax,
        color="k", label="Random"
    )
    plt.setp(g1.collections, alpha=.3)
    plt.setp(g2.collections, alpha=.3)
    plt.setp(g1.lines, alpha=.3)
    plt.setp(g2.lines, alpha=.3)
    ax.set_ylabel("MAE")
    ax.set_xlabel("Ligands in training")
    ax.legend()

    fig.savefig("obo_MAE.png", dpi=600)
    fig.savefig("obo_MAE.pdf")

    plt.clf()
    fig, ax = plt.subplots()
    g1 = sns.pointplot(
        df_al, x="x", y="y_conf",
        errorbar=ebar,
        capsize=.4, join=False, ax=ax,
        color="r", label="AL"
    )
    plt.setp(g1.collections, alpha=.3)
    plt.setp(g1.lines, alpha=.3)
    plt.setp(g2.lines, alpha=.3)
    ax.set_ylabel("Uncertainty")
    ax.set_xlabel("Ligands in training")
    ax.legend()

    fig.savefig("obo_uncertainty.png", dpi=600)
    fig.savefig("obo_uncertainty.pdf")


if __name__ == '__main__':
    run_obo(use_al=True)
    run_obo(use_al=False)
    plot_obo()
