import logging
import logging
import os
import random

import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor

from chemdes.one_ligand import ReactionNcOneLigand
from chemdes.schema import pd, Molecule
from chemdes.tasks.preprocess import load_molecular_descriptors, preprocess_descriptor_df
from chemdes.twinsk.active_learner import TwinActiveLearner
from chemdes.utils import json_load, SEED
from chemdes.utils import strip_extension

random.seed(SEED)
np.random.seed(SEED)


def prepare_ligand_to_fom():
    # load descriptors
    ligands, des_df = load_molecular_descriptors("../ligand_descriptors/molecular_descriptors_2022_03_14.csv",
                                                 warning=True)
    des_df = preprocess_descriptor_df(des_df)

    # load reactions
    reaction_data = json_load("output/2022_0304_LS001_MK003_reaction_data.json", warning=True)
    fom_fields = [k for k in reaction_data[0] if k.startswith("FOM_")]
    ligands_investigated = [d["ligand"] for d in reaction_data]

    # read figure of merit
    fom_df = pandas.DataFrame()
    fom_dict = dict(zip(fom_fields, [[] for i in fom_fields]))
    for ligand in ligands:
        if ligand in ligands_investigated:
            for fom in fom_fields:
                fom_dict[fom].append(reaction_data[ligands_investigated.index(ligand)][fom])
        else:
            for fom in fom_fields:
                fom_dict[fom].append(None)
    for fom in fom_fields:
        fom_df[fom] = fom_dict[fom]

    assert set(ligands_investigated).issubset(set(ligands))

    logging.warning("# of ligands featurized: {}".format(len(ligands)))
    logging.warning("# of features: {}".format(des_df.shape[1]))
    logging.warning("# of ligands with available reaction data (i.e. investigated): {}".format(len(reaction_data)))
    logging.warning("these ligands are: \n{}".format("\n".join([l.iupac_name for l in ligands_investigated])))
    logging.warning("figure of merit strategy: {}".format(fom_df.columns.tolist()))
    assert len(des_df) == len(fom_df)
    # merge = pd.concat([des_df, fom_df], axis=1)
    # merge.to_csv("output/merge.csv", index=False)
    return ligands, des_df, fom_df


def prepare_ligand_concentration_to_fom():
    # load descriptors
    ligands, des_df = load_molecular_descriptors("../ligand_descriptors/molecular_descriptors_2022_03_14.csv",
                                                 warning=True)
    des_df = preprocess_descriptor_df(des_df)
    ligand_to_des_record = dict(zip(ligands, des_df.to_dict(orient="records")))

    # load reactions
    reaction_data = json_load("output/2022_0304_LS001_MK003_reaction_data.json", warning=True)
    ligands_investigated = []
    n_reactions_for_each_ligand = []
    investigated_records = []
    for d in reaction_data:
        ligand = d["ligand"]
        ligands_investigated.append(ligand)
        real_reactions = d["real_reactions"]
        n_reactions_for_each_ligand.append(len(real_reactions))
        for reaction in real_reactions:
            reaction: ReactionNcOneLigand
            fom = reaction.properties["fom"]
            ligand_amount = reaction.ligand.concentration * reaction.ligand.volume
            record = {"ligand_inchi": ligand.inchi, "ligand_iupac_name": ligand.iupac_name, "fom": fom,
                      "ligand_amount": ligand_amount}
            investigated_records.append(record)

    # number of concentrations
    n_real_reactions = int(np.mean(n_reactions_for_each_ligand))
    all_reactions = []
    for d in reaction_data:
        all_reactions += d["real_reactions"] + d["ref_reactions"] + d["blank_reactions"]
    real_amounts = [r.ligand.concentration * r.ligand.volume for r in all_reactions]
    real_amounts_max = max(real_amounts)
    real_amounts_min = min(real_amounts)

    # add fake concentrations for uninvestigated ligand
    unlabelled_records = []
    for i, ligand in enumerate(ligands):
        if ligand in ligands_investigated:
            continue
        rs = np.random.RandomState(SEED + i)
        # fake_amounts = rs.uniform(real_amounts_min, real_amounts_max, n_real_reactions)
        fake_amounts = np.linspace(real_amounts_min, real_amounts_max, 100)
        for fa in fake_amounts:
            record = {"ligand_inchi": ligand.inchi, "ligand_iupac_name": ligand.iupac_name, "fom": np.nan,
                      "ligand_amount": fa}
            unlabelled_records.append(record)

    records = investigated_records + unlabelled_records

    record_ligands = []
    for r in records:
        l = Molecule.from_str(r["ligand_inchi"], "inchi", r["ligand_iupac_name"])
        record_ligands.append(l)
        r.update(ligand_to_des_record[l])
        r.pop("ligand_inchi")
        r.pop("ligand_iupac_name")
    df = pd.DataFrame.from_records(records)
    df_x = df[[c for c in df.columns if c != "fom"]]
    df_y = df["fom"]
    logging.warning("# of ligands featurized: {}".format(len(ligands)))
    logging.warning("# of features: {}".format(des_df.shape[1]))
    logging.warning("# of ligands with available reaction data (i.e. investigated): {}".format(len(reaction_data)))
    logging.warning("these ligands are: \n{}".format("\n".join([l.iupac_name for l in ligands_investigated])))
    return record_ligands, df_x, df_y


def learn_ligand_to_fom():
    """ learn F(ligand) -> fom """
    ligands, des_df, fom_df = prepare_ligand_to_fom()
    base_estimator = RandomForestRegressor(n_estimators=100, random_state=SEED)

    for fom in fom_df.columns:
        if "BINARY" in fom:
            continue

        y_raw = fom_df[fom].values  # may contain nan
        X_raw = des_df.values

        learner = TwinActiveLearner(fom + "__" + base_estimator.__class__.__name__, base_estimator, X_raw, y_raw)
        learner.teach(learner.teachable_indices)
        outcomes = learner.prediction_outcomes(X=None)

        df = outcome_as_df(outcomes, ligands, y_raw, ligand_amounts=None)
        df.to_csv("output/{}.csv".format(learner.name), index=False)


def learn_ligand_concentration_to_fom():
    """ learn F(ligand) -> fom """
    ligands, df_x, df_y = prepare_ligand_concentration_to_fom()
    base_estimator = RandomForestRegressor(n_estimators=100, random_state=SEED)
    X_raw = df_x.values
    y_raw = df_y.values
    learner = TwinActiveLearner("WC__" + base_estimator.__class__.__name__, base_estimator, X_raw, y_raw)
    learner.teach(learner.teachable_indices)
    outcomes = learner.prediction_outcomes(X=None)

    ligand_amounts = df_x["ligand_amount"].tolist()
    df = outcome_as_df(outcomes, ligands, y_raw, ligand_amounts)
    df.to_csv("output/{}.csv".format(learner.name), index=False)


def outcome_as_df(outcomes, ligands: [Molecule], y_real, ligand_amounts=None):
    df = pd.DataFrame()
    df["ligand"] = [l.iupac_name for l in ligands]
    if ligand_amounts is not None:
        assert len(ligands) == len(ligand_amounts)
        df["amount"] = ligand_amounts
    df["y_real"] = y_real
    for k in outcomes:
        df[k] = outcomes[k]
    return df


if __name__ == '__main__':
    logging.basicConfig(filename='{}.log'.format(strip_extension(os.path.basename(__file__))), filemode="w")

    learn_ligand_concentration_to_fom()
    learn_ligand_to_fom()
