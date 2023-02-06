import glob

import numpy as np
import pandas as pd

from lsal.alearn import SingleLigandPrediction
from lsal.db.indices import _get_model_id, _get_campaign_id, DESCRIPTOR_TO_CATEGORY, _get_prediction_id
from lsal.db.iteration_paths import IterationPaths
from lsal.schema import L1XReaction, L1XReactionCollection, Molecule
from lsal.utils import json_load, pkl_load, tqdm, FilePath


def prepare_lig_doc(ligand: Molecule, dimred_dict: dict):
    doc = {
        "_id": ligand.label,
        "smiles": ligand.smiles,
        "inchi": ligand.inchi,
        "cas_number": ligand.properties['cas_number'],
        "complexity_sa_score": ligand.properties['complexity_sa_score'],
        "complexity_BertzCT": ligand.properties['complexity_BertzCT'],
    }

    for desname, desval in ligand.properties['features'].items():
        doc[f"DESCRIPTOR@{DESCRIPTOR_TO_CATEGORY[desname]}@{desname}"] = desval
    doc.update(dimred_dict[doc["_id"]])
    return doc


def prepare_model_doc(ip: IterationPaths):
    assert ip.model_folder is not None
    training_reactions_rc = json_load(ip.path_training_rc_json)
    training_reactions_rc: L1XReactionCollection
    training_reactions = training_reactions_rc.reactions
    ranking_records = pd.read_csv(ip.path_ranking_dataframe).to_dict(orient="records")
    suggestions = dict()
    for directed_uscore, vendor_csv in ip.path_dict_vendor.items():
        suggestions[directed_uscore] = pd.read_csv(vendor_csv)['ligand_label'].tolist()
    doc = {
        "_id": _get_model_id(ip.name),
        "training_reactions": [r.identifier for r in training_reactions],
        "training_ligands": [lig.label for lig in training_reactions_rc.unique_ligands],
        "ranking_records": ranking_records,
        "suggestions": suggestions,
    }
    return doc


def prepare_cfpool_docs(ip: IterationPaths, ligands: list[Molecule], dmat_chem_npy: FilePath, ncfs=100):
    assert ip.model_folder is not None
    with open(dmat_chem_npy, 'rb') as f:
        dmat_chem = np.load(f)
    labels = [lig.label for lig in ligands]
    label_to_list_index = {ligands[i].label: i for i in range(len(ligands))}
    # list_index_to_label = {v: k for k, v in label_to_list_index.items()}
    df_rkp = pd.read_csv(ip.path_ranking_dataframe)

    cf_docs = []

    for directed_u_score, vendor_csv in ip.path_dict_vendor.items():
        df_vendor = pd.read_csv(vendor_csv, low_memory=False)
        rank_method_colname = [c for c in df_vendor.columns if c.startswith("rank_average_")][0]
        rank_method = rank_method_colname.replace("rank_average_pred_", "")
        label_to_rkp = dict(zip(
            df_rkp['ligand_label'].tolist(), df_rkp[rank_method_colname].tolist()
        ))

        for row in tqdm(df_vendor.to_dict(orient="records"),
                        desc=f"working on vendor list: {ip.name}@{directed_u_score}"):
            base_label = row['ligand_label']
            base_index = label_to_list_index[base_label]
            sim_array = dmat_chem[base_index]
            label_to_sim = dict(zip(labels, sim_array.tolist()))
            records_mrb = []
            for cf_label in sorted(labels, key=lambda x: label_to_sim[x], reverse=True):
                cf_index = label_to_list_index[cf_label]
                if cf_index == base_index:
                    continue
                sim = dmat_chem[base_index][cf_index]

                if cf_label not in label_to_rkp:
                    continue

                record = {
                    "_id": f"{_get_model_id(ip.name)}@{directed_u_score}@{base_label}@{cf_label}",
                    "model_id": _get_model_id(ip.name),
                    "directed_u_score": directed_u_score,
                    "rank_method": rank_method,
                    "ligand_label_cf": cf_label,
                    "ligand_label_base": base_label,
                    "similarity": sim,
                    "rank_value_base": label_to_rkp[base_label],
                    "rank_value_cf": label_to_rkp[cf_label],
                    "rank_value_delta": label_to_rkp[base_label] - label_to_rkp[cf_label]
                }
                records_mrb.append(record)
                if len(records_mrb) > ncfs:
                    break
            cf_docs += records_mrb
    return cf_docs


def prepare_campaign_doc(ip: IterationPaths):
    doc = {k: v for k, v in ip.as_dict().items() if not k.startswith("@")}
    doc['_id'] = _get_campaign_id(ip)
    rc = json_load(ip.expt_rc_json)
    rc: L1XReactionCollection
    doc['reactions'] = [r.identifier for r in rc.reactions]
    doc['ligands'] = [lig.label for lig in rc.unique_ligands]
    return doc


def prepare_pred_docs(ip: IterationPaths):
    prediction_folder = ip.path_pred_folder
    pred_docs = []
    for pred_pkl in tqdm(sorted(glob.glob(prediction_folder + "/*.pkl")), desc=f"loading predictions: {ip.name}..."):
        slps = pkl_load(pred_pkl, print_timing=False)
        for slp in slps:
            slp: SingleLigandPrediction
            pred_dict = {
                "_id": _get_prediction_id(ip.name, slp.ligand.label),
                "ligand_label": slp.ligand.label,
                "ligand_amounts": slp.amounts.tolist(),
                "mu": slp.pred_mu.tolist(),
                "std": slp.pred_std.tolist(),
                "model_id": _get_model_id(ip.name),
            }
            pred_docs.append(pred_dict)
    return pred_docs


def prepare_reaction_doc(rc_master: L1XReactionCollection, reaction: L1XReaction):
    if reaction.is_reaction_real:
        rtype = "real"
        lig_amount = reaction.ligand_solution.amount
        lig_label = reaction.ligand.label
        ref_reactions = rc_master.get_reference_reactions(reaction)
        assert len(ref_reactions) > 0
        ref_ids = [refr.identifier for refr in ref_reactions]
        ref_ods = [r.properties['OpticalDensity'] for r in ref_reactions]
        ref_foms = [r.properties['FigureOfMerit'] for r in ref_reactions]
        ref_ods_mean = np.mean(ref_ods)
        ref_foms_mean = np.mean(ref_foms)
        ref_ods_std = np.std(ref_ods)
        ref_foms_std = np.std(ref_foms)
    else:
        lig_label = None
        ref_ids = []
        ref_ods = []
        ref_foms = []
        ref_ods_mean = None
        ref_foms_mean = None
        ref_ods_std = None
        ref_foms_std = None
        if reaction.is_reaction_nc_reference:
            rtype = "ref"
            lig_amount = 0.0
        elif reaction.is_reaction_blank_reference:
            rtype = "blank"
            lig_amount = None
        else:
            raise ValueError

    r = {
        '_id': reaction.identifier,
        'ReactionIdentifier': reaction.identifier,
        'BatchName': reaction.batch_name,
        "LigandLabel": lig_label,
        'OpticalDensity': reaction.properties['OpticalDensity'],
        'FigureOfMerit': reaction.properties['FigureOfMerit'],
        'LigandAmount': lig_amount,
        "ReactionType": rtype,
        'RefIds': ref_ids,
        'RefODs': ref_ods,
        'RefFOMs': ref_foms,
        'RefODs_mu': ref_ods_mean,
        'RefODs_std': ref_ods_std,
        'RefFOMs_mu': ref_foms_mean,
        'RefFOMs_std': ref_foms_std,
    }
    return r
