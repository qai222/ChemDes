import glob

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from lsal.alearn import SingleLigandPrediction
from lsal.schema import L1XReactionCollection, L1XReaction
from lsal.utils import createdir, json_load, draw_svg, file_exists, get_workplace_data_folder, get_folder, get_basename, \
    pkl_load

"""
use this to dump data as static assets files
not sure this is better or worse than using a db
"""

_work_folder = get_workplace_data_folder(__file__)
_code_folder = get_folder(__file__)
_basename = get_basename(__file__)
createdir(_work_folder)

ASSETS_FOLDER = f"{_work_folder}/assets"

CAMPAIGN_META = {
    "SL0519": {"is_extra": False, "round_index": 0},
    "AL0907": {"is_extra": True, "round_index": 1},  # extra expt campaign does not correspond to a model
    "AL1026": {"is_extra": False, "round_index": 2},
    "AL1213": {"is_extra": False, "round_index": 3},
}

MODEL_NAMES = [m for m in CAMPAIGN_META if not CAMPAIGN_META[m]['is_extra']]

LIGANDS = json_load("../../MolecularInventory/ligands.json.gz")


def simplify_reaction(rc_master: L1XReactionCollection, reaction: L1XReaction):
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


def dump_svg():
    folder = f"{ASSETS_FOLDER}/svg"
    if set([lig.label for lig in LIGANDS]) == set([get_basename(f) for f in glob.glob(folder + "/*.svg")]):
        logger.info("files already exist, skipping SVG dumping...")
        return
    for lig in tqdm(LIGANDS):
        smiles = lig.smiles
        label = lig.label
        svg_fn = f"{folder}/{label}.svg"
        if file_exists(svg_fn):
            continue
        svg_text = draw_svg(smiles, fn=None)
        with open(svg_fn, "w") as f:
            f.write(svg_text)


def dump_df_pred():
    folder = f"{ASSETS_FOLDER}/pred"
    for model_name in MODEL_NAMES:
        prediction_folder = f"{_work_folder}/../learning_{model_name}/prediction/"
        dump_folder = f"{folder}/{model_name}"
        createdir(dump_folder)
        pred_pkls = sorted(glob.glob(prediction_folder + "/*.pkl"))
        if len(LIGANDS) == len(glob.glob(dump_folder + "/*.parquet")):
            logger.info(f"files already exist, skipping PRED dumping for {model_name}...")
            continue
        logger.info(f"found pred parquet #=={len(pred_pkls)} at {dump_folder}")
        for pred_pkl in tqdm(pred_pkls, desc=f"dumping PRED FOR: {model_name}"):
            slps = pkl_load(pred_pkl)
            for slp in slps:
                slp: SingleLigandPrediction
                x = slp.amounts
                y = slp.pred_mu
                yerr = slp.pred_std
                df_pred = pd.DataFrame(np.array([x, y, yerr]).T, columns=['x', 'y', 'yerr'])
                df_pred.to_parquet(f"{dump_folder}/pred_{slp.ligand.label}.parquet", compression=None)


def dump_df_expt():
    folder = f"{ASSETS_FOLDER}/expt"
    for campaign_name in CAMPAIGN_META:
        dump_folder = f"{folder}/{campaign_name}"
        createdir(dump_folder)
        rc = json_load(f"{_code_folder}/../collect/reaction_collection_{campaign_name}.json.gz")
        rc: L1XReactionCollection

        lig2reactions = rc.ligand_to_reactions_mapping()
        if len(lig2reactions) == len(glob.glob(dump_folder + "/*.parquet")):
            logger.info(f"files already exist, skipping EXPT dumping for {campaign_name}...")
            continue

        for lig, reactions in lig2reactions.items():
            records = [simplify_reaction(rc, r) for r in reactions]
            records = sorted(records, key=lambda x: x['LigandAmount'])
            df = pd.DataFrame.from_records(records)
            df.to_parquet(dump_folder + f"/expt_{lig.label}.parquet", compression=None)


def dump_df_cfpool(ncfs=100):
    folder = f"{ASSETS_FOLDER}/cfpool"
    with open(f"{_work_folder}/../dimred/dmat_chem.npy", 'rb') as f:
        dmat_chem = np.load(f)
    labels = [lig.label for lig in LIGANDS]
    label_to_list_index = {LIGANDS[i].label: i for i in range(len(LIGANDS))}
    # list_index_to_label = {v: k for k, v in label_to_list_index.items()}

    for model_name in MODEL_NAMES:
        df_rkp = pd.read_csv(f"{_code_folder}/../learning_{model_name}/ranking_df/qr_ranking.csv", low_memory=False)
        for vendor_csv in sorted(glob.glob(f"{_code_folder}/../learning_{model_name}/suggestion/vendor/vendor_*.csv")):
            _, u_score, space, direction = get_basename(vendor_csv).split("__")

            df_vendor = pd.read_csv(vendor_csv, low_memory=False)
            rank_method = [c for c in df_vendor.columns if c.startswith("rank_average_")][0]

            folder_mr = f"{folder}/{model_name}___{rank_method}"
            createdir(folder_mr)
            existing_parquets = sorted(glob.glob(f"{folder_mr}/*.parquet"))
            logger.info(f"existing parquets: {len(existing_parquets)} vs vendor_df: {len(df_vendor)}")
            if len(glob.glob(f"{folder_mr}/*.parquet")) == len(df_vendor):
                logger.info(f"files already exist, skipping CFPOOL dumping for {model_name}___{rank_method}...")
                continue
            logger.info(f"calculating cfs for base ligands from: {model_name}: {vendor_csv}")

            label_to_rkp = dict(zip(
                df_rkp['ligand_label'].tolist(), df_rkp[rank_method].tolist()
            ))

            for row in df_vendor.to_dict(orient="records"):
                cluster = row['cluster']
                is_taught = row['is_taught']
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
                        # "model_name": mrl.name,
                        # "rank_method": rank_method,
                        # "ligand_label_base": base_label,
                        "cluster_base": cluster,
                        "is_taught_base": is_taught,
                        "ligand_label_cf": cf_label,
                        "similarity": sim,
                        "rank_value_base": label_to_rkp[base_label],
                        "rank_value_cf": label_to_rkp[cf_label],
                        "rank_value_delta": label_to_rkp[base_label] - label_to_rkp[cf_label]
                    }
                    records_mrb.append(record)
                    if len(records_mrb) > ncfs:
                        break
                df_mrb = pd.DataFrame.from_records(records_mrb)
                df_mrb.to_parquet(f"{folder_mr}/{base_label}.parquet")


if __name__ == '__main__':
    createdir(ASSETS_FOLDER)
    # dump_df_cfpool()
    # dump_df_pred()
    # dump_df_expt()
    dump_svg()  # this is all you need if using mongo backend
