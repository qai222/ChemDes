import pandas as pd
from loguru import logger
from pymongo import errors
from pymongo.collection import Collection
from tqdm import tqdm

from lsal.db.document import prepare_campaign_doc, prepare_lig_doc, prepare_model_doc, prepare_reaction_doc, \
    prepare_pred_docs, prepare_cfpool_docs
from lsal.db.iteration_paths import IterationPaths, load_cps
from lsal.schema import L1XReactionCollection
from lsal.utils import json_load, FilePath


def insert_ligands(ligand_json: FilePath, dimred_csv: FilePath, coll_ligand: Collection, update=False):
    ligands = json_load(ligand_json)
    df_dimred = pd.read_csv(dimred_csv, index_col="ligand_label")
    dimred_dict = df_dimred.to_dict(orient="index")
    for lig in tqdm(ligands, desc="updating ligands..."):
        doc = prepare_lig_doc(lig, dimred_dict)
        if update:
            coll_ligand.replace_one({"_id": doc['_id']}, doc, upsert=True)
        else:
            try:
                coll_ligand.insert_one(doc)
            except errors.DuplicateKeyError as e:
                logger.warning(e)


def iteration_update(
        ip_yaml: FilePath,
        coll_model: Collection,
        coll_ligand: Collection,
        coll_reaction: Collection,
        coll_campaign: Collection,
        coll_pred: Collection,
        coll_cfpool: Collection,
        ligand_json: FilePath,
        dmat_chem_npy: FilePath,
):
    ips = load_cps(ip_yaml)
    ips: list[IterationPaths]

    for ip in ips:
        # update reactions
        rc = json_load(ip.expt_rc_json)
        rc: L1XReactionCollection
        for r in rc.reactions:
            doc_reaction = prepare_reaction_doc(rc, r)
            try:
                coll_reaction.insert_one(doc_reaction)
            except errors.DuplicateKeyError:
                pass

        # update campaign
        doc_campaign = prepare_campaign_doc(ip)
        try:
            coll_campaign.insert_one(doc_campaign)
        except errors.DuplicateKeyError:
            pass

        if ip.model_folder is None:
            continue

        # update model
        doc_model = prepare_model_doc(ip)
        try:
            coll_model.insert_one(doc_model)
        except errors.DuplicateKeyError:
            pass

        # update ligand `SuggestedBy`
        for directed_u_score, ligand_label_list in doc_model['suggestions'].items():
            suggest_by = {"model_id": doc_model["_id"], "directed_u_score": directed_u_score}
            coll_ligand.update_many({"_id": {"$in": ligand_label_list}}, {"$addToSet": {"SuggestedBy": suggest_by}})

        # update ligand `UsedBy`
        coll_ligand.update_many({"_id": {"$in": doc_model['training_ligands']}},
                                {"$addToSet": {"UsedBy": doc_model["_id"]}})

        # update ligand utility scores
        for ranking_record in tqdm(doc_model['ranking_records'],
                                   desc=f"update ligand utility scores: {doc_model['_id']}"):
            ligand_label = ranking_record['ligand_label']
            utility_scores = {k.replace("rank_average_pred_", ""): v for k, v in ranking_record.items() if
                              k.startswith("rank_average_pred_")}
            utility_scores["model_id"] = doc_model['_id']
            coll_ligand.update_one({"_id": ligand_label}, {"$addToSet": {f"utility_scores": utility_scores}})

        # update predictions
        docs = prepare_pred_docs(ip)
        for doc in tqdm(docs, desc="inserting predictions..."):
            try:
                coll_pred.insert_one(doc)
            except errors.DuplicateKeyError:
                pass

        # update cfpool
        docs = prepare_cfpool_docs(ip, json_load(ligand_json), dmat_chem_npy, ncfs=100)
        for doc in tqdm(docs, desc="inserting cfpools..."):
            try:
                coll_cfpool.insert_one(doc)
            except errors.DuplicateKeyError:
                pass
