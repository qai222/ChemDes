from loguru import logger
from pymongo import errors
from pymongo.collection import Collection

from lsal.db.iteration_paths import IterationPaths
from lsal.schema import L1XReaction, Molecule


def _get_descriptor_category():
    descriptor_explanation = """
    # polarizability
    avgpol axxpol ayypol azzpol molpol dipole SLogP
    # surface
    ASA+ ASA- ASA_H ASA_P asa maximalprojectionarea maximalprojectionradius minimalprojectionarea minimalprojectionradius psa vdwsa volume   
    # count
    chainatomcount chainbondcount fsp3 fusedringcount rotatablebondcount acceptorcount accsitecount donorcount donsitecount mass nHeavyAtom fragCpx nC nO nN nP nS nRing
    # topological
    hararyindex balabanindex hyperwienerindex wienerindex wienerpolarity
    """
    cat = dict()
    lines = descriptor_explanation.strip().split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            k = line.strip().split()[1]
            des_names = lines[i + 1].strip().split()
            cat[k] = des_names
    des_to_cat = dict()
    for category, des_list in cat.items():
        for des in des_list:
            des_to_cat[des] = category
    return cat, des_to_cat


def _get_ligand_id(ligand: Molecule):
    return ligand.label


def _get_reaction_id(reaction: L1XReaction):
    return reaction.identifier


def _get_model_id(model_name: str):
    return f"MODEL:{model_name}"


def _get_prediction_id(model_name: str, ligand_label: str):
    return f"PREDICTION:{model_name}@{ligand_label}"


def _get_campaign_id(itpa: IterationPaths):
    return f"CAMPAIGN:{itpa.name}"


def insert_many_ignore_duplicates(collection: Collection, data: list[dict]):
    try:
        collection.insert_many(
            data,
            ordered=False,
            bypass_document_validation=True
        )
    except errors.BulkWriteError as e:
        logger.warning(e.details)


CATEGORY_TO_DESCRIPTOR_LIST, DESCRIPTOR_TO_CATEGORY = _get_descriptor_category()
