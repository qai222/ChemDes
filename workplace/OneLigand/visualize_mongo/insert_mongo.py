from pymongo import MongoClient

from lsal.db.insert import iteration_update

if __name__ == '__main__':
    CLIENT = MongoClient("mongodb://<username>:<password>@<server ip>:<port>")
    DATABASE = CLIENT["ALLiS"]
    COLL_PREDICTION = DATABASE['PREDICTION']
    COLL_CFPOOL = DATABASE['CFPOOL']
    COLL_CAMPAIGN = DATABASE['CAMPAIGN']
    COLL_REACTION = DATABASE['REACTION']
    COLL_LIGAND = DATABASE['LIGAND']
    COLL_MODEL = DATABASE['MODEL']

    # # uncomment if inserting ligand pool
    # from lsal.db.insert import insert_ligands
    # insert_ligands(
    #     "../../MolecularInventory/ligands.json.gz",
    #     dimred_csv="../dimred/df_dimred.csv",
    #     coll_ligand=COLL_LIGAND,
    #     update=False,
    # )

    iteration_update(
        "iteration_paths.yaml",
        COLL_MODEL,
        COLL_LIGAND,
        COLL_REACTION,
        COLL_CAMPAIGN,
        COLL_PREDICTION,
        COLL_CFPOOL,
        ligand_json="../../MolecularInventory/ligands.json.gz",
        dmat_chem_npy="../../../workplace_data/OneLigand/dimred/dmat_chem.npy",
    )
