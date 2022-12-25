from lsal.alearn.one_ligand_worker import OneLigandWorker

from lsal.utils import get_basename, get_workplace_data_folder, get_folder

_work_folder = get_workplace_data_folder(__file__)
_code_folder = get_folder(__file__)
_basename = get_basename(__file__)

if __name__ == '__main__':
    worker = OneLigandWorker(
        code_dir=_code_folder,
        work_dir=_work_folder,
        reaction_collection_json=[
            f"{_code_folder}/reaction_collection_train_SL0519.json.gz",
        ],
        prediction_ligand_pool_json=f"{_code_folder}/../../MolecularInventory/ligands.json.gz",
        # test_predict=500,  # uncomment for test `predict`
    )
    worker.run(
        [
            'teach',
            'predict',
            "query",
            "ranking_dataframe",
            "suggestions",
        ]
    )
    worker.final_collect()
