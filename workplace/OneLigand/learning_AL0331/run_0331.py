from lsal.alearn.one_ligand_worker import OneLigandWorker, L1XReactionCollection

from lsal.utils import get_basename, get_workplace_data_folder, get_folder, json_load, json_dump

_work_folder = get_workplace_data_folder(__file__)
_code_folder = get_folder(__file__)
_basename = get_basename(__file__)

rc_al0303 = json_load(f"{_code_folder}/../learning_AL0303/reaction_collection_train_AL0303.json.gz")
rc_expt0331 = json_load(f"{_code_folder}/../collect/reaction_collection_AL0331.json.gz")

rc_train = L1XReactionCollection(rc_al0303.reactions + rc_expt0331.reactions)
rc_train_json = f"{_code_folder}/reaction_collection_train_AL0331.json.gz"
json_dump(rc_train, rc_train_json, gz=True)

if __name__ == '__main__':
    worker = OneLigandWorker(
        code_dir=_code_folder,
        work_dir=_work_folder,
        reaction_collection_json=[
            rc_train_json,
        ],
        prediction_ligand_pool_json=f"{_code_folder}/../../MolecularInventory/ligands.json.gz",
        # test_predict=500,  # uncomment for test `predict`
    )
    worker.run(
        [
            # 'teach',
            # 'predict',
            # "query",
            # "ranking_dataframe",
            "suggestions",
        ]
    )
    worker.final_collect()
