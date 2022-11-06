from lsal.alearn.one_ligand_worker import OneLigandWorker, L1XReactionCollection

from lsal.utils import get_basename, get_workplace_data_folder, get_folder, json_load, json_dump

_work_folder = get_workplace_data_folder(__file__)
_code_folder = get_folder(__file__)
_basename = get_basename(__file__)

"""
NOTE:
- reaction collection of AL1026 contains 14 ligands
- 3 ligands from AL0907 are added to training as duplicate `mf`:
      mf0002,POOL-00036699,CN1CCC(CCCC2CCN(C)CC2)CC1
      mf0005,POOL-00042820,CCCCCCCCOCCCCCCCC
      mf0007,POOL-00042314,CCCCCCOCCCCCC
- total 14(AL1026)+3(AL0907)+21(SL0519)=38 ligands are used in fitting
- total # of reactions = 38 * 40 - 1(outlier)
"""

rc1026 = json_load(f"{_work_folder}/../collect/reaction_collection_AL1026.json.gz")
rc0519 = json_load(f"{_work_folder}/../collect/reaction_collection_SL0519.json.gz")
rc0907 = json_load(f"{_work_folder}/../collect/reaction_collection_AL0907.json.gz")
reactions_from_0907 = [
    r for r in rc0907.real_reactions if r.ligand.label in [
        "POOL-00036699",
        "POOL-00042820",
        "POOL-00042314",
    ]
]
rc0907 = L1XReactionCollection(reactions_from_0907)
rc_train = L1XReactionCollection(rc0519.reactions + rc1026.reactions + rc0907.reactions)
rc_train_json = f"{_code_folder}/reaction_collection_train_1026.json.gz"
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
            'teach',
            'predict',
            "query",
            "ranking_dataframe",
            "suggestions",
        ]
    )
    worker.final_collect()
