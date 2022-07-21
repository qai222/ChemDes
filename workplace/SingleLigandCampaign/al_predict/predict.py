import os
import time

import pandas as pd
import psutil
from tqdm import tqdm

from lsal.campaign import SingleWorkflow
from lsal.schema import Molecule
from lsal.utils import pkl_dump, createdir
from lsal.utils import pkl_load


def get_pool_ligands_to_records():
    pool_ligands = []
    des_records = []
    df_inv = pd.read_csv("../../Screening/results/05_summary/inv.csv")
    df_des = pd.read_csv("../../Screening/results/05_summary/des.csv")
    for i, (inv_r, des_r) in enumerate(
            zip(
                df_inv.to_dict(orient="records"),
                df_des.to_dict(orient="records"),
            )
    ):
        ligand = Molecule(
            identifier=inv_r["identifier"],
            iupac_name=inv_r["identifier"],
            name=inv_r["label"],
            smiles=inv_r["smiles"],
            int_label=i,
            mol_type="PLIGAND",
        )
        pool_ligands.append(ligand)
        des_records.append(des_r)
    ligand_to_des_record = dict(zip(pool_ligands, des_records))
    print("# of pool ligands: {}".format(len(ligand_to_des_record)))
    return pool_ligands, ligand_to_des_record


def elapsed_since(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = elapsed_since(start)
        mem_after = get_process_memory()
        print("{}: memory before: {:,}, after: {:,}, consumed: {:,}; exec time: {}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before,
            elapsed_time))
        return result

    return wrapper


if __name__ == '__main__':

    pool_ligands, pool_l2dr = get_pool_ligands_to_records()
    swf: SingleWorkflow

    # fomdef = "fom2"
    # fomdef = "fom3"
    fomdef = "fom1"
    pkl_dump(pool_l2dr, "pool_l2dr.pkl")
    pkl_dump(list(pool_l2dr.keys()), "pool_ligands.pkl")

    swf = pkl_load("../al_workflow/models/onepot/{}--std--notune--all.pkl".format(fomdef))
    for i, (ligand, record) in tqdm(enumerate(list(pool_l2dr.items()))):
        predictions = swf.learner.predict([ligand, ], swf.amounts, {ligand: record})[0]
        save_folder = "predictions_{}".format(fomdef)
        createdir(save_folder)
        saveas = save_folder + "/pred_{0:06d}.pkl".format(i)
        pkl_dump(predictions, saveas, print_timing=False)
"""
# of pool ligands: 44291
loaded fom2--std--notune--all.pkl in: 39.8819 s
44291it [6:21:54,  1.93it/s]

# of pool ligands: 44291
dumped pool_l2dr.pkl in: 0.1460 s
loaded fom3--std--notune--all.pkl in: 42.0127 s
44291it [6:26:47,  1.91it/s]

"""

# @profile
# def one_run():
#     i = 0
#     for chunk in chunks(list(pool_l2dr.items()), 20):
#         ligands = []
#         records = []
#         for c in chunk:
#             ligands.append(c[0])
#             records.append(c[1])
#     # for ligand, record in tqdm(pool_l2dr.items()):
#     #     predictions = swf.learner.predict([ligand, ], swf.amounts, {ligand: record})[0]
#         predictions = swf.learner.predict(ligands, swf.amounts, dict(zip(ligands, records)))
#         saveas = "pred_{0:06d}.pkl".format(i)
#         pkl_dump(predictions, saveas, print_timing=False)
#         break
#         # i += 1
# one_run()
# """
# 100%|██████████| 44291/44291 [6:45:33<00:00,  1.82it/s]
# """

"""
nchunk == 20
# of pool ligands: 45695
loaded fom2--std--notune--all.pkl in: 42.1154 s
one_run: memory before: 3,724,722,176, after: 4,322,451,456, consumed: 597,729,280; exec time: 00:00:06

"""
