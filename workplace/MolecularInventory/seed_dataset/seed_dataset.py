import datetime
import time

import pandas as pd
from loguru import logger

from lsal.campaign import LoaderInventory
from lsal.schema import Molecule
from lsal.tasks.descal import calculate_cxcalc, calculate_mordred
from lsal.tasks.fgdetect import dfg
from lsal.utils import json_dump, get_basename, get_workplace_data_folder, inspect_tasks
from lsal.utils import json_load, file_exists, plot_molcloud

_workplace_folder = get_workplace_data_folder(__file__)
_basename = get_basename(__file__)

_ligand_list_json_file = f"{_basename}_ligand_list.json"
_solvent_list_json_file = f"{_basename}_solvent_list.json"
_ligand_list_csv_file = f"{_basename}_ligand_list.csv"
_solvent_list_csv_file = f"{_basename}_solvent_list.csv"
_ligand_descriptor_csv_file = "{}_ligand_descriptor_{}.csv".format(_basename,
                                                                   datetime.datetime.now().strftime("%Y_%m_%d"))
_molcloud_figure = "{}_ligand_cloud.png".format(_basename)
_ligand_functional_groups_csv_file = "{}_ligand_functional_group.csv".format(_basename)


def task_0_datagen():
    """
    generate json/csv data for seed dataset inventory
    """
    ligand_loader = LoaderInventory("ligand", "LIGAND")
    ligands = ligand_loader.load(f"{_workplace_folder}/2022_0217_ligand_InChI_mk.xlsx")

    solvent_loader = LoaderInventory("solvent", "SOLVENT")
    solvents = solvent_loader.load(f"{_workplace_folder}/2022_0217_solvent_InChI.csv")

    Molecule.write_molecules(ligands, _ligand_list_csv_file, output="csv")
    Molecule.write_molecules(solvents, _solvent_list_csv_file, output="csv")

    json_dump(ligands, _ligand_list_json_file)
    json_dump(solvents, _solvent_list_json_file)


def task_1_calculate_descriptors():
    """
    calculate molecular descriptors for ligands
    """
    assert file_exists(_ligand_list_json_file)
    mols = json_load(_ligand_list_json_file)
    mordred_df = calculate_mordred(smis=[m.smiles for m in mols])
    cxcalc_df = calculate_cxcalc("cxcalc.exe", smis=[m.smiles for m in mols])

    # # include pka if needed
    # pka_df = opera_pka("ligand_descriptors_OPERA2.7Pred.csv")
    # des_df = pd.concat([pka_df, cxcalc_df, mordred_df], axis=1)

    des_df = pd.concat([cxcalc_df, mordred_df], axis=1)

    descriptors = des_df.columns.tolist()
    logger.info(f"# of descriptors: {len(descriptors)}")
    descriptors = '\n'.join(descriptors)
    logger.info(f"list of descriptors: \n{descriptors}")

    des_df["InChI"] = [m.inchi for m in mols]
    des_df["IUPAC Name"] = [m.iupac_name for m in mols]
    des_df.to_csv(_ligand_descriptor_csv_file, index=False)


def task_2_molmap():
    """
    generate a molmap for ligands
    """
    inv_smis = pd.read_csv(_ligand_list_csv_file)["smiles"].tolist()
    plot_molcloud(inv_smis, 15, _molcloud_figure)


def task_3_functional_group_detection():
    inv_smis = pd.read_csv(_ligand_list_csv_file)["smiles"].tolist()
    data = dfg(inv_smis)
    records = []
    for k, v in data.items():
        r = {'smi': k}
        r.update(v)
        records.append(r)
    df = pd.DataFrame.from_records(records)
    df.to_csv(_ligand_functional_groups_csv_file, index=False)


if __name__ == "__main__":
    tasks = sorted(inspect_tasks(task_header='task_').items(), key=lambda x: x[0])

    perform_tasks = [0, 1, 2, 3]

    tasks = [tasks[i] for i in perform_tasks]
    logger.info(f"tasks loaded: {[t[0] for t in tasks]}")
    for task_function_name, task_function in tasks:
        logger.info(f'working on: {task_function_name}')
        ts1 = time.perf_counter()
        task_function()
        ts2 = time.perf_counter()
        logger.info(f'time cost for ||{task_function_name}||: {round(ts2 - ts1, 3)} s')
