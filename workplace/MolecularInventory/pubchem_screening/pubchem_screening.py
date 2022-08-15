import glob
import time

import pandas as pd
from loguru import logger
from rdkit.Chem import MolFromSmiles, MolToSmiles, Descriptors
from tqdm import tqdm

from lsal.schema import Molecule
from lsal.tasks.descal import calculate_cxcalc_parallel, chunk_out_to_df, calculate_mordred, combine_files
from lsal.tasks.screen import delta_plot, domain_range, get_smi2record, smi2poolinv
from lsal.utils import file_exists, has_isotope
from lsal.utils import get_workplace_data_folder, inspect_tasks
from lsal.utils import read_smi, get_basename, json_load, json_dump
from lsal.utils import remove_stereo, parse_formula, createdir, write_smi

_workplace_data_folder = get_workplace_data_folder(__file__)
_basename = get_basename(__file__)

# parameters
_smiles_screening_mwmax = 400  # max molecular weight
_allowed_elements = {"C", "H", "O", "N", "P", "S", "Br", "F", }

# large files to be saved in workplace data folder
_pubchem_compound_csv = f"{_workplace_data_folder}/PubChem_compound.csv"
_pubchem_compound_csv_after_smiles_screen = f"{_workplace_data_folder}/PubChem_compound_smiles_screened.csv"
_workfolder_for_descriptor_calculation = f"{_workplace_data_folder}/calculate/"
_workfolder_for_screen_by_descriptor = f"{_workplace_data_folder}/screen_by_descriptor/"
_cxcalc_output = f"{_workplace_data_folder}/des_cxcalc.csv"
_cxcalc_input = f"{_workplace_data_folder}/des_cxcalc.smi"
_mordred_output = f"{_workplace_data_folder}/des_mordred.csv"
_screened_json = f"{_workplace_data_folder}/screened_ligands_and_descriptors.json"

createdir(_workfolder_for_descriptor_calculation)


def task_0_collect_pubchem_compounds():
    """
    two ways to download pubchem compounds related to a vendor/vendors
    - use direct search from ncbi, the url is:
        https://www.ncbi.nlm.nih.gov/pccompound?term=(%22has%20src%20vendor%22%5BFilter%5D)%20AND%20%22Sigma-Aldrich%22%5BSourceName%5D
    - use ChemScraper.download_vendor_compounds
    """
    if file_exists(_pubchem_compound_csv):
        logger.warning(f"file already exists, download skipped: {_pubchem_compound_csv}")
        return
    from ChemScraper import download_vendor_compounds
    download_vendor_compounds(
        vendors=('Sigma-Aldrich',),
        saveas=_pubchem_compound_csv,
        count_limit=10,  # comment out to download all entries
        field_string='cid,mw,mf,isosmiles',
    )


def task_1_smiles_screening():
    df = pd.read_csv(_pubchem_compound_csv)
    logger.info(f'loaded # of molecules: {len(df)}')
    keep_tuples = []
    for t in tqdm(df.itertuples(index=False, name=None)):
        cid, mw, formula, smiles = t
        # - more than one component
        if "." in smiles:
            continue
        # - apparent charge in formula
        if "+" in formula or "-" in formula:
            continue
        # - invalid formula
        try:
            fdict = parse_formula(formula)
        except ValueError:
            continue
        # - carbon should be there
        if "C" not in fdict.keys():
            continue
        # - not a subset of allowed elements
        if not set(fdict.keys()).issubset(_allowed_elements):
            continue
        smiles = remove_stereo(smiles)
        # - invalid smiles
        try:
            m = MolFromSmiles(smiles)
        except ValueError:
            continue
        # - any isotope
        if has_isotope(m):
            continue
        # - mw larger than 400
        mw = Descriptors.ExactMolWt(m)
        if mw > _smiles_screening_mwmax:
            continue
        smiles = MolToSmiles(m)
        keep_tuples.append((cid, smiles))
    df_screened = pd.DataFrame(keep_tuples, columns=['cid', 'smiles'])
    df_screened.drop_duplicates(subset=['smiles', ], inplace=True)
    df_screened.to_csv(_pubchem_compound_csv_after_smiles_screen, index=False)
    logger.info(f'screened to # of molecules: {len(df_screened)}')
    return df_screened


def task_2_descriptor_cxcalc():
    """
    calculate descriptors in parallel
    """
    assert file_exists(_pubchem_compound_csv_after_smiles_screen)
    smis = pd.read_csv(_pubchem_compound_csv_after_smiles_screen)['smiles']

    if len(glob.glob(f"{_workfolder_for_descriptor_calculation}/*.out")) > 0:
        logger.warning(f"use existing cxcalc output files: {_workfolder_for_descriptor_calculation}")
        in_files = sorted(glob.glob(f"{_workfolder_for_descriptor_calculation}/*.smi"))
        combine_files(in_files, _cxcalc_input)
        out_files = sorted(glob.glob(f"{_workfolder_for_descriptor_calculation}/*.out"))
        dfs = []
        for out_file in out_files:
            try:
                df = chunk_out_to_df(out_file)
            except Exception as e:
                logger.critical(f'error in parsing: {out_file}')
                raise e
            dfs.append(df)
        df = pd.concat(dfs, )
    else:
        logger.warning("run cxcalc in parallel")
        df = calculate_cxcalc_parallel(
            logger, smis=smis, workdir=_workfolder_for_descriptor_calculation,
            combined_input=_cxcalc_input, nproc=6, chunk_size=1000,
        )
    all_input_smis = read_smi(_cxcalc_input)
    data = {smi: r[1].to_dict() for smi, r in zip(all_input_smis, df.iterrows())}
    records = []
    final_smis = []
    for smi in smis:
        try:
            records.append(data[smi])
            final_smis.append(smi)
        except KeyError:
            continue
    df = pd.DataFrame.from_records(records)
    assert len(df) == len(final_smis)
    df.to_csv(_cxcalc_output, index=False)
    write_smi(final_smis, _cxcalc_input)


def task_3_descriptor_mordred():
    assert file_exists(_cxcalc_input)
    smis = read_smi(_cxcalc_input)
    mordred_df = calculate_mordred(smis)
    mordred_df.to_csv(_mordred_output, index=False)


def task_4_screen_by_des():
    # load pool smis
    mordred_df = pd.read_csv(_mordred_output)
    cxcalc_df = pd.read_csv(_cxcalc_output)
    cxcalc_smis = read_smi(_cxcalc_input)
    assert len(mordred_df) == len(cxcalc_df) == len(cxcalc_smis)
    pool_smis = cxcalc_smis

    # combine cxcalc and mordred
    cm_df = pd.concat([cxcalc_df, mordred_df], axis=1)
    available_features = cm_df.columns.tolist()
    logger.info("available features: {}".format(available_features))
    smi2record = get_smi2record(pool_smis, cm_df, None)

    # delta plot
    _seed_dataset_descriptors = "../seed_dataset/seed_dataset_ligand_descriptor_2022_06_16.csv"
    logger.info(f"screen against: {_seed_dataset_descriptors}")
    lim = domain_range(_seed_dataset_descriptors, available_features)
    delta_dfs = delta_plot(
        pool_smis, lim, cm_df,
        f"{_workfolder_for_screen_by_descriptor}/{get_basename(__file__)}.png",
        available_features,
        wdir=_workfolder_for_screen_by_descriptor,
        logger=logger,
    )


def task_5_collect_screening_results(delta_string="0.40"):
    records, smis = json_load(f"{_workfolder_for_screen_by_descriptor}/{delta_string}.json")
    df_inv = smi2poolinv(smis)
    df_des = pd.DataFrame.from_records(records)
    df_des.to_csv('screened_ligands_des.csv', index=False)
    df_inv.to_csv('screened_ligands_inv.csv', index=False)
    pool_ligands = []
    des_records = []
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
    ligand_and_des_record = list(zip(pool_ligands, des_records))
    json_dump(ligand_and_des_record, _screened_json)


if __name__ == "__main__":
    tasks = sorted(inspect_tasks(task_header='task_').items(), key=lambda x: x[0])

    perform_tasks = [1, 2, 3, 4, 5]

    tasks = [tasks[i] for i in perform_tasks]
    logger.info(f"tasks loaded: {[t[0] for t in tasks]}")
    for task_function_name, task_function in tasks:
        logger.info(f'working on: {task_function_name}')
        ts1 = time.perf_counter()
        task_function()
        ts2 = time.perf_counter()
        logger.info(f'time cost for ||{task_function_name}||: {round(ts2 - ts1, 3)} s')
