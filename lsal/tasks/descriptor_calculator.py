import glob
import subprocess
import time
from collections import Counter
from functools import reduce
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from lsal.utils import to_float, FilePath, file_exists, get_timestamp, removefile, os, chunks, combine_files, read_smi

"""
Three calculators are used:
1. `cxcalc` in CHEMAXON: 
    - remember to add the `bin` folder to `PATH`
    - list of descriptors https://docs.chemaxon.com/display/docs/cxcalc-calculator-functions.md

2. `mordred` by Moriwaki:
    - cite 10.1186/s13321-018-0258-y
    - list of descriptors https://mordred-descriptor.github.io/documentation/master/descriptors.html

3. `opera` (pKa only) by Mansouri:
    - cite 10.1186/s13321-019-0384-1
"""

_cxcalc_descriptors = """
# polarizability
avgpol axxpol ayypol azzpol molpol
# surface
asa maximalprojectionarea maximalprojectionradius minimalprojectionarea minimalprojectionradius psa vdwsa volume   
# count
chainatomcount chainbondcount fsp3 fusedringcount rotatablebondcount acceptorcount accsitecount donorcount donsitecount mass
# topological
hararyindex balabanindex hyperwienerindex wienerindex wienerpolarity
# polarizability, somehow cxcalc always put dipole at the end
dipole
"""
_cxcalc_descriptors = [l for l in _cxcalc_descriptors.strip().split("\n") if not l.startswith("#")]
_cxcalc_descriptors = [l.split() for l in _cxcalc_descriptors]
_cxcalc_descriptors = reduce(lambda x, y: x + y, _cxcalc_descriptors)
_cxcalc_descriptors = list(_cxcalc_descriptors)


def parse_cxcalc_out(
        out_string: str, descriptors: list[str] = _cxcalc_descriptors,
        in_smis: Union[list[str], str] = None
):
    if isinstance(in_smis, str):
        assert file_exists(in_smis)
        in_smis = read_smi(in_smis)

    lines = out_string.splitlines()
    lines = lines[1:]
    assert len(lines) >= 1

    # descriptor column names
    colnames = ["id"] + descriptors.copy()
    if "asa" in colnames:
        asa_index = colnames.index("asa")
        # accessible surface area given by positive (ASA+) and negative (ASA À ) partial charges on atoms
        # and also surface area induced by hydrophobic (SA_H) and polar (SA_P) atoms
        colnames = colnames[:asa_index] + ["ASA+", "ASA-", "ASA_H", "ASA_P"] + colnames[asa_index:]

    # check number of cells for each row
    n_cells = [len(line.split()) for line in lines]
    if len(set(n_cells)) > 1:
        logger.critical(f"# of cells are not consistent: {Counter(n_cells)}")

    values = np.zeros((len(lines), len(colnames)))
    valid_indices = []
    for i, line in enumerate(lines):
        # print(len(line.split()), i, out_file)
        try:
            values[i] = [float(v) for v in line.split()]
            valid_indices.append(i)
        except ValueError:
            logger.warning(f'the {i}th line is funny: {line}')

    df = pd.DataFrame(data=values, columns=colnames)
    df.pop("id")
    if in_smis is None:
        return df
    else:
        df = df.iloc[valid_indices]
        return [in_smis[i] for i in valid_indices], df


def calculate_cxcalc(smis: list[str], bin: Union[Path, str] = "cxcalc.exe",
                     descriptors: list[str] = _cxcalc_descriptors, remove_mol_file=True):
    mol_file = "cxcalc_tmp_{}.smi".format(get_timestamp())
    s = "\n".join(smis)
    assert not file_exists(mol_file)
    with open(mol_file, "w") as f:
        f.write(s)
    cmd = [bin, ] + [mol_file, ] + descriptors
    result = subprocess.run(cmd, capture_output=True)
    data = result.stdout.decode("utf-8").strip()
    final_input_smis, df = parse_cxcalc_out(data, descriptors, in_smis=smis)
    assert not df.isnull().any().any()
    if remove_mol_file:
        removefile(mol_file)
    return final_input_smis, df


_mordred_descriptors = (
    "SLogP", "nHeavyAtom", "fragCpx", "nC", "nO", "nN", "nP", "nS", "nRing",
)


def calculate_mordred(smis: list[str], descriptor_names=_mordred_descriptors) -> pd.DataFrame:
    from mordred import Calculator, descriptors, Descriptor
    from rdkit import Chem
    used_descriptors = []
    for des in Calculator(descriptors).descriptors:
        des: Descriptor
        if des.__str__() in descriptor_names:
            used_descriptors.append(des)
    assert len(used_descriptors) == len(descriptor_names)
    calc = Calculator(used_descriptors)
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    df = calc.pandas(mols)
    assert not df.isnull().any().any()
    return df


def opera_pka(opera_output: Union[FilePath, pd.DataFrame] = "mols-smi_OPERA2.7Pred.csv") -> pd.DataFrame:
    """
    use opera to predict aqueous pKa of 2d molecular structures
    1. download opera release 2.7 from https://github.com/kmansouri/OPERA
    2. write molecules to mols.smi file
    3. run opera pka predictor, output to csv
    4. run this function to read csv
    """
    # parse output
    if isinstance(opera_output, str):
        assert file_exists(opera_output)
        opera_df = pd.read_csv(opera_output)
    else:
        opera_df = opera_output
    opera_df: pd.DataFrame
    opera_df = opera_df[["pKa_a_pred", "pKa_b_pred"]]

    records = []
    for r in opera_df.to_dict(orient="records"):
        pka = to_float(r["pKa_a_pred"])
        pkb = to_float(r["pKa_b_pred"])
        assert not (pka is None and pkb is None)
        if pka is None:
            is_acidic = 0
            p = pkb
        else:
            is_acidic = 1
            p = pka
        records.append({"is_acidic": is_acidic, "pKa": p})
    return pd.DataFrame.from_records(records)


def calculate_cxcalc_raw(bin="cxcalc.exe", mol_files: list = (), descriptors=_cxcalc_descriptors):
    child_ps = []
    for mf in mol_files:
        out_file = mf + ".out"
        cmd = [bin, ] + [mf, ] + descriptors
        with open(out_file, "w") as f:
            p = subprocess.Popen(cmd, stdout=f)
            child_ps.append(p)
    for cp in child_ps:
        cp.wait()


def cxcalc_parallel_input_write(
        smis=list[str],
        workdir: FilePath = "./",
        chunk_size=1000,
        nproc=6,
        input_template='smi_{0:03d}.smi',
):
    input_smi_files = []
    input_smi_chunks = []
    for ichunk, smi_chunk in enumerate(chunks(smis, chunk_size)):
        input_file = os.path.join(workdir, input_template.format(ichunk))
        with open(input_file, 'w') as f:
            f.write('\n'.join(smi_chunk))
        input_smi_chunks.append(list(smi_chunk))
        input_smi_files.append(input_file)
    return input_smi_files, input_smi_chunks


def cxcalc_parallel_collect_results(
        input_files: list[FilePath],
        out_files: list[FilePath],
        descriptors: list[str] = _cxcalc_descriptors,
):
    # collect results
    dfs = []
    final_input_smis = []
    for input_file, out_file in zip(input_files, out_files):
        with open(out_file) as out:
            out_string = out.read()
        in_smis, df = parse_cxcalc_out(out_string, descriptors, in_smis=input_file)
        final_input_smis += in_smis
        dfs.append(df)
    df = pd.concat(dfs)
    return final_input_smis, df


def cxcalc_parallel_calculate(
        bin: Union[Path, str] = "cxcalc.exe",
        smis=list[str],
        descriptors: list[str] = _cxcalc_descriptors,
        workdir: FilePath = "./",
        chunk_size=1000,
        nproc=6,
        input_template='smi_{0:03d}.smi',
        combined_input='input.smi',
):
    assert len(glob.glob(f'{workdir}/*')) == 0, f'work dir is not empty!: {workdir}'
    input_smi_files, input_smi_chunks = cxcalc_parallel_input_write(smis, workdir, chunk_size, nproc, input_template)
    combine_files(input_smi_files, combined_input)
    input_files_chunks = list(chunks(input_smi_files, nproc))
    # run cxcalc in parallel
    for chunk in tqdm(input_files_chunks, desc='cxcalc parallel'):
        ts1 = time.perf_counter()
        calculate_cxcalc_raw(bin=bin, mol_files=chunk, descriptors=descriptors)
        ts2 = time.perf_counter()
        logger.info('cxcalc batch cost: {:.3f} s'.format(ts2 - ts1))

    # collect results
    out_files = sorted(glob.glob(f"{workdir}/*.out"))
    assert len(input_smi_files) == len(out_files)
    final_input_smis, df = cxcalc_parallel_collect_results(input_smi_files, out_files, descriptors)
    return final_input_smis, df
