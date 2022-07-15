import glob
import time

import numpy as np
import pandas as pd

from lsal.tasks.descal import calculate_cxcalc_raw, _cxcalc_descriptors
from lsal.utils import FilePath

"""
split smiles to run cxcalc in parallel, then combine results together
"""


def split_smi(smi_file: FilePath):
    from itertools import zip_longest
    def grouper(n, iterable, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)

    n = 1000
    with open(smi_file) as f:
        for i, g in enumerate(grouper(n, f, fillvalue=''), 1):
            with open('results/02_descriptor_cxcalc_split/smi_{0:03d}.smi'.format(i), 'w') as fout:
                fout.writelines(g)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def combine_chunk_smis(smi_files):
    c = ""
    for smi_file in smi_files:
        with open(smi_file, "r") as f:
            c += f.read()
    with open("02_descriptor_cxcalc.smi", "w") as f:
        f.write(c)


def chunk_out_to_df(out_file: FilePath, descriptors: list[str] = _cxcalc_descriptors) -> pd.DataFrame:
    with open(out_file, "r") as f:
        lines = f.readlines()
    n_cols = len(lines[1].split())
    colnames = ["id"] + descriptors.copy()
    if "asa" in colnames:
        asa_index = colnames.index("asa")
        # accessible surface area given by positive (ASA+) and negative (ASA À ) partial charges on atoms
        # and also surface area induced by hydrophobic (SA_H) and polar (SA_P) atoms
        colnames = colnames[:asa_index] + ["ASA+", "ASA-", "ASA_H", "ASA_P"] + colnames[asa_index:]
    assert len(colnames) == n_cols

    lines = lines[1:]
    values = np.zeros((len(lines), len(colnames)))
    for i in range(0, len(lines)):
        line = lines[i]
        # print(len(line.split()), i, out_file)
        values[i] = [float(v) for v in line.split()]
    df = pd.DataFrame(data=values, columns=colnames)
    df.pop("id")
    assert not df.isnull().any().any()
    return df


if __name__ == '__main__':

    # takes ~6 h
    split_smi("results/01_pubchem_screen.smi")
    mol_files = sorted(glob.glob("results/02_descriptor_cxcalc_split/smi_*.smi"))
    mol_files_chunks = list(chunks(mol_files, 6))
    for chunk in mol_files_chunks:
        ts1 = time.perf_counter()
        calculate_cxcalc_raw(mol_files=chunk)
        ts2 = time.perf_counter()
        print("cxcalc: {:.2f}s".format(ts2 - ts1))

    mol_files = sorted(glob.glob("results/02_descriptor_cxcalc_split/smi_*.smi"))
    out_files = sorted(glob.glob("results/02_descriptor_cxcalc_split/smi_*.smi.out"))
    dfs = []
    for out_file in out_files:
        df = chunk_out_to_df(out_file)
        dfs.append(df)
    df = pd.concat(dfs, )
    combine_chunk_smis(mol_files)
    df.to_csv("results/02_descriptor_cxcalc.csv", index=False)
