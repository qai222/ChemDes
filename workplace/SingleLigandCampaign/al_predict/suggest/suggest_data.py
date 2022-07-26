import glob
import re

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from lsal.utils import FilePath, json_dump, get_basename, pkl_load

_remove_isotope = True

_complexities = ["sa_score", "BertzCT"]


def get_fom_def(s: str):
    return re.findall(r"fom\d", s)[0]


def has_isotope(smi: str):
    mol = Chem.MolFromSmiles(smi)
    real_smiles = Chem.MolToSmiles(mol)
    atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
    for atom, isotope in atom_data:
        if isotope:
            atom.SetIsotope(0)
    smiles = Chem.MolToSmiles(mol)
    return real_smiles != smiles


def remove_isotope(df: pd.DataFrame):
    records = df.to_dict(orient="records")
    if _remove_isotope:
        records = [r for r in tqdm(records) if not has_isotope(r["smiles"])]
    df = pd.DataFrame.from_records(records)
    return df


def load_dfs(ps: list[FilePath]):
    dfs = dict()
    for csv in ps:
        fom = get_fom_def(csv)
        if "learn" not in csv:
            df = pd.read_csv(csv)
            if _remove_isotope:
                df = remove_isotope(df)
            dfs[fom] = df
            # df.to_csv(csv, index=False)
    return dfs


if __name__ == '__main__':
    dfs = load_dfs(glob.glob("../vis_predictions/predictions_fom*.csv"))
    smiles_data = pkl_load("../vis_predictions/export_vis_data_ligands.pkl")
    smiles_data = {k: {comp: v[comp] for comp in _complexities + ["img",]} for k, v in smiles_data.items()}
    data = {
        "dfs": dfs,
        "smiles_data": smiles_data,
    }
    json_dump(data, "{}.json".format(get_basename(__file__)))
