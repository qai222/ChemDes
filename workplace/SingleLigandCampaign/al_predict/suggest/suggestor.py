import numba
import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from lsal.tasks.sampler import ks_sampler
from lsal.utils import json_load, scale_df

SUGGEST_K = 200
SUGGEST_TOP = True
COMPLEXITY_INDEX = 0
COMPLEXITY_PERCENTILE = 25
NN = 5
MD = 0.2
MCS = 8
MS = 10

_data = json_load("suggest_data.json")
_dfs = _data["dfs"]
_smiles_data = _data["smiles_data"]
_available_fom = ("fom2", "fom3")
_available_metric = ('mu-top2%', 'std-top2%mu', 'std')
_complexity_descriptors = ["sa_score", "BertzCT"]
_distance_metric = "manhattan"

_smiles_to_cid = {}
for i, r in tqdm(pd.read_csv("../../../Screening/results/01_pubchem_screen.csv").iterrows()):
    _smiles_to_cid[r["smiles"]] = r["cid"]


def get_img_str(smi: str):
    return _smiles_data[smi]["img"]


@numba.njit()
def tanimoto_dist(a, b):
    dotprod = np.dot(a, b)
    tc = dotprod / (np.sum(a) + np.sum(b) - dotprod)
    return 1.0 - tc


def get_pool_ligands_to_records():
    pool_ligands = []
    des_records = []
    df_inv = pd.read_csv("../../../Screening/results/05_summary/inv.csv")
    df_des = pd.read_csv("../../../Screening/results/05_summary/des.csv")
    for i, (inv_r, des_r) in enumerate(
            zip(
                df_inv.to_dict(orient="records"),
                df_des.to_dict(orient="records"),
            )
    ):
        ligand = inv_r["smiles"]
        pool_ligands.append(ligand)
        des_records.append(des_r)
    ligand_to_des_record = dict(zip(pool_ligands, des_records))
    print("# of pool ligands: {}".format(len(ligand_to_des_record)))
    return ligand_to_des_record


SmiToRecords = get_pool_ligands_to_records()


def suggest_from_predictions(
        fom_type: str, metric: str, k: int, top: bool = True,
        complexity_type: str = _complexity_descriptors[0], complexity_cutoff: float = 25
):
    df = _dfs[fom_type][["smiles", metric]]
    df = df.assign(**{complexity_type: [_smiles_data[smi][complexity_type] for smi in df["smiles"]]})

    comp_values = df[complexity_type]
    complexity_cutoff = float(np.percentile(comp_values, complexity_cutoff))

    df: pd.DataFrame
    records = df.to_dict(orient="records")
    records = sorted([r for r in records if r[complexity_type] <= complexity_cutoff], key=lambda x: x[metric],
                     reverse=top)[:k]
    return pd.DataFrame.from_records(records)


def distmat_features(smis: list[str]):
    records = [SmiToRecords[smi] for smi in smis]
    df = pd.DataFrame.from_records(records)
    df = scale_df(df)
    distance_matrix = pairwise_distances(df.values, metric=_distance_metric)
    return distance_matrix


def distmat_fps(smiles: list[str]):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    # get fingerprints
    X = []
    for mol in mols:
        arr = np.zeros((0,))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X.append(arr)
    return pairwise_distances(X, metric=tanimoto_dist)


def make_suggestions(
        fom_type: str = _available_fom[0], metric: str = _available_metric[0],
        k: int = SUGGEST_K,
        top: bool = SUGGEST_TOP,
        complexity_type: str = _complexity_descriptors[COMPLEXITY_INDEX],
        complexity_cutoff: float = COMPLEXITY_PERCENTILE,
        k_ks: int = 40,
):
    suggest_df = suggest_from_predictions(
        fom_type, metric, k, top, complexity_type, complexity_cutoff
    )

    dmat_fe = distmat_features(suggest_df["smiles"])
    indices = ks_sampler(dmat_fe, k=k_ks)
    suggest_df = suggest_df.iloc[indices]
    suggest_df: pd.DataFrame
    # get sigma links
    sigma_links = [get_sigma_link(smi) for smi in tqdm(suggest_df["smiles"])]
    suggest_df = suggest_df.assign(url=sigma_links)
    suggest_df.to_csv(
        "suggestor_{}--{}--from+x{}--{}--compcut{:.0f}.csv".format(
            fom_type, metric, top, complexity_type, complexity_cutoff
        ), index=False
    )


def get_sigma_link(smiles: str):
    cid = _smiles_to_cid[smiles]
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/categories/compound/{}/JSON".format(cid)
    response = requests.get(url)
    sigma_url = None
    try:
        if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
            cats = response.json()['SourceCategories']['Categories']
            vendor_cat = None
            for cat in cats:
                if cat["Category"] == 'Chemical Vendors':
                    vendor_cat = cat
            # assert vendor_cat is not None
            sigma_source = None
            for source in vendor_cat['Sources']:
                source_name = source['SourceName']
                if 'sigma' in source_name.lower():
                    sigma_source = source
            sigma_url = sigma_source['SourceRecordURL']
    except:
        pass
    return sigma_url


if __name__ == '__main__':
    make_suggestions(fom_type="fom2", metric="std", k=200, top=True, complexity_type="BertzCT", complexity_cutoff=25)
    make_suggestions(fom_type="fom2", metric="mu-top2%", k=200, top=True, complexity_type="BertzCT",
                     complexity_cutoff=25)
    make_suggestions(fom_type="fom2", metric="mu-top2%", k=200, top=False, complexity_type="BertzCT",
                     complexity_cutoff=25)
