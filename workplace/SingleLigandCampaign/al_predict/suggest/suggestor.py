import glob
import numba
import umap
import numpy as np
import pandas as pd
import hdbscan
from rdkit import Chem
from lsal.schema import Molecule
from lsal.utils import FilePath, json_load, pkl_load, scale_df, SEED
from sklearn.metrics import pairwise_distances
from lsal.tasks.dimred import scatter2d
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import re

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
_available_metric = ('mu-top2%', 'std-top2%mu')
_complexity_descriptors = ["sa_score", "BertzCT"]
_distance_metric = "manhattan"

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
        fom_type: str, metric: str, k: int, top:bool=True,
        complexity_type: str=_complexity_descriptors[0], complexity_cutoff: float = 25
):
    df = _dfs[fom_type][["smiles", metric]]
    df = df.assign(**{complexity_type: [_smiles_data[smi][complexity_type] for smi in df["smiles"]]})

    comp_values = df[complexity_type]
    complexity_cutoff = float(np.percentile(comp_values, complexity_cutoff))

    df:pd.DataFrame
    records = df.to_dict(orient="records")
    records = sorted([r for r in records if r[complexity_type] <= complexity_cutoff], key=lambda x: x[metric], reverse=top)[:k]
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
        fom_type: str=_available_fom[0], metric: str=_available_metric[0], k: int=SUGGEST_K, top: bool = SUGGEST_TOP,
        complexity_type: str = _complexity_descriptors[COMPLEXITY_INDEX], complexity_cutoff: float = COMPLEXITY_PERCENTILE,
        nn=NN, md=MD,
):
    suggest_df = suggest_from_predictions(
        fom_type, metric, k, top, complexity_type, complexity_cutoff
    )

    dmat_fe = distmat_features(suggest_df["smiles"])
    dmat_fp = distmat_fps(suggest_df["smiles"])

    dimred_transformer = umap.UMAP(
        n_neighbors=nn, min_dist=md, metric="precomputed", random_state=SEED)
    dimred_fe = dimred_transformer.fit_transform(dmat_fe)
    dimred_fp = dimred_transformer.fit_transform(dmat_fp)

    hdb = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS, gen_min_span_tree=False)
    hdb.fit(dimred_fe)
    fe_labels = [lab for lab in hdb.labels_]
    hdb.fit(dimred_fp)
    fp_labels = [lab for lab in hdb.labels_]
    return suggest_df, dimred_fp, dimred_fe, fp_labels, fe_labels

make_suggestions()

# fom_type: str = _available_fom[0], metric: str = _available_metric[0], k: int = 50, top: bool = True,
# complexity_type: str = _complexity_descriptors[0], complexity_cutoff: float = 25,
# nn = 5, md = 0.2,
