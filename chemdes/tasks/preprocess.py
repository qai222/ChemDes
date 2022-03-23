import logging
from pathlib import Path
from typing import Union

from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

from chemdes.one_ligand import Molecule
from chemdes.schema import pd
from chemdes.utils import json_load


def load_molecular_descriptors(fn: Union[Path, str], warning=False):
    if warning:
        logging.warning("loading file: {}".format(fn))
    molecules = []
    df = pd.read_csv(fn)
    assert not df.isnull().values.any()
    for r in df.to_dict("records"):
        inchi = r["InChI"]
        iupac_name = r["IUPAC Name"]
        mol = Molecule.from_str(inchi, "i", iupac_name)
        molecules.append(mol)
    df = df[[c for c in df.columns if c not in ["InChI", "IUPAC Name"]]]
    return molecules, df


def preprocess_descriptor_df(data_df, scale=False, vthreshould=False):
    data_df = data_df.select_dtypes('number')
    x = data_df.values  # returns a numpy array
    if scale:
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        data_df = pd.DataFrame(x_scaled, columns=data_df.columns, index=data_df.index)
    if vthreshould:
        sel = VarianceThreshold(threshold=0.01)
        sel_var = sel.fit_transform(data_df)
        data_df = data_df[data_df.columns[sel.get_support(indices=True)]]
    return data_df


# def unlabelled_ligands_to_df_Xy(ligands: list[Molecule], ligand_to_des_record, cmax: float, cmin: float, nfake=1000,
#                                 randomc=False) -> tuple[
#     pd.DataFrame, pd.DataFrame]:
#     """ ligand feature + fake concentrations """
#     final_cols = set()
#     unlabelled_records = []
#     for i, ligand in enumerate(ligands):
#         if randomc:
#             rs = np.random.RandomState(SEED + i)
#             fake_amounts = rs.uniform(cmin, cmax, nfake)
#         else:
#             fake_amounts = np.linspace(cmin, cmax, nfake)
#         des_record = ligand_to_des_record[ligand]
#         for fa in fake_amounts:
#             record = {"ligand_inchi": ligand.inchi, "ligand_iupac_name": ligand.iupac_name, "ligand_amount": fa}
#             record.update(des_record)
#             record["fom"] = np.nan
#             if len(final_cols) == 0:
#                 final_cols.update(set(record.keys()))
#             unlabelled_records.append(record)
#     df = pd.DataFrame.from_records(unlabelled_records, columns=sorted(final_cols))
#     df_X = df[[c for c in df.columns if c != "fom"]]
#     df_y = df["fom"]
#     return df_X, df_y


def reactions_to_df_Xy(ligand_to_categorized_reactions: dict, ligand_to_des_record) -> tuple[
    list[Molecule], pd.DataFrame, pd.DataFrame]:
    records = []
    final_cols = set()
    ligands = []
    for ligand in ligand_to_categorized_reactions:
        ligands.append(ligand)
        des_record = ligand_to_des_record[ligand]
        for reaction in ligand_to_categorized_reactions[ligand][0]:
            record = {"ligand_inchi": ligand.inchi, "ligand_iupac_name": ligand.iupac_name,
                      "ligand_amount": reaction.ligand.concentration * reaction.ligand.volume}
            record.update(des_record)
            record["fom"] = reaction.properties["fom"]
            if len(final_cols) == 0:
                final_cols.update(set(record.keys()))
            records.append(record)
    df = pd.DataFrame.from_records(records, columns=sorted(final_cols))
    df_X = df[[c for c in df.columns if c != "fom"]]
    df_y = df["fom"]
    return ligands, df_X, df_y


def load_ligand_to_des_record(mdes_csv: Union[Path, str], ):
    Ligands, DesDf = load_molecular_descriptors(mdes_csv,
                                                warning=True)
    DesDf = preprocess_descriptor_df(DesDf, scale=False, vthreshould=False)
    LigandToDesRecord = dict(zip(Ligands, DesDf.to_dict(orient="records")))
    return LigandToDesRecord


def load_descriptors_and_fom(mdes_csv: Union[Path, str], reactions_json: Union[Path, str]):
    LigandToDesRecord = load_ligand_to_des_record(mdes_csv)

    ligand_to_categorized_reactions = json_load(reactions_json)
    ligand_to_categorized_reactions = {Molecule.from_repr(k): v for k, v in ligand_to_categorized_reactions.items()}

    labelled_ligands, df_X_labelled, df_y_labelled = reactions_to_df_Xy(ligand_to_categorized_reactions,
                                                                        LigandToDesRecord)
    return labelled_ligands, df_X_labelled, df_y_labelled
