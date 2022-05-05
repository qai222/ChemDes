import logging
import os

import pandas as pd

from lsal.schema import Molecule
from lsal.utils import FilePath, file_exists

_InventoryColumns = ["Name", "IUPAC Name", "InChI", ]
_LigandMoleculeLabelTemplate = "Ligand-{0:0>4}"


def load_raw_ligand_inventory(fn: FilePath) -> list[Molecule]:
    logging.warning("loading ligand inventory file: {}".format(fn))
    assert file_exists(fn)
    _, extension = os.path.splitext(fn)
    if extension == ".csv":
        df = pd.read_csv(fn)
    elif extension == ".xlsx":
        ef = pd.ExcelFile(fn)
        assert len(ef.sheet_names) == 1
        df = ef.parse(ef.sheet_names[0])
    else:
        raise AssertionError("inventory file should be either csv or xlsx")
    assert set(_InventoryColumns).issubset(set(df.columns)), "Inventory DF does not have required columns"
    df = df.dropna(axis=0, how="all", subset=_InventoryColumns)
    molecules = []
    for irow, row in enumerate(df.to_dict("records")):
        name = row["Name"]
        iupac_name = row["IUPAC Name"]
        inchi = row["InChI"]
        label = _LigandMoleculeLabelTemplate.format(irow)
        m = Molecule(identifier=inchi, iupac_name=iupac_name, name=name, label=label,
                     properties={"raw_file": os.path.basename(fn)})
        molecules.append(m)
    logging.warning("# of molecules loaded: {}".format(len(molecules)))
    return molecules


def load_ligand_to_descriptors(fn: FilePath, inventory: list[Molecule]) -> dict[Molecule, dict]:
    logging.warning("loading file: {}".format(fn))
    ligands = []
    des_df = pd.read_csv(fn)
    assert not des_df.isnull().values.any()
    for r in des_df.to_dict("records"):
        inchi = r["InChI"]
        mol = Molecule.select_from_inventory(inchi, inventory, "inchi")
        ligands.append(mol)
    data_df = des_df.select_dtypes('number')
    return dict(zip(ligands, data_df.to_dict(orient="records")))


def ligand_to_ml_input(
        ligand_to_data: dict,
        data_type: str,
        ligand_inventory: list[Molecule], descriptor_csv: FilePath,
):
    ligand_to_des_record = load_ligand_to_descriptors(descriptor_csv, ligand_inventory)
    records = []
    df_ligands = []
    final_cols = set()
    if data_type == "reaction":
        for ligand, reactions in ligand_to_data.items():
            des_record = ligand_to_des_record[ligand]
            for reaction in reactions:
                record = {
                    "ligand_amount": reaction.ligand_solution.concentration * reaction.ligand_solution.volume,
                    "fom": reaction.properties["fom"],
                }
                record.update(des_record)
                if len(final_cols) == 0:
                    final_cols.update(set(record.keys()))
                records.append(record)
                df_ligands.append(ligand)
        df = pd.DataFrame.from_records(records, columns=sorted(final_cols))
        df_X = df[[c for c in df.columns if c != "fom"]]
        df_y = df["fom"]
        logging.warning("ML INPUT:\n df_X: {}\t df_y: {}".format(df_X.shape, df_y.shape))
    elif data_type == "amount":
        for ligand, amounts in ligand_to_data.items():
            des_record = ligand_to_des_record[ligand]
            for amount in amounts:
                record = {
                    "ligand_amount": amount
                }
                record.update(des_record)
                if len(final_cols) == 0:
                    final_cols.update(set(record.keys()))
                records.append(record)
                df_ligands.append(ligand)
        df = pd.DataFrame.from_records(records, columns=sorted(final_cols))
        df_X = df
        df_y = None
        logging.warning("ML INPUT:\n df_X: {}\t df_y: {}".format(df_X.shape, df_y))
    else:
        raise ValueError("`data_type` not understood: {}".format(data_type))
    return df_ligands, df_X, df_y
