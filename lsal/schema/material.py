from __future__ import annotations

import abc
from collections import OrderedDict
from copy import deepcopy
from typing import Union

import pandas as pd
from loguru import logger
from monty.json import MSONable

from lsal.utils import inchi2smiles, MolFromInchi, FilePath, file_exists, get_extension, get_basename, np, flatten_json


class Material(MSONable, abc.ABC):

    def __init__(
            self, identifier: str, identifier_type: str, properties: OrderedDict = None, mat_type: str = None
    ):
        """
        :param identifier: unique identifier for this material
        :param identifier_type: what does this identifier represent?
        :param properties: a dictionary
        :param mat_type: materials type
        """
        self.identifier = identifier
        self.identifier_type = identifier_type
        if properties is None:
            properties = OrderedDict()
        self.properties = properties
        self.mat_type = mat_type.upper()

    @property
    @abc.abstractmethod
    def label(self):
        pass

    def __gt__(self, other):
        return self.__repr__().__gt__(other.__repr__())

    def __lt__(self, other):
        return self.__repr__().__lt__(other.__repr__())

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        return self.identifier == other.identifier

    def __repr__(self):
        return "{} - {}: {}".format(self.__class__.__name__, self.identifier_type, self.identifier)

    @property
    def is_featurized(self):
        try:
            return len(self.properties['features']) > 0
        except KeyError:
            return False


class NanoCrystal(Material):

    def __init__(self, identifier: str, properties: OrderedDict = None):
        super().__init__(identifier, "batch_number", properties, "NC")
        self.batch_number = self.identifier

    @property
    def label(self):
        return "{}-{}".format(self.mat_type, self.identifier)


class Molecule(Material):

    def __init__(self, identifier: str, iupac_name: str = None, name: str = None, smiles: str = None,
                 int_label: int = None, mol_type: str = None, properties=None):
        super().__init__(identifier, "inchi", properties, mol_type)
        self.mol_type = mol_type
        self.inchi = self.identifier
        self.iupac_name = iupac_name
        self.name = name
        self.int_label = int_label
        if smiles is None:
            self.smiles = inchi2smiles(self.inchi)
        else:
            self.smiles = smiles

    def __repr__(self):
        return "{} - {}: {}".format(self.__class__.__name__, self.label, self.name)

    @property
    def label(self):
        label_template = "-{0:08d}"
        return self.mat_type + label_template.format(self.int_label)

    @property
    def rdmol(self):
        return MolFromInchi(self.inchi)

    @staticmethod
    def write_molecules(mols: list[Molecule], fn: FilePath, output="smi"):
        if output == "smi":
            s = "\n".join([m.smiles for m in mols])
            with open(fn, "w") as f:
                f.write(s)
            return s
        elif output == "csv":
            records = []
            nested_sep = "___"
            for m in mols:
                r = {k: v for k, v in m.as_record().items() if not k.startswith("@")}
                r = flatten_json(r, sep=nested_sep)
                r = {k.rstrip(nested_sep): v for k, v in r.items()}
                records.append(r)
            df = pd.DataFrame.from_records(records)
            df.set_index(df.pop('label'), inplace=True)
            df.reset_index(inplace=True)
            df.to_csv(fn, index=False)
            return df
        else:
            raise ValueError("Unknown output extension: {}".format(output))

    def as_record(self) -> dict:
        d = self.as_dict()
        d["label"] = self.label
        return d

    @staticmethod
    def select_from_list(value, mol_list: list[Molecule], field: str) -> Molecule:
        for m in mol_list:
            if getattr(m, field) == value:
                return m
        raise ValueError("not found in the inventory: {} == {}".format(field, value))

    @staticmethod
    def l1_input(ligands: list[Molecule], amounts: Union[list[float], np.ndarray] = None):
        """
        generate input for model predictions
        """
        assert all(lig.is_featurized for lig in ligands)
        records = []
        final_cols = set()
        ligand_col = []

        if amounts is None:
            for lig in ligands:
                record = deepcopy(lig.properties['features'])
                records.append(record)
                ligand_col.append(lig)
                if len(final_cols) == 0:
                    final_cols.update(set(record.keys()))
        else:
            for lig in ligands:
                for amount in amounts:
                    record = deepcopy(lig.properties['features'])
                    record.update({'ligand_amount': amount})
                    records.append(record)
                    ligand_col.append(lig)
                    if len(final_cols) == 0:
                        final_cols.update(set(record.keys()))
        df = pd.DataFrame.from_records(records, columns=sorted(final_cols))
        return ligand_col, df


def featurize_molecules(molecules: list[Molecule], feature_dataframe: pd.DataFrame):
    assert feature_dataframe.shape[0] == len(molecules)
    assert not feature_dataframe.isnull().any().any()
    for m, d in zip(molecules, feature_dataframe.to_dict(orient='records')):
        m.properties['features'] = OrderedDict(d)


def load_molecules(
        fn: FilePath, col_to_mol_kw: dict[str, str], mol_type: str = 'LIGAND',
) -> list[Molecule]:
    mol_type = mol_type.upper()

    logger.info(f"LOADING: {mol_type} from {fn}")
    assert file_exists(fn)
    extension = get_extension(fn)
    if extension == "csv":
        df = pd.read_csv(fn)
    elif extension == "xlsx":
        ef = pd.ExcelFile(fn)
        assert len(ef.sheet_names) == 1, "there should be only one sheet in the xlsx file"
        df = ef.parse(ef.sheet_names[0])
    else:
        raise ValueError(f"extension not understood: {extension}")

    required_columns = col_to_mol_kw.keys()
    assert set(required_columns).issubset(set(df.columns)), f"csv does not have required columns: {required_columns}"

    df = df[required_columns]
    df = df.dropna(axis=0, how="all")

    assign_label = 'label' not in df.columns
    if assign_label:
        logger.info(f'we WILL assign labels based on row index and mol_type=={mol_type}')

    molecules = []
    mol_kws = ['identifier', 'iupac_name', 'name']
    for irow, row in enumerate(df.to_dict("records")):

        if assign_label:
            int_label = irow
        else:
            label = row['label']
            mol_type, int_label = label.split('-')
            int_label = int(int_label)

        mol_kwargs = dict(
            int_label=int_label,
            mol_type=mol_type,
            properties=OrderedDict({"load_from": get_basename(fn)})
        )

        for colname, value in row.items():
            # TODO parse nested properties
            try:
                mol_kw = col_to_mol_kw[colname]
                assert mol_kw in mol_kws
                mol_kwargs[mol_kw] = value
            except (AssertionError, KeyError) as e:
                pass
        m = Molecule(**mol_kwargs)
        molecules.append(m)
    return molecules


def load_featurized_molecules(
        inv_csv: FilePath,
        des_csv: FilePath,
        mol_type: str,
        col_to_mol_kw: dict[str, str] = None,
) -> list[Molecule]:
    # load inv csv
    molecules = load_molecules(inv_csv, col_to_mol_kw, mol_type)
    des_df = pd.read_csv(des_csv)
    assert des_df.shape[0] == len(molecules)
    assert not des_df.isnull().values.any()
    featurize_molecules(molecules, des_df)
    return molecules
