from __future__ import annotations

import abc
from collections import OrderedDict

import pandas as pd
from monty.json import MSONable

from lsal.utils import inchi2smiles, MolFromInchi, FilePath


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
    def label(self): pass

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
        label_template = "-{0:0>4}"
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
            for m in mols:
                r = {k: v for k, v in m.as_record().items() if not k.startswith("@") and k != "properties"}
                for k in m.properties:
                    r["properties__{}".format(k)] = m.properties[k]
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
    def select_from_inventory(value, inventory: list[Molecule], field: str) -> Molecule:
        for m in inventory:
            if getattr(m, field) == value:
                return m
        raise ValueError("not found in the inventory: {} == {}".format(field, value))


def featurize_molecules(molecules: list[Molecule], feature_dataframe: pd.DataFrame):
    assert feature_dataframe.shape[0] == len(molecules)
    assert not feature_dataframe.isnull().any().any()
    for m, d in zip(molecules, feature_dataframe.to_dict(orient='records')):
        m.properties['features'] = OrderedDict(d)
