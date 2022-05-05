from __future__ import annotations

import abc

import pandas as pd
from monty.json import MSONable

from lsal.utils import inchi2smiles, MolFromInchi, FilePath


class Material(MSONable, abc.ABC):

    def __init__(self, identifier: str, identifier_type: str, properties=None, label: str = None):
        self.identifier = identifier
        self.identifier_type = identifier_type
        if properties is None:
            properties = dict()
        self.properties = properties
        self.label = label

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

    @staticmethod
    def select_from_inventory(value, inventory: list[Material], field: str) -> Material:
        for m in inventory:
            if getattr(m, field) == value:
                return m
        raise ValueError("not found in the inventory: {} == {}".format(field, value))


class NanoCrystal(Material):

    def __init__(self, identifier: str, properties: dict = None, label: str = None):
        super().__init__(identifier, "formula", properties, label)
        self.formula = self.identifier


class SolventMolecule(Material):
    def __init__(self, identifier: str, properties: dict = None, label: str = None):
        super().__init__(identifier, "solvent_name", properties, label)
        self.solvent_name = self.identifier


class Molecule(Material):

    def __init__(self, identifier: str, iupac_name: str = None, name: str = None, smiles: str = None,
                 label: str = None, properties=None):
        super().__init__(identifier, "inchi", properties, label)
        self.inchi = self.identifier
        self.iupac_name = iupac_name
        self.name = name
        self.label = label
        if smiles is None:
            self.smiles = inchi2smiles(self.inchi)
        else:
            self.smiles = smiles

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
                r = {k: v for k, v in m.as_dict().items() if not k.startswith("@") and k != "properties"}
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
