import os

import mordred
from monty.json import MSONable

from chemdes.utils import *


class Molecule(MSONable):

    def __init__(self, inchi: str, iupac_name: str = "unknwon"):
        self.inchi = inchi
        self.iupac_name = iupac_name

    @property
    def rdmol(self):
        return Chem.MolFromInchi(self.inchi)

    @property
    def smiles(self) -> str:
        return inchi2smiles(self.inchi)

    @classmethod
    def from_str(cls, s: str, repr_type="inchi", iupac_name="unknown"):
        if repr_type.startswith("i"):
            return cls(s, iupac_name=iupac_name)
        elif repr_type.startswith("s"):
            return cls(smiles2inchi(s), iupac_name=iupac_name)
        else:
            raise NotImplementedError("`repr_type` not implemented: {}".format(repr_type))

    def __repr__(self):
        return "{}: {}".format(self.__class__.__name__, self.inchi)

    def __hash__(self):
        return hash(self.inchi)

    def __eq__(self, other):
        return self.inchi == other.inchi

    @staticmethod
    def write_smi(mols, fn):
        with open(fn, "w") as f:
            f.write("\n".join([m.smiles for m in mols]))


class Descriptor(MSONable):

    def __init__(self, name: str, source: str, description: str = None, parameters: dict = None):
        self.name = name
        self.source = source
        if description is None:
            description = name
        self.description = description
        if parameters is None:
            parameters = dict()
        self.parameters = parameters

    def __repr__(self):
        return "{} -- {}".format(self.source, self.name)

    def __hash__(self):
        # TODO also hash params
        return hash(self.name + self.source)

    @classmethod
    def from_mordred_descriptor(cls, des: mordred.Descriptor):
        params = des.get_parameter_dict()

        for k, v in params.items():
            if any(isinstance(v, t) for t in (float, str, int)):
                continue
            elif v is None:
                continue
            else:
                params[k] = v.description()

        name = str(des)
        decription = des.__doc__
        return cls(name, "MORDRED-{}".format(mordred.__version__), decription, params)


def load_inventory(fn: typing.Union[pathlib.Path, str], to_mols=True):
    assert os.path.isfile(fn)
    _, extension = os.path.splitext(fn)
    if extension == ".csv":
        df = pd.read_csv(fn)
    elif extension == ".xlsx":
        df = pd.read_excel(fn)
    else:
        raise AssertionError("inventory file should be either csv or xlsx")
    assert "InChI" in df.columns, "InChI must be specified in the inventory"
    df = df.dropna(axis=0, how="all", subset=["InChI"])
    if to_mols:
        mols = []
        for row in df.to_dict("records"):
            inchi = row["InChI"]
            try:
                name = row["IUPAC Name"]
            except KeyError:
                name = "unknown"
            m = Molecule.from_str(inchi, "inchi", iupac_name=name)
            mols.append(m)
        return mols
    else:
        return df
