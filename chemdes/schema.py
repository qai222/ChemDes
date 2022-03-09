import mordred
from monty.json import MSONable

from chemdes.utils import *


class Molecule(MSONable):

    def __init__(self, inchi: str):
        self.inchi = inchi

    @property
    def rdmol(self):
        return Chem.MolFromInchi(self.inchi)

    @property
    def smiles(self) -> str:
        return inchi2smiles(self.inchi)

    @classmethod
    def from_str(cls, s: str, repr_type="inchi"):
        if repr_type.startswith("i"):
            return cls(s)
        elif repr_type.startswith("s"):
            return cls(smiles2inchi(s))
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
