from __future__ import annotations

import abc
import itertools
import logging

from monty.json import MSONable

from lsal.schema.material import Material
from lsal.utils import msonable_repr
from lsal.utils import rgetattr

_Precision = 5


class ReactionInfo(MSONable, abc.ABC):

    def __init__(self, properties=None):
        if properties is None:
            properties = dict()
        self.properties = properties

    def __repr__(self):
        return msonable_repr(self, precision=_Precision)

    def __hash__(self):
        return hash(self.__repr__())

    def __gt__(self, other):
        return self.__repr__().__gt__(other.__repr__())

    def __lt__(self, other):
        return self.__repr__().__lt__(other.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def check(self):
        for k, v in self.as_dict().items():
            if k.startswith("@"):
                continue
            if v is None:
                logging.warning("{}: {} is {}!".format(self.__repr__(), k, v))


class ReactionCondition(ReactionInfo):
    def __init__(self, name: str, value: float or int, properties: dict = None):
        super().__init__(properties)
        assert set(name).issuperset({"(", ")"}), "a bracketed unit should present in the name of a condition!"
        self.value = value
        self.name = name


class Reactant(ReactionInfo):
    def __init__(self, material: Material, properties: dict = None):
        super().__init__(properties)
        self.material = material


class ReactantSolvent(Reactant):
    def __init__(self, material: Material, volume: float, volume_unit: str = "ul", properties: dict = None):
        super().__init__(material, properties)
        self.volume_unit = volume_unit
        self.volume = volume


class ReactantSolution(Reactant):
    def __init__(self, solute: Material,
                 volume: float, concentration: float or None,
                 solvent: Material or None,
                 properties: dict = None, volume_unit: str = "ul", concentration_unit: str or None = "M"):
        super().__init__(solute, properties)
        self.solute = solute
        self.volume_unit = volume_unit
        self.concentration_unit = concentration_unit
        self.solvent = solvent
        self.concentration = concentration
        self.volume = volume


class GeneralReaction(MSONable, abc.ABC):
    def __init__(self, identifier: str, reactants: list[Reactant], conditions: list[ReactionCondition],
                 properties: dict = None):
        if properties is None:
            properties = dict()
        self.properties = properties
        self.reactants = reactants
        self.conditions = conditions
        self.identifier = identifier

    def __repr__(self):
        s = "Reaction: {}\n".format(self.identifier)
        for k, v in self.properties.items():
            s += "\t Property: {} = {}\n".format(k, v)
        for reactant in self.reactants:
            s += "\t Reactant: {}\n".format(reactant.__repr__())
        for condition in self.conditions:
            s += "\t Condition: {}\n".format(condition.__repr__())
        return s

    def __hash__(self):
        return hash(self.identifier)

    def __gt__(self, other):
        return self.identifier.__gt__(other.identifier)

    def __lt__(self, other):
        return self.identifier.__lt__(other.identifier)

    def __eq__(self, other):
        return self.identifier == other.identifier

    def check(self):
        logging.warning("checking reaction: {}".format(self.identifier))
        for r in self.reactants:
            r.check()
        for c in self.conditions:
            c.check()

    @staticmethod
    def group_reactions(reactions: list[GeneralReaction], field: str):
        """ group reactions by a field, the field can be dot-structured, e.g. "ligand_solution.solute" """
        groups = []
        unique_keys = []
        keyfunc = lambda x: rgetattr(x, field)
        rs = sorted(reactions, key=keyfunc)
        for k, g in itertools.groupby(rs, key=keyfunc):
            groups.append(list(g))
            unique_keys.append(k)
        return unique_keys, groups
