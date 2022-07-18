from __future__ import annotations

import abc
import itertools
import logging
from copy import deepcopy
from typing import Tuple

from monty.json import MSONable

from lsal.schema.material import Molecule, NanoCrystal
from lsal.utils import msonable_repr, rgetattr

_Precision = 5
_EPS = 1 ** -_Precision


class ReactionInfo(MSONable, abc.ABC):

    def __init__(self, properties: dict = None):
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


class ReactantSolution(ReactionInfo):
    def __init__(self, solute: NanoCrystal or Molecule, volume: float, concentration: float or None, solvent: Molecule,
                 volume_unit: str, concentration_unit: str or None, properties: dict = None, ):
        super().__init__(properties)
        self.solute = solute
        self.solvent = solvent
        self.concentration = concentration  # concentration can be None for e.g. nanocrystal
        self.volume = volume
        if self.concentration == 0:
            assert self.solvent == self.solute, "zero concentration but different solvent and solute: {} - {}".format(
                self.solvent, self.solute)
        self.volume_unit = volume_unit
        self.concentration_unit = concentration_unit

    @property
    def amount(self) -> float:
        return self.concentration * self.volume

    @property
    def amount_unit(self) -> str:
        return "{}*{}".format(self.volume_unit, self.concentration_unit)

    @property
    def is_solvent(self) -> bool:
        return self.concentration == 0


class GeneralReaction(MSONable, abc.ABC):
    def __init__(self, identifier: str, reactants: list[ReactantSolution], conditions: list[ReactionCondition],
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


class LigandExchangeReaction(GeneralReaction):

    def __init__(
            self,
            identifier: str,
            conditions: [ReactionCondition],
            solvent: ReactantSolution = None,
            nc_solution: ReactantSolution = None,
            ligand_solutions: list[ReactantSolution] = None,
            properties: dict = None,
    ):
        super().__init__(identifier, ligand_solutions + [solvent, nc_solution], conditions, properties)
        self.solvent = solvent
        self.nc_solution = nc_solution
        self.ligand_solutions = ligand_solutions
        assert len(self.ligands) == len(
            set(self.ligands)), "one solution for one ligand, but we have # solutions vs # ligands: {} vs {}".format(
            len(set(self.ligands)), len(self.ligands))
        assert self.solvent.is_solvent, "the solvent given is not really a solvent: {}".format(self.solvent)

    @property
    def is_reaction_nc_reference(self) -> bool:
        """ whether the reaction is a reference reaction in which only NC solution and solvent were added """
        nc_good = self.nc_solution is not None and self.nc_solution.volume > _EPS
        solvent_good = self.solvent is not None and self.solvent.volume > _EPS
        no_ligand = len(self.ligand_solutions) == 0 or all(
            ls is None or ls.volume < _EPS for ls in self.ligand_solutions)
        return nc_good and solvent_good and no_ligand

    @property
    def is_reaction_blank_reference(self) -> bool:
        """ whether the reaction is a reference reaction in which only solvent was added """
        no_nc = self.nc_solution is None or self.nc_solution.volume < _EPS
        solvent_good = self.solvent is not None and self.solvent.volume > _EPS
        no_ligand = len(self.ligand_solutions) == 0 or all(
            ls is None or ls.volume < _EPS for ls in self.ligand_solutions)
        return no_nc and no_ligand and solvent_good

    @property
    def is_reaction_real(self) -> bool:
        """ whether the reaction is neither a blank nor a ref """
        return not self.is_reaction_blank_reference and not self.is_reaction_nc_reference

    @property
    def ligands(self):
        return tuple(sorted([ls.solute for ls in self.ligand_solutions]))

    @property
    def unique_ligands(self):
        return tuple(sorted(set(self.ligands)))

    @staticmethod
    def group_reactions(reactions: list[GeneralReaction], field: str):
        """ group reactions by a field, the field can be dot-structured, e.g. "nc_solution.solute" """
        groups = []
        unique_keys = []

        def keyfunc(x):
            return rgetattr(x, field)

        rs = sorted(reactions, key=keyfunc)
        for k, g in itertools.groupby(rs, key=keyfunc):
            groups.append(list(g))
            unique_keys.append(k)
        return unique_keys, groups


class ReactionCollection(MSONable):
    # TODO the reactions in a collection should have something in common (e.g. solvent/mixing conditions)
    def __init__(self, reactions: list[LigandExchangeReaction], properties: dict = None):
        self.reactions = reactions
        if properties is None:
            properties = dict()
        self.properties = properties

    @property
    def identifiers(self) -> Tuple[str]:
        return tuple([r.identifier for r in self.reactions])

    @property
    def ref_reactions(self):
        reactions = []
        for r in self.reactions:
            if r.is_reaction_nc_reference:
                reactions.append(r)
            else:
                continue
        return reactions

    def get_reference_reactions(self, reaction: LigandExchangeReaction):
        # map ref to each reaction
        refs = []
        for ref_r in self.ref_reactions:
            if ref_r.identifier.split("@@")[0] == reaction.identifier.split("@@")[0]:
                refs.append(ref_r)
        return refs

    def ligand_amounts(self, ligand: Molecule) -> list[float]:
        # only works for single ligand system
        reactions = self.get_lcomb_to_reactions()[(ligand,)]
        reactions: list[LigandExchangeReaction]
        assert len(reactions) > 0
        return [r.ligand_solutions[0].amount for r in reactions]

    @property
    def ligand_amount_range(self):
        amounts = []
        amount_unit = []
        for r in self.real_reactions:
            for ls in r.ligand_solutions:
                amounts.append(ls.amount)
                amount_unit.append(ls.amount_unit)
        assert len(set(amount_unit)) == 1
        return min(amounts), max(amounts), amount_unit[0]

    @classmethod
    def subset_by_lcombs(cls, campaign_reactions: ReactionCollection, lc_subset):
        reactions = [r for r in campaign_reactions.real_reactions if r.ligands in lc_subset]
        return cls(reactions, properties=deepcopy(campaign_reactions.properties))

    @property
    def real_reactions(self):
        reactions = []
        for r in self.reactions:
            if r.is_reaction_real:
                reactions.append(r)
            else:
                continue
        return reactions

    @property
    def unique_lcombs(self):
        lcombs = set()
        for r in self.real_reactions:
            lcombs.add(r.unique_ligands)
        return sorted(lcombs)

    def __repr__(self):
        s = "{}\n".format(self.__class__.__name__)
        s += "\t# of reactions: {}\n".format(len(self.reactions))
        s += "\t# of ligands/ligand-combinations: {}\n".format(len(self.get_lcomb_to_reactions()))
        return s

    def get_lcomb_to_reactions(self, limit_to=None):
        reactions = self.real_reactions

        lcombs, grouped_reactions = LigandExchangeReaction.group_reactions(reactions, field="unique_ligands")
        lcombs_to_reactions = dict(zip(lcombs, grouped_reactions))
        if limit_to is None:
            return lcombs_to_reactions
        else:
            return {c: lcombs_to_reactions[c] for c in limit_to}

    @staticmethod
    def assign_reaction_results(reactions: list[LigandExchangeReaction], peak_data: dict[str, dict]):
        assert len(peak_data) == len(reactions)
        for r in reactions:
            data = peak_data[r.identifier]
            r.properties.update(data)
