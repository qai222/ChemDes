from __future__ import annotations

import abc
import itertools
from copy import deepcopy
from typing import Tuple, List, Iterable, Union

import numpy as np
import pandas as pd
from loguru import logger
from monty.json import MSONable

from lsal.schema.material import Molecule, NanoCrystal
from lsal.utils import msonable_repr, rgetattr, flatten_json

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
                logger.warning("{}: {} is {}!".format(self.__repr__(), k, v))


class ReactionCondition(ReactionInfo):
    def __init__(self, name: str, value: Union[float, int], properties: dict = None):
        super().__init__(properties)
        assert set(name).issuperset({"(", ")"}), "a bracketed unit should present in the name of a condition!"
        self.value = value
        self.name = name


class ReactantSolution(ReactionInfo):
    def __init__(self, solute: Union[NanoCrystal, Molecule], volume: float, concentration: Union[float, None],
                 solvent: Union[Molecule, None],
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

    def as_flat_dict(
            self, exclude_kw=(
                    "@module", "@version", "@class", "properties___features"
            ), sep="___",
    ):
        return {k.rstrip(sep): v for k, v in flatten_json(self.as_dict(), sep=sep).items() if
                all(kw not in k for kw in exclude_kw)}

    def __hash__(self):
        return hash(self.identifier)

    def __gt__(self, other):
        return self.identifier.__gt__(other.identifier)

    def __lt__(self, other):
        return self.identifier.__lt__(other.identifier)

    def __eq__(self, other):
        return self.identifier == other.identifier

    def check(self):
        logger.warning("checking reaction: {}".format(self.identifier))
        for r in self.reactants:
            r.check()
        for c in self.conditions:
            c.check()

    @staticmethod
    def group_reactions(reactions: Iterable[GeneralReaction], field: str):
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


class LXReaction(GeneralReaction):

    def __init__(
            self,
            identifier: str,
            conditions: list[ReactionCondition],
            solvent: ReactantSolution = None,
            nc_solution: ReactantSolution = None,
            ligand_solutions: list[ReactantSolution] = None,
            properties: dict = None,
    ):
        super().__init__(identifier, ligand_solutions + [solvent, nc_solution], conditions, properties)
        self.solvent = solvent
        self.nc_solution = nc_solution
        self.ligand_solutions = ligand_solutions
        assert len(self.ligand_tuple) == len(
            set(self.ligand_tuple)), "one solution for one ligand, " \
                                     "but we have # solutions vs # ligands: {} vs {}".format(
            len(set(self.ligand_tuple)), len(self.ligand_tuple))
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
    def ligand_tuple(self) -> Tuple[Molecule, ...]:
        return tuple(sorted([ls.solute for ls in self.ligand_solutions]))

    @property
    def unique_ligands(self) -> Tuple[Molecule, ...]:
        return tuple(sorted(set(self.ligand_tuple)))


class L1XReaction(LXReaction):
    @property
    def ligand(self):
        try:
            return self.ligand_tuple[0]
        except IndexError:
            return None

    @property
    def ligand_solution(self):
        try:
            return self.ligand_solutions[0]
        except IndexError:
            return None


class LXReactionCollection(MSONable, abc.ABC):
    # TODO the reactions in a collection should have something in common (e.g. solvent/mixing conditions)
    def __init__(self, reactions, properties: dict = None):
        self.reactions = reactions
        if properties is None:
            properties = dict()
        self.properties = properties

    def __len__(self):
        return len(self.reactions)

    @property
    def identifiers(self):
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

    def get_reference_reactions(self, reaction: LXReaction):
        # given a reaction return its corresponding reference reactions
        # i.e. same identifier
        refs = []
        for ref_r in self.ref_reactions:
            if ref_r.identifier.split("@@")[0] == reaction.identifier.split("@@")[0]:
                refs.append(ref_r)
        return refs

    @property
    def real_reactions(self):
        reactions = []
        for r in self.reactions:
            if r.is_reaction_real:
                reactions.append(r)
            else:
                continue
        return reactions

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def ligand_to_reactions_mapping(self, limit_to=None):
        pass


class L1XReactionCollection(LXReactionCollection):
    # TODO the reactions in a collection should have something in common (e.g. solvent/mixing conditions)
    def __init__(self, reactions: List[L1XReaction], properties: dict = None):
        super().__init__(reactions, properties)

    @property
    def ligand_amount_range(self):
        amounts = []
        amount_unit = []
        for r in self.real_reactions:
            amounts.append(r.ligand_solution.amount)
            amount_unit.append(r.ligand_solution.amount_unit)
        assert len(set(amount_unit)) == 1
        return min(amounts), max(amounts), amount_unit[0]

    def amount_lin_space(self, n_preds):
        amin, amax, _ = self.ligand_amount_range
        return np.linspace(amin, amax, n_preds)

    def amount_geo_space(self, n_preds):
        amin, amax, _ = self.ligand_amount_range
        return np.geomspace(amin, amax, n_preds)

    @classmethod
    def subset_by_ligands(cls, campaign_reactions: L1XReactionCollection, allowed_ligands: List[Molecule]):
        """ select real reactions by allowed ligands """
        reactions = [r for r in campaign_reactions.real_reactions if r.ligand in allowed_ligands]
        return cls(reactions, properties=deepcopy(campaign_reactions.properties))

    @property
    def ligands(self) -> List[Molecule]:
        return [r.ligand for r in self.real_reactions]

    @property
    def unique_ligands(self) -> list:
        return sorted(set(self.ligands))

    def __repr__(self):
        s = "{}\n".format(self.__class__.__name__)
        s += "\t# of reactions: {}\n".format(len(self.reactions))
        s += f"\t# of real reactions: {len(self.real_reactions)}\n"
        s += f"\t# of blank reactions: {len([r for r in self.reactions if r.is_reaction_blank_reference])}\n"
        s += f"\t# of ref reactions: {len([r for r in self.reactions if r.is_reaction_nc_reference])}\n"
        s += "\t# of ligands: {}\n".format(len(self.unique_ligands))
        for lig, reactions in self.ligand_to_reactions_mapping().items():
            lig: Molecule
            s += f"\t ligand: {lig.label} \t {len(reactions)} \t {lig.smiles}\n "
        return s

    def ligand_to_reactions_mapping(self, limit_to: Iterable[Molecule] = None) -> dict[Molecule, list[L1XReaction]]:
        reactions = self.real_reactions
        ligands, grouped_reactions = L1XReaction.group_reactions(reactions, field="ligand")
        ligand_to_reactions = dict(zip(ligands, grouped_reactions))
        if limit_to is None:
            limit_to = ligands
        return {c: ligand_to_reactions[c] for c in limit_to}

    def l1_input(self, fom_def: str, fill_nan=True) -> Tuple[List[Molecule], pd.DataFrame, pd.DataFrame]:
        assert all(lig.is_featurized for lig in self.ligands)
        records = []
        ligands = []
        final_cols = set()
        # random state of model fitting depends on the order of input rows
        # here ligands are sorted in `self.ligand_to_reactions_mapping()`
        for lig, reactions in self.ligand_to_reactions_mapping().items():
            records_of_this_ligand = []
            for r in reactions:
                record = {"ligand_amount": r.ligand_solution.amount, "FigureOfMerit": r.properties[fom_def], }
                record.update(lig.properties['features'])
                records_of_this_ligand.append(record)
                ligands.append(lig)
                if len(final_cols) == 0:
                    final_cols.update(set(record.keys()))
            records += records_of_this_ligand
        df = pd.DataFrame.from_records(records, columns=sorted(final_cols))
        df_x = df[[c for c in df.columns if c != "FigureOfMerit"]]
        df_y = df["FigureOfMerit"]
        if fill_nan:
            df_y.fillna(0, inplace=True)
        logger.info("ML INPUT: df_X: {}\t df_y: {}".format(df_x.shape, df_y.shape))
        return ligands, df_x, df_y

    def as_dataframe(self):
        rs = []
        for r in self.real_reactions:
            fr = r.as_flat_dict()
            rs.append(fr)
        return pd.DataFrame.from_records(rs)


def assign_reaction_results(reactions, peak_data: dict[str, dict]):
    assert len(peak_data) == len(reactions)
    for r in reactions:
        data = peak_data[r.identifier]
        r.properties.update(data)
