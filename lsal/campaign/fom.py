import re
from typing import Any, Union

import numpy as np
import pandas as pd

from lsal.campaign.loader import ReactionCollection
from lsal.schema import Molecule, LigandExchangeReaction
from lsal.utils import is_close_relative

"""
expt properties
od: `*_PL_OD390`
plsum: `*_PL_sum`

possible figures of merit
1. `*_PL_sum/OD390`
2. `*_PL_sum/OD390 / mean(*_PL_sum/OD390)` of references (i.e. `*_PL_FOM`)
3. `*_PL_sum/OD390 - mean(*_PL_sum/OD390)` of references
4. `*_PL_sum/OD390 / first(*_PL_sum/OD390)` reaction with the lowest nonzero ligand concentration
5. `*_PL_sum/OD390 - first(*_PL_sum/OD390)` reaction with the lowest nonzero ligand concentration
"""


class FomCalculator:
    def __init__(self, reaction_collection: ReactionCollection):
        self.reaction_collection = reaction_collection
        self.ligand_to_reactions = {k[0]: v for k, v in reaction_collection.get_lcomb_to_reactions().items()}

    def get_average_ref(self, r: LigandExchangeReaction, property_name: str):
        return PropertyGetter.get_reference_value(r, self.reaction_collection, property_name, average=True)

    def get_internal_ref(self, r: LigandExchangeReaction, property_name: str):
        reactions_same_ligand = self.ligand_to_reactions[r.ligands[0]]  # assuming single ligand
        reactions_same_ligand: list[LigandExchangeReaction]
        amount_and_fom = [(rr.ligand_solutions[0].amount, PropertyGetter.get_property_value(rr, property_name)) for rr
                          in
                          reactions_same_ligand]
        ref_fom = sorted(amount_and_fom, key=lambda x: x[0])[0][1]
        return ref_fom

    def fom1(self, r: LigandExchangeReaction) -> float:
        fom = PropertyGetter.get_property_value(r, "pPLQY")
        return fom

    def fom2(self, r: LigandExchangeReaction) -> float:
        fom = PropertyGetter.get_property_value(r, "pPLQY")
        return fom / self.get_average_ref(r, "pPLQY")

    def fom3(self, r: LigandExchangeReaction) -> float:
        fom = PropertyGetter.get_property_value(r, "pPLQY")
        return fom - self.get_average_ref(r, "pPLQY")

    def fom4(self, r: LigandExchangeReaction) -> float:
        fom = PropertyGetter.get_property_value(r, "pPLQY")
        ref_fom = self.get_internal_ref(r, "pPLQY")
        return fom / ref_fom

    def fom5(self, r: LigandExchangeReaction) -> float:
        fom = PropertyGetter.get_property_value(r, "pPLQY")
        ref_fom = self.get_internal_ref(r, "pPLQY")
        return fom - ref_fom

    @property
    def fom_function_names(self):
        function_names = []
        for attr in dir(self):
            if re.match("fom\d", attr):
                function_names.append(attr)
        return function_names

    def update_foms(self) -> ReactionCollection:
        for r in self.reaction_collection.reactions:
            for fname in self.fom_function_names:
                fom_func = getattr(self, fname)
                if r.is_reaction_real:
                    fom = fom_func(r)
                else:
                    fom = np.nan
                r.properties[fname] = fom
        return self.reaction_collection


class PropertyGetter:
    NameToSuffix = {
        "OD": "_PL_OD390",
        "PLSUM": "_PL_sum",
        "pPLQY": "_PL_sum/OD390",
        "fom1": "fom1",
        "fom2": "fom2",
        "fom3": "fom3",
        "fom4": "fom4",
        "fom5": "fom5",
    }

    @staticmethod
    def get_property_value(r, property_name: str):
        assert property_name in PropertyGetter.NameToSuffix
        suffix = PropertyGetter.NameToSuffix[property_name]
        value = PropertyGetter._get_reaction_property(r, suffix)
        if property_name == "pPLQY":
            value2 = PropertyGetter._get_reaction_property(r, "_PL_sum") / PropertyGetter._get_reaction_property(r,
                                                                                                                 "_PL_OD390")
            assert is_close_relative(value2, value, 1e-5) or pd.isna(value) or pd.isna(value2)
        return value

    @staticmethod
    def _get_reaction_property(r: LigandExchangeReaction, property_suffix: str) -> float:
        possible_properties = [k for k in r.properties if k.strip("'").endswith(property_suffix)]
        assert len(possible_properties) == 1
        k = possible_properties[0]
        v = r.properties[k]
        try:
            assert isinstance(v, float)
        except AssertionError:
            v = np.nan
        return v

    @staticmethod
    def get_reference_value(
            r: LigandExchangeReaction, reaction_collection: ReactionCollection, property_name: str, average=True
    ) -> Union[float, list[float]]:
        ref_values = []
        for ref_reaction in reaction_collection.get_reference_reactions(r):
            ref_value = PropertyGetter.get_property_value(ref_reaction, property_name)
            ref_values.append(ref_value)
        if average:
            return float(np.mean(ref_values))
        else:
            return ref_values

    @staticmethod
    def get_amount_property_data(
            reaction_collection: ReactionCollection, property_name: str
    ) -> dict[Molecule, dict[str, Any]]:
        # TODO right now this only works for single ligand reactions
        # TODO add spline fitting
        ligand_to_reactions = reaction_collection.get_lcomb_to_reactions()
        ligand_to_reactions = {k[0]: v for k, v in ligand_to_reactions.items()}
        data = dict()
        for ligand, reactions in ligand_to_reactions.items():
            amounts = []
            amount_units = []
            values = []
            ref_values = []
            identifiers = []
            for r in reactions:
                r: LigandExchangeReaction
                ref_values += PropertyGetter.get_reference_value(r, reaction_collection, property_name, average=False)
                amount = r.ligand_solutions[0].amount
                amount_unit = r.ligand_solutions[0].amount_unit
                amount_units.append(amount_unit)
                value = PropertyGetter.get_property_value(r, property_name)
                amounts.append(amount)
                values.append(value)
                identifiers.append(r.identifier)
            data[ligand] = {
                "amount": amounts, "amount_unit": amount_units[0],
                "values": values, "ref_values": ref_values,
                "identifiers": identifiers
            }
        return data

# PropertyGetter = Callable[[LigandExchangeReaction, ], float]
