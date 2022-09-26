from __future__ import annotations

import functools
import inspect
import os
import pprint
import re
from copy import deepcopy
from typing import Any, Union, Tuple, List, Callable

import numpy as np
import pandas as pd
from loguru import logger
from monty.json import MSONable

from lsal.schema import Molecule, ReactionCondition, ReactantSolution, NanoCrystal, _EPS, L1XReactionCollection, \
    L1XReaction, LXReaction, assign_reaction_results
from lsal.utils import FilePath, padding_vial_label, strip_extension, get_extension, get_basename, \
    is_close_relative, file_exists

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


class ExptLoader(MSONable):

    def __init__(
            self,

            # file path
            expt_input: FilePath,
            expt_output: FilePath,

            # inventory
            ligand_inventory: list[Molecule],
            solvent_inventory: list[Molecule],

            # identify ligands
            ligand_identifier_convert: dict[Any, Any],
            ligand_identifier_type: str,

            # robot input file specifications
            expt_input_columns: Tuple[str] = (
                    'Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)',
                    'Reagent5 (ul)', 'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)',
                    'Reagent10 (ul)', 'Labware ID:', 'Reaction Parameters', 'Parameter Values', 'Reagents',
                    'Reagent Name', 'Reagent Identity', 'Reagent Concentration (uM)', 'Liquid Class',
            ),
            expt_input_reagent_columns: Tuple[str] = (
                    "Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)"
            ),
            expt_input_vial_column: str = 'Vial Site',
            reagent_volume_unit: str = 'ul',
            reagent_concentration_unit: str = 'uM',
            reagent_possible_solvent: Tuple[str] = ('m-xylene',),

            # peak file specification
            expt_output_vial_column: str = "wellLabel",
    ):
        self.expt_input = expt_input
        self.expt_output = expt_output
        self.ligand_inventory = ligand_inventory
        self.solvent_inventory = solvent_inventory
        self.ligand_identifier_convert = ligand_identifier_convert
        self.ligand_identifier_type = ligand_identifier_type
        self.expt_input_columns = expt_input_columns
        self.expt_input_reagent_columns = expt_input_reagent_columns
        self.expt_input_vial_column = expt_input_vial_column
        self.expt_output_vial_column = expt_output_vial_column
        self.reagent_volume_unit = reagent_volume_unit
        self.reagent_concentration_unit = reagent_concentration_unit
        self.reagent_possible_solvent = reagent_possible_solvent

    def load_peak_info(self) -> dict[str, dict]:
        peak_df = pd.read_csv(self.expt_output)
        peak_df.drop(peak_df.filter(regex="Unnamed"), axis=1, inplace=True)
        peak_df.dropna(axis=0, inplace=True, how="all")

        identifier_prefix = strip_extension(get_basename(self.expt_input))

        vial_col = [c for c in peak_df.columns if self.expt_output_vial_column in c]
        assert len(vial_col) == 1, "`{}` can only appear once in the columns of: {}".format(
            self.expt_output_vial_column, self.expt_output
        )
        vial_col = vial_col[0]

        data = dict()
        for record in peak_df.to_dict(orient="records"):
            vial = identifier_prefix + "@@" + padding_vial_label(record[vial_col])
            data[vial] = dict()
            data[vial]["peak_file"] = os.path.basename(self.expt_output)
            data[vial].update(record)
        return data

    def load_robot_input_l1(self, ) -> List[L1XReaction]:
        # load dataframe
        robot_input_name = strip_extension(os.path.basename(self.expt_input))
        ext = get_extension(self.expt_input)
        if ext == "csv":
            robotinput_df = pd.read_csv(self.expt_input)
        elif ext == "xls":
            robotinput_df = pd.read_excel(self.expt_input, 0)
        else:
            raise NotImplementedError

        # remove excess cells
        robotinput_df.drop(robotinput_df.filter(regex="Unnamed"), axis=1, inplace=True)
        robotinput_df.dropna(axis=0, inplace=True, how="all")

        # sanity check
        assert set(self.expt_input_reagent_columns).issubset(
            set(self.expt_input_columns)
        ), "`reagent_columns` is not a subset of `input_columns`!"
        assert set(robotinput_df.columns) == set(
            self.expt_input_columns
        ), "`input_columns` is not identical to what we read from: {}".format(self.expt_input)

        # load general reaction conditions
        condition_df = robotinput_df.loc[:, ["Reaction Parameters", "Parameter Values"]]
        condition_df = condition_df.dropna(axis=0, how="all")
        reaction_conditions = []
        for k, v in zip(condition_df.iloc[:, 0], condition_df.iloc[:, 1]):
            rc = ReactionCondition(k, v)
            reaction_conditions.append(rc)

        # load reagents
        reagent_df = robotinput_df.loc[:, self.expt_input_reagent_columns]
        reagent_df.dropna(axis=0, inplace=True, how="all")
        reagent_index_to_reactant = self.reagent_df_parser(
            df=reagent_df, ligand_inventory=self.ligand_inventory,
            solvent_inventory=self.solvent_inventory,
            volume_unit=self.reagent_volume_unit,
            concentration_unit=self.reagent_concentration_unit,
            used_solvents=self.reagent_possible_solvent,
            molecule_identity_convert=self.ligand_identifier_convert,
            molecule_identity_type=self.ligand_identifier_type,
        )

        # load volumes
        volume_df = robotinput_df.loc[:, ["Vial Site", ] + list(reagent_index_to_reactant.keys())]
        reactions = []
        for record in volume_df.to_dict(orient="records"):
            # vial
            logger.info("loading reaction: {}".format(record))
            vial = padding_vial_label(record[self.expt_input_vial_column])
            identifier = "{}@@{}".format(robot_input_name, vial)

            ligand_reactants = []
            solvent_reactant, nc_reactant = None, None
            for reagent_index, reactant in reagent_index_to_reactant.items():
                volume = record[reagent_index]
                if volume < _EPS:
                    continue
                actual_reactant = deepcopy(reactant)
                actual_reactant.volume = volume
                reactant_def = actual_reactant.properties["definition"]
                if reactant_def == "nc":
                    nc_reactant = actual_reactant
                elif reactant_def == "ligand_solution":
                    ligand_reactants.append(actual_reactant)
                elif reactant_def == "solvent":
                    solvent_reactant = actual_reactant
                else:
                    raise ValueError("wrong definition: {}".format(reactant_def))

            reaction = L1XReaction(
                identifier=identifier, conditions=reaction_conditions, solvent=solvent_reactant,
                nc_solution=nc_reactant, ligand_solutions=ligand_reactants, properties=dict(loader=self.as_dict()),
            )
            reactions.append(reaction)
        return reactions

    @staticmethod
    def reagent_df_parser(
            df: pd.DataFrame, ligand_inventory: list[Molecule], solvent_inventory: list[Molecule],
            volume_unit: str, concentration_unit: str, used_solvents: Tuple[str],
            molecule_identity_convert: dict[Any, Any], molecule_identity_type: str,
    ) -> dict[str, ReactantSolution]:
        """
        parse the cells in robotinput defining reagents

        the key is to identify ligands using the lookup table `molecule_identity_convert` and
        `Molecule.select_from_list` with `molecule_identity_type`
        """
        solvent_material = None
        reagent_index_to_reactant = dict()
        for record in df.to_dict(orient="records"):
            reagent_index = record["Reagents"]
            reagent_name = record["Reagent Name"]
            reagent_identity = record["Reagent Identity"]
            reagent_concentration = record["Reagent Concentration (uM)"]

            if reagent_name.startswith("CPB") and pd.isnull(reagent_concentration):
                logger.info("Found nanocrystal: {}".format(reagent_name))
                material = NanoCrystal(identifier=reagent_name)
                reactant = ReactantSolution(solute=material, volume=np.nan, concentration=None, solvent=None,
                                            properties={"definition": "nc"}, volume_unit=volume_unit,
                                            concentration_unit=concentration_unit, )
                reagent_index_to_reactant[reagent_index] = reactant
            elif reagent_name.lower() in used_solvents and pd.isnull(reagent_concentration):
                solvent_name = reagent_name.lower()
                solvent_material = Molecule.select_from_list(solvent_name, solvent_inventory, "name")
                reactant = ReactantSolution(
                    solute=solvent_material, volume=np.nan, concentration=0.0, solvent=solvent_material,
                    properties={"definition": "solvent"},
                    volume_unit=volume_unit, concentration_unit=concentration_unit,
                )
            else:
                reagent_identifier = molecule_identity_convert[reagent_identity]
                ligand_molecule = Molecule.select_from_list(reagent_identifier, ligand_inventory,
                                                            molecule_identity_type)
                reactant = ReactantSolution(
                    solute=ligand_molecule, volume=np.nan, concentration=reagent_concentration, solvent=None,
                    properties={"definition": "ligand_solution"},
                    volume_unit=volume_unit, concentration_unit=concentration_unit
                )
            reagent_index_to_reactant[reagent_index] = reactant

        for reagent_index, reactant in reagent_index_to_reactant.items():
            reactant.solvent = solvent_material
        return {"{} (ul)".format(k): v for k, v in reagent_index_to_reactant.items()}
        # note the "ul" in keys here are hardcoded for robotinput files

    def load_l1(self) -> L1XReactionCollection:
        logger.info(">>> start loading l1 reactions with loader params:")
        logger.info(
            pprint.pformat({k: v for k, v in self.as_dict().items() if isinstance(v, str) or isinstance(v, float)},
                           indent=4))
        reactions = self.load_robot_input_l1()
        peak_data = self.load_peak_info()
        assign_reaction_results(reactions, peak_data)
        rc = L1XReactionCollection(reactions=reactions)
        FomCalculator(rc).update_foms()
        logger.info(rc.__repr__())
        return rc

    @staticmethod
    def collect_reactions_l1(
            # file path
            folder: FilePath,

            # inventory
            ligand_inventory: list[Molecule],
            solvent_inventory: list[Molecule],

            # identify ligands
            ligand_identifier_convert: dict[Any, Any],
            ligand_identifier_type: str,

            # robot input file specifications
            expt_input_columns: Tuple[str] = (
                    'Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)',
                    'Reagent5 (ul)', 'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)',
                    'Reagent10 (ul)', 'Labware ID:', 'Reaction Parameters', 'Parameter Values', 'Reagents',
                    'Reagent Name', 'Reagent Identity', 'Reagent Concentration (uM)', 'Liquid Class',
            ),
            expt_input_reagent_columns: Tuple[str] = (
                    "Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)"
            ),
            expt_input_vial_column: str = 'Vial Site',
            reagent_volume_unit: str = 'ul',
            reagent_concentration_unit: str = 'uM',
            reagent_possible_solvent: Tuple[str] = ('m-xylene',),

            # peak file specification
            expt_output_vial_column: str = "wellLabel",

            checker: ReactionCheckerL1 = None,
    ) -> L1XReactionCollection:
        """
        load robotinput and peakinfo files to a `ReactionCollection`
        there should be a `file_pairs.csv` in the folder describing the pairing between input and output files
        """
        pair_file: FilePath
        pair_file = f"{folder}/file_pairs.csv"
        assert file_exists(pair_file), f"the file describing pairing does not exist: {pair_file}"
        df_file = pd.read_csv(pair_file)
        assert "robotinput" in df_file.columns
        assert "peakinfo" in df_file.columns
        experiment_input_files = [os.path.join(folder, fn) for fn in df_file.robotinput.tolist()]
        experiment_output_files = [os.path.join(folder, fn) for fn in df_file.peakinfo.tolist()]

        all_reactions = []
        discarded = []
        for input_file, output_file in zip(experiment_input_files, experiment_output_files):
            loader = ExptLoader(
                input_file, output_file, ligand_inventory, solvent_inventory,
                ligand_identifier_convert, ligand_identifier_type,
                expt_input_columns, expt_input_reagent_columns, expt_input_vial_column, reagent_volume_unit,
                reagent_concentration_unit, reagent_possible_solvent, expt_output_vial_column,
            )
            batch_collection = loader.load_l1()
            if checker is None:
                checker = ReactionCheckerL1()

            logger.info(f">>> Checking reactions with a checker:\n{checker.__repr__()}")
            checker.batch_reactions = batch_collection
            for r in batch_collection.reactions:
                checker.check(r)
                if checker.is_passed:
                    all_reactions.append(r)
                else:
                    discarded.append((r, checker.msgs))
                    logger.warning(f"DISCARD reaction: {r.identifier}")
        for discarded_reaction, msgs in discarded:
            msg = '\n'.join([m for m in msgs if m.startswith('CHECK FAILED')])
            logger.warning(f"discard reaction: {discarded_reaction.identifier}\nCHECK MSGS:\n{msg}")
        logger.critical(f"Loading finished: # of reaction passed checks=={len(all_reactions)}")
        logger.critical(f"# of reactions discarded=={len(discarded)}")
        return L1XReactionCollection(all_reactions)


class FomCalculator:
    """
    this is not limited to L1X reactions
    """

    def __init__(self, reaction_collection: L1XReactionCollection):
        self.reaction_collection = reaction_collection
        self.ligand_to_reactions = {k: v for k, v in reaction_collection.ligand_to_reactions_mapping().items()}

    def get_average_ref(self, r: LXReaction, property_name: str):
        return PropertyGetter.get_reference_value(r, self.reaction_collection, property_name, average=True)

    def get_internal_ref(self, r: LXReaction, property_name: str):
        reactions_same_ligand = self.ligand_to_reactions[r.ligand_tuple[0]]  # assuming single ligand
        reactions_same_ligand: list[LXReaction]
        amount_and_fom = [(rr.ligand_solutions[0].amount, PropertyGetter.get_property_value(rr, property_name)) for rr
                          in
                          reactions_same_ligand]
        ref_fom = sorted(amount_and_fom, key=lambda x: x[0])[0][1]
        return ref_fom

    def fom1(self, r: LXReaction) -> float:
        fom = PropertyGetter.get_property_value(r, "pPLQY")
        return fom

    def fom2(self, r: LXReaction) -> float:
        fom = PropertyGetter.get_property_value(r, "pPLQY")
        return fom / self.get_average_ref(r, "pPLQY")

    def fom3(self, r: LXReaction) -> float:
        fom = PropertyGetter.get_property_value(r, "pPLQY")
        return fom - self.get_average_ref(r, "pPLQY")

    def fom4(self, r: LXReaction) -> float:
        fom = PropertyGetter.get_property_value(r, "pPLQY")
        ref_fom = self.get_internal_ref(r, "pPLQY")
        return fom / ref_fom

    def fom5(self, r: LXReaction) -> float:
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

    def update_foms(self):
        for r in self.reaction_collection.reactions:
            for fname in self.fom_function_names:
                fom_func = getattr(self, fname)
                if r.is_reaction_nc_reference:
                    if fname == "fom4":
                        fom = 1
                    elif fname == "fom5":
                        fom = 0
                    else:
                        fom = fom_func(r)
                elif r.is_reaction_blank_reference:
                    fom = np.nan
                else:
                    fom = fom_func(r)
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
            value2_n = PropertyGetter._get_reaction_property(r, "_PL_sum")
            value2_d = PropertyGetter._get_reaction_property(r, "_PL_OD390")
            try:
                value2 = value2_n / value2_d
            except ZeroDivisionError:
                assert abs(value) < 1e-5  # pPLQY should be padded zero if PL_OD390 is zero
                value2 = 0.0
            assert is_close_relative(value2, value, 1e-5) or pd.isna(value) or pd.isna(value2)
        return value

    @staticmethod
    def _get_reaction_property(r: LXReaction, property_suffix: str) -> float:
        possible_properties = [k for k in r.properties if k.strip("'").endswith(property_suffix)]
        assert len(possible_properties) == 1, f"possible property ends with {property_suffix}: {possible_properties}"
        k = possible_properties[0]
        v = r.properties[k]
        try:
            assert isinstance(v, float)
        except AssertionError:
            v = np.nan
        return v

    @staticmethod
    def get_reference_value(
            r: LXReaction, reaction_collection: L1XReactionCollection, property_name: str, average=True
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
    def get_amount_property_data_l1(
            reaction_collection: L1XReactionCollection, property_name: str
    ) -> dict[Molecule, dict[str, Any]]:
        ligand_to_reactions = reaction_collection.ligand_to_reactions_mapping()
        data = dict()
        for ligand, reactions in ligand_to_reactions.items():
            amounts = []
            amount_units = []
            values = []
            ref_values = []
            identifiers = []
            for r in reactions:
                r: LXReaction
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


def reaction_checker_l1(func: Callable[[L1XReaction, ...], Tuple[bool, dict]]) -> Callable[
    [L1XReaction, ...], str]:
    @functools.wraps(func)
    def wrapper_checker(self, r: L1XReaction):
        check_passed, check_info = func(self, r)
        if check_passed:
            return f"CHECK PASSED @@ {r.identifier} @@ {func.__name__}\n{func.__doc__}\n{pprint.pformat(check_info)}"
        else:
            return f"CHECK FAILED @@ {r.identifier} @@ {func.__name__}\n{func.__doc__}\n{pprint.pformat(check_info)}"

    return wrapper_checker


class ReactionCheckerL1:
    def __init__(self, exclude_reaction_specifications: list[Tuple[Any, Any]] = None):
        if exclude_reaction_specifications is None:
            exclude_reaction_specifications = []
        self.exclude_reaction_specifications = exclude_reaction_specifications
        self.batch_reactions = None
        self.msgs = []

    @property
    def checker_methods(self) -> list[Callable[[L1XReaction], str]]:
        checker_methods = [
            getattr(self, name) for name in dir(self) if name.startswith("checker__") and name != 'checker__fom2'
        ]
        checker_methods.append('checker__fom2')
        checker_methods = [cm for cm in checker_methods if inspect.ismethod(cm)]
        return checker_methods

    def __repr__(self):
        s = f"{self.__class__.__name__}:"
        for cf in self.checker_methods:
            s += f"\n{cf.__name__}: {cf.__doc__}"
        return s

    def check(self, r: L1XReaction):
        logger.warning(f">>> CHECKING REACTION: {r.identifier}")
        logger.info(f"reaction properties: {pprint.pformat({k: v for k, v in r.properties.items() if k != 'loader'})}")
        assert self.batch_reactions is not None
        self.msgs = []
        for cf in self.checker_methods:
            msg = cf(r)
            if msg.startswith("CHECK FAILED"):
                logger.critical(msg)
            else:
                logger.info(msg)
            self.msgs.append(msg)

    @property
    def is_passed(self):
        assert len(self.msgs) > 0, "no msg, did you call `self.check()`?"
        return all(m.startswith("CHECK PASSED") for m in self.msgs)

    @reaction_checker_l1
    def checker__dummy(self, r: L1XReaction):
        """ dummy checker, always pass """
        return True, dict()

    # @reaction_checker_l1
    # def checker__real_reaction(self, r: L1XReaction):
    #     """ if this reaction is a non-blank/reference reaction """
    #     return r.is_reaction_real, dict(is_blank=r.is_reaction_blank_reference, is_ref=r.is_reaction_nc_reference)

    @reaction_checker_l1
    def checker__fom2(self, r: L1XReaction):
        """ fom2 should be in (3.5, -1e-5) """
        return 3.5 > r.properties['fom2'] > -1e-5, dict(fom2=r.properties['fom2'])

    @reaction_checker_l1
    def checker__reaction_specification(self, r: L1XReaction):
        """ if this reaction contains specific ligand and substring in its identifier """
        if not r.is_reaction_real:
            return True, dict(not_real_reaction=True)

        check_pass = True
        ligand = r.ligand_tuple[0]
        ligand: Molecule
        for exclude_ligand_identifier, exclude_reaction_identifier_substring in self.exclude_reaction_specifications:
            assert exclude_ligand_identifier is not None or exclude_reaction_identifier_substring is not None
            if exclude_ligand_identifier is None:
                if exclude_reaction_identifier_substring in r.identifier:
                    check_pass = False
            elif exclude_reaction_identifier_substring is None:
                if ligand.identifier == exclude_ligand_identifier:
                    check_pass = False
            else:
                if ligand.identifier == exclude_ligand_identifier and exclude_reaction_identifier_substring in r.identifier:
                    check_pass = False
        return check_pass, dict(ligand_identifier=ligand.identifier, reaction_identifier=r.identifier)

    @reaction_checker_l1
    def checker__od(self, r: L1XReaction):
        """ optical density should be in [0.6x, 1.4x] where x==<avg ref OD>  """
        actual_od = PropertyGetter().get_property_value(r, 'OD')
        ref_od = PropertyGetter().get_reference_value(r, self.batch_reactions, 'OD', average=True)
        assert ref_od > 0

        od_valid = 0.6 * ref_od < actual_od < 1.4 * ref_od
        info_dict = dict(actual_od=actual_od, ref_od=ref_od)

        if r.is_reaction_real:
            logger.warning(
                f"{r.ligand.label}: {r.ligand_solution.amount}\nOD-FOM2: {actual_od} - {r.properties['fom2']}"
            )
            if od_valid:
                return True, info_dict
            else:
                logger.warning(
                    f"this reaction is a real reaction but it has invalid OD: {actual_od} "
                    f"comparing with its reference OD: {ref_od}"
                    "\nThis check is forced to pass and its fom2 is set to zero."
                )
                r.properties['fom2'] = 0.0
                return True, info_dict
        else:
            return od_valid, info_dict
