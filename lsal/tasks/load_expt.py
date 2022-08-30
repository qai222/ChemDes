from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any, Union
from typing import Tuple, List, Callable

import numpy as np
import pandas as pd
from loguru import logger

from lsal.schema import Molecule, ReactionCondition, ReactantSolution, NanoCrystal, _EPS, L1XReactionCollection, \
    L1XReaction, LXReaction, assign_reaction_results
from lsal.utils import FilePath, padding_vial_label, strip_extension, get_extension, get_basename
from lsal.utils import is_close_relative, file_exists

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


def load_peak_info(fn: FilePath, identifier_prefix: str, vial_column: str = "wellLabel") -> dict[str, dict]:
    peak_df = pd.read_csv(fn)
    peak_df.drop(peak_df.filter(regex="Unnamed"), axis=1, inplace=True)
    peak_df.dropna(axis=0, inplace=True, how="all")

    vial_col = [c for c in peak_df.columns if vial_column in c]
    assert len(vial_col) == 1, "`{}` can only appear once in the columns of: {}".format(vial_column, fn)
    vial_col = vial_col[0]

    data = dict()
    for record in peak_df.to_dict(orient="records"):
        vial = identifier_prefix + "@@" + padding_vial_label(record[vial_col])
        data[vial] = dict()
        data[vial]["peak_file"] = os.path.basename(fn)
        data[vial].update(record)
    return data


def load_robot_input_l1(
        fn: FilePath,
        ligand_inventory: list[Molecule],
        solvent_inventory: list[Molecule],
        input_columns: Tuple[str] = (
                'Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)',
                'Reagent5 (ul)', 'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)',
                'Reagent10 (ul)', 'Labware ID:', 'Reaction Parameters', 'Parameter Values', 'Reagents',
                'Reagent Name', 'Reagent Identity', 'Reagent Concentration (uM)', 'Liquid Class',
        ),
        reagent_columns: Tuple[str] = ("Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)"),
        vial_column: str = 'Vial Site',
        volume_unit: str = 'ul',
        concentration_unit: str = 'uM',
        possible_solvent: Tuple[str] = ('m-xylene',),
) -> List[L1XReaction]:
    """
    load robotinput to ligand reactions

    :param fn: robotinput file
    :param solvent_inventory:
    :param ligand_inventory: molecules in the inventory,
            labels in the input file will be used to map reagents to ligands
    :param input_columns: useful columns in the robotinput file
    :param reagent_columns: columns regarding reagents used
    :param vial_column: columns used to identify each reaction (vial)
    :param volume_unit: e.g. "ul"
    :param concentration_unit: e.g. "uM"
    :param possible_solvent: list of low case solvent names
    :return: a list of reactions
    """
    robot_input_name = strip_extension(os.path.basename(fn))
    ext = get_extension(fn)
    if ext == "csv":
        robotinput_df = pd.read_csv(fn)
    elif ext == "xls":
        robotinput_df = pd.read_excel(fn, 0)
    else:
        raise NotImplementedError

    # remove excess cells
    robotinput_df.drop(robotinput_df.filter(regex="Unnamed"), axis=1, inplace=True)
    robotinput_df.dropna(axis=0, inplace=True, how="all")

    # sanity check
    assert set(reagent_columns).issubset(
        set(input_columns)), "`reagent_columns` is not a subset of `input_columns`!"
    assert set(robotinput_df.columns) == set(
        input_columns), "`input_columns` is not identical to what we read from: {}".format(fn)

    # load general reaction conditions
    condition_df = robotinput_df.loc[:, ["Reaction Parameters", "Parameter Values"]]
    condition_df = condition_df.dropna(axis=0, how="all")
    reaction_conditions = []
    for k, v in zip(condition_df.iloc[:, 0], condition_df.iloc[:, 1]):
        rc = ReactionCondition(k, v)
        reaction_conditions.append(rc)

    # load reagents
    reagent_df = robotinput_df.loc[:, reagent_columns]
    reagent_df.dropna(axis=0, inplace=True, how="all")
    reagent_index_to_reactant = reagent_df_parser(
        df=reagent_df, ligand_inventory=ligand_inventory,
        solvent_inventory=solvent_inventory,
        volume_unit=volume_unit,
        concentration_unit=concentration_unit,
        used_solvents=possible_solvent,
    )

    # load volumes
    volume_df = robotinput_df.loc[:, ["Vial Site", ] + list(reagent_index_to_reactant.keys())]
    reactions = []
    for record in volume_df.to_dict(orient="records"):
        # vial
        logger.info("loading reaction: {}".format(record))
        vial = padding_vial_label(record[vial_column])
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
            nc_solution=nc_reactant, ligand_solutions=ligand_reactants, properties=None,
        )
        reactions.append(reaction)
    return reactions


def reagent_df_parser(
        df: pd.DataFrame, ligand_inventory: list[Molecule], solvent_inventory: list[Molecule],
        volume_unit: str, concentration_unit: str, used_solvents: Tuple[str],
) -> dict[str, ReactantSolution]:
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
            reagent_int_label = int(reagent_identity)
            ligand_molecule = Molecule.select_from_list(reagent_int_label, ligand_inventory, "int_label")
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


def load_reactions_from_expt_files_l1(
        experiment_input_files: list[FilePath],
        experiment_output_files: list[FilePath],
        ligand_inventory: list[Molecule],
        solvent_inventory: list[Molecule],
        properties: dict = None,
) -> L1XReactionCollection:
    all_reactions = []
    for ifile, ofile in zip(experiment_input_files, experiment_output_files):
        reactions = load_robot_input_l1(ifile, ligand_inventory, solvent_inventory, )
        peak_data = load_peak_info(ofile, identifier_prefix=strip_extension(get_basename(ifile)))
        assign_reaction_results(reactions, peak_data)
        all_reactions += reactions

    # remove suspicious reactions
    n_real = 0
    n_blank = 0
    n_ref = 0
    for r in all_reactions:
        r: L1XReaction
        if r.is_reaction_blank_reference:
            n_blank += 1
        elif r.is_reaction_nc_reference:
            n_ref += 1
        elif r.is_reaction_real:
            n_real += 1
        else:
            raise Exception("reaction type cannot be determined: {}".format(r))
    logger.warning("REACTIONS LOADED: blank/ref/real: {}/{}/{}".format(n_blank, n_ref, n_real))
    slc = L1XReactionCollection(reactions=all_reactions, properties=properties)
    slc.properties.update(
        dict(experiment_input_files=experiment_input_files, experiment_output_files=experiment_output_files,
             ligand_inventory=ligand_inventory, solvent_inventory=solvent_inventory, ))
    return slc


def is_internal_fom(fom_name: str) -> bool:
    """ does the fom calculation need reference experiments? """
    if fom_name in ("fom2", "fom3"):
        return True
    return False


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
            value2 = value2_n / value2_d
            assert is_close_relative(value2, value, 1e-5) or pd.isna(value) or pd.isna(value2)
        return value

    @staticmethod
    def _get_reaction_property(r: LXReaction, property_suffix: str) -> float:
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


def collect_reactions_l1(
        folder: FilePath,
        ligand_inventory: list[Molecule],
        solvent_inventory: list[Molecule],
        exclude_reaction: Callable = None,
) -> L1XReactionCollection:
    """
    load robotinput and peakinfo files to a `ReactionCollection`
    there should be a `file_pairs.csv` in the folder describing the pairing between input and output files
    """
    pair_file: FilePath
    pair_file = f"{folder}/file_pairs.csv"
    assert file_exists(pair_file), f"the file describing pairing does not exist: {pair_file}"
    df_file = pd.read_csv(pair_file)
    experiment_input_files = [os.path.join(folder, fn) for fn in df_file.robotinput.tolist()]
    experiment_output_files = [os.path.join(folder, fn) for fn in df_file.peakinfo.tolist()]
    reaction_collection = load_reactions_from_expt_files_l1(
        experiment_input_files=experiment_input_files,
        experiment_output_files=experiment_output_files,
        ligand_inventory=ligand_inventory,
        solvent_inventory=solvent_inventory,
    )
    if exclude_reaction is None:
        exclude_reaction = lambda x: False

    final_reactions = []
    for r in reaction_collection.reactions:
        if exclude_reaction(r):
            logger.warning(f"reaction is excluded: {r.identifier}")
        else:
            final_reactions.append(r)

    return L1XReactionCollection(final_reactions, reaction_collection.properties)
