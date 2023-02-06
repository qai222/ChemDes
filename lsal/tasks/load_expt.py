from __future__ import annotations

import dataclasses
import functools
import inspect
import os
import pprint
from copy import deepcopy
from typing import Any, Tuple, List, Callable

import numpy as np
import pandas as pd
from loguru import logger
from monty.json import MSONable

from lsal.schema import Molecule, ReactionCondition, ReactantSolution, NanoCrystal, _EPS, L1XReactionCollection, \
    L1XReaction, assign_reaction_results
from lsal.utils import FilePath, padding_vial_label, strip_extension, get_extension, get_basename, \
    file_exists


@dataclasses.dataclass
class BatchParams:
    # identify ligands
    ligand_identifier_convert: dict[Any, Any]
    ligand_identifier_type: str  # this is the identifier type after conversion

    # robot input file specifications
    expt_input_columns: Tuple[str, ...]
    expt_input_reagent_columns: Tuple[str, ...]
    expt_input_condition_columns: Tuple[str, str]
    expt_input_vial_column: str
    reagent_volume_unit: str
    reagent_concentration_unit: str
    reagent_possible_solvent: Tuple[str, ...]

    # peak file specification
    expt_output_vial_column: str
    expt_output_wall_tag_column_suffix: str
    expt_output_od_column_suffix: str
    expt_output_fom_column_suffix: str

    def as_dict(self):
        return dataclasses.asdict(self)


class BatchLoader:
    """
    load a batch (plate) of reactions, usually two ligands
    """

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
            ligand_identifier_type: str,  # this is the identifier type after conversion

            # robot input file specifications
            expt_input_columns: Tuple[str, ...],
            expt_input_reagent_columns: Tuple[str, ...],
            expt_input_condition_columns: Tuple[str, str],
            expt_input_vial_column: str,
            reagent_volume_unit: str,
            reagent_concentration_unit: str,
            reagent_possible_solvent: Tuple[str, ...],

            # peak file specification
            expt_output_vial_column: str,
            expt_output_wall_tag_column_suffix: str,
            expt_output_od_column_suffix: str,
            expt_output_fom_column_suffix: str,
    ):
        self.expt_input_condition_columns = expt_input_condition_columns
        self.expt_output_fom_column_suffix = expt_output_fom_column_suffix
        self.expt_output_od_column_suffix = expt_output_od_column_suffix
        self.expt_output_wall_tag_column_suffix = expt_output_wall_tag_column_suffix
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

    @staticmethod
    def init_keys() -> list[str]:
        signature = inspect.signature(BatchLoader.__init__).parameters
        return [n.name for n in signature.values() if n.name != 'self']

    @property
    def params(self) -> dict:
        return {
            n: getattr(self, n) for n in self.init_keys()
            if n not in ['self', 'ligand_inventory', 'solvent_inventory']
        }

    @property
    def batch_identifier(self) -> str:
        return strip_extension(get_basename(self.expt_input))

    @staticmethod
    def get_column_with_suffix(df: pd.DataFrame, suffix: str):
        possible_columns = [c for c in df.columns if c.endswith(suffix)]
        if len(possible_columns) > 1:
            possible_columns = [c for c in possible_columns if "_ref_" not in c.lower()]
        assert len(possible_columns) > 0, f"no possible column found for suffix: {suffix}"
        assert len(possible_columns) == 1, f"multiple possible columns: {possible_columns}"
        return possible_columns[0]

    def load_peak_info(self) -> dict[str, dict[str, Any]]:
        peak_df = pd.read_csv(self.expt_output, low_memory=False)

        # remove excess cells
        peak_df.dropna(axis=0, inplace=True, how="all")

        wall_tag_column = self.get_column_with_suffix(peak_df, self.expt_output_wall_tag_column_suffix)
        od_column = self.get_column_with_suffix(peak_df, self.expt_output_od_column_suffix)
        fom_column = self.get_column_with_suffix(peak_df, self.expt_output_fom_column_suffix)

        # collect peak info
        data = dict()
        for record in peak_df.to_dict(orient="records"):
            reaction_name = f"{self.batch_identifier}@@{padding_vial_label(record[self.expt_output_vial_column])}"
            reaction_peak_info = {
                "PeakFile": get_basename(self.expt_output),
                "OpticalDensity": record[od_column],
                "FigureOfMerit": record[fom_column],
                "WallTag": record[wall_tag_column],
            }
            data[reaction_name] = reaction_peak_info
        return data

    def load_robot_input_l1(self, ) -> List[L1XReaction]:
        # load dataframe
        ext = get_extension(self.expt_input)
        if ext == "csv":
            robotinput_df = pd.read_csv(self.expt_input, low_memory=False)
        elif ext == "xls":
            robotinput_df = pd.read_excel(self.expt_input, 0)
        else:
            raise NotImplementedError

        # remove excess cells
        robotinput_df.dropna(axis=0, inplace=True, how="all")

        # sanity check
        assert set(self.expt_input_reagent_columns).issubset(
            set(self.expt_input_columns)
        ), "`reagent_columns` is not a subset of `input_columns`!"
        # assert set(robotinput_df.columns) == set(
        #     self.expt_input_columns
        # ), "`input_columns` is not identical to what we read from: {}".format(self.expt_input)

        # load general reaction conditions
        condition_df = robotinput_df.loc[:, self.expt_input_condition_columns]
        assert len(self.expt_input_condition_columns) == 2, "should only have 2 condition columns"
        condition_df = condition_df.dropna(axis=0, how="all")
        reaction_conditions = []
        for k, v in zip(condition_df.iloc[:, 0], condition_df.iloc[:, 1]):
            rc = ReactionCondition(k, v)
            reaction_conditions.append(rc)

        # load reagents
        reagent_df = robotinput_df.loc[:, self.expt_input_reagent_columns]
        reagent_df.dropna(axis=0, inplace=True, how="all")
        reagent_index_to_reactant = self.reagent_df_parser(
            df=reagent_df,
            ligand_inventory=self.ligand_inventory,
            solvent_inventory=self.solvent_inventory,
            volume_unit=self.reagent_volume_unit,
            concentration_unit=self.reagent_concentration_unit,
            used_solvents=self.reagent_possible_solvent,
            molecule_identity_convert=self.ligand_identifier_convert,
            molecule_identity_type=self.ligand_identifier_type,
        )

        # load volumes
        volume_df = robotinput_df.loc[:, [self.expt_input_vial_column, ] + list(reagent_index_to_reactant.keys())]
        reactions = []
        for record in volume_df.to_dict(orient="records"):
            # vial
            logger.info("loading reaction: {}".format(record))
            vial = padding_vial_label(record[self.expt_input_vial_column])
            identifier = "{}@@{}".format(self.batch_identifier, vial)

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
                nc_solution=nc_reactant, ligand_solutions=ligand_reactants,
                properties=dict(batch_name=self.batch_identifier),
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
            if pd.isna(reagent_name):
                if not pd.isna(reagent_identity) and not pd.isna(reagent_concentration):
                    logger.warning(f"found a reagent without name: {reagent_identity}")
                    reagent_name = reagent_identity
                else:
                    continue
            if reagent_name.startswith("CPB") and pd.isna(reagent_concentration):
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
        logger.warning(f"{self.__class__.__name__}: LOADING BATCH=={self.batch_identifier}")
        logger.info(pprint.pformat(self.params, indent=4))
        reactions = self.load_robot_input_l1()
        peak_data = self.load_peak_info()
        assign_reaction_results(reactions, peak_data)
        rc = L1XReactionCollection(reactions=reactions, properties=dict(batch_name=self.batch_identifier))
        logger.info(rc.__repr__())
        return rc


class CampaignLoader:
    """
    a campaign is just a set of batches
    """

    def __init__(
            self,
            campaign_name: str, campaign_folder: FilePath,
            ligand_inventory: list[Molecule],
            solvent_inventory: list[Molecule],
            batch_params: BatchParams,
    ):
        self.batch_params = batch_params.as_dict()
        self.solvent_inventory = solvent_inventory
        self.ligand_inventory = ligand_inventory
        self.campaign_folder = campaign_folder
        self.campaign_name = campaign_name

        batch_params_defined_in_campaign = {'expt_input', 'expt_output', 'ligand_inventory', 'solvent_inventory', }
        defined_batch_params = set(self.batch_params.keys()).union(batch_params_defined_in_campaign)
        assert defined_batch_params == set(BatchLoader.init_keys())

    @property
    def io_dict(self):
        pair_file: FilePath
        pair_file = f"{self.campaign_folder}/file_pairs.csv"
        assert file_exists(pair_file), f"the file describing pairing does not exist: {pair_file}"
        df_file = pd.read_csv(pair_file)
        assert "robotinput" in df_file.columns
        assert "peakinfo" in df_file.columns
        experiment_input_files = [os.path.join(self.campaign_folder, fn) for fn in df_file.robotinput.tolist()]
        experiment_output_files = [os.path.join(self.campaign_folder, fn) for fn in df_file.peakinfo.tolist()]
        return dict(zip(experiment_input_files, experiment_output_files))

    @property
    def batch_names(self):
        return [strip_extension(get_basename(input_file)) for input_file in self.io_dict.keys()]

    def load(
            self,
            batch_checkers: dict[str, BatchCheckerL1]
    ):
        logger.warning(f"{self.__class__.__name__}: {self.campaign_name} @ {self.campaign_folder}")
        check_msgs = dict()
        reactions_campaign = []
        reactions_campaign_passed = []
        reactions_campaign_discarded = []
        for input_file, output_file in self.io_dict.items():
            batch_loader = BatchLoader(
                expt_input=input_file,
                expt_output=output_file,
                ligand_inventory=self.ligand_inventory,
                solvent_inventory=self.solvent_inventory,
                **self.batch_params
            )
            batch_collection = batch_loader.load_l1()

            batch_name = batch_collection.properties['batch_name']
            batch_checker = batch_checkers[batch_name]
            passed_reactions, discarded_reactions = batch_checker.check_batch(batch_collection)

            check_msgs.update(batch_checker.check_msgs)
            reactions_campaign += batch_collection.reactions
            reactions_campaign_passed += passed_reactions
            reactions_campaign_discarded += discarded_reactions
        logger.info(f"Campaign reactions:\n {L1XReactionCollection(reactions_campaign).__repr__()}")
        logger.info(f"Campaign reactions passed:\n {L1XReactionCollection(reactions_campaign_passed).__repr__()}")
        return check_msgs, reactions_campaign, reactions_campaign_passed, reactions_campaign_discarded


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


class BatchCheckerL1(MSONable):
    def __init__(
            self,
            exclude_ligand_identifiers: list[str] = None,
            check_msgs: dict[str, list[str]] = None,
            check_results: dict[str, bool] = None,
    ):
        if check_results is None:
            check_results = dict()
        self.check_results = check_results
        if check_msgs is None:
            check_msgs = dict()
        self.check_msgs = check_msgs
        if exclude_ligand_identifiers is None:
            exclude_ligand_identifiers = []
        self.exclude_ligand_identifiers = exclude_ligand_identifiers
        self.batch_rc = None

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
        logger.info(f"reaction properties: {pprint.pformat(r.properties)}")
        assert self.batch_rc is not None
        msgs = []
        passed = True
        for cf in self.checker_methods:
            msg = cf(r)
            if msg.startswith("CHECK FAILED"):
                logger.critical(msg)
                passed = False
            else:
                logger.info(msg)
            msgs.append(msg)
        self.check_msgs[r.identifier] = msgs
        self.check_results[r.identifier] = passed

    def check_batch(self, batch_rc: L1XReactionCollection) -> Tuple[list[L1XReaction], list[L1XReaction]]:
        assert len(self.check_msgs) == 0
        assert len(self.check_results) == 0
        self.batch_rc = batch_rc
        passed = []
        discarded = []
        for r in self.batch_rc.reactions:
            self.check(r)
            if self.check_results[r.identifier]:
                passed.append(r)
            else:
                discarded.append(r)
        return passed, discarded

    @reaction_checker_l1
    def checker__dummy(self, r: L1XReaction):
        """ dummy checker, always pass """
        return True, dict()

    @reaction_checker_l1
    def checker__fom_and_od(self, r: L1XReaction):
        """ figure of merit for real reactions should be in (3.5, -1e-5) or NaN (iff OD is NaN) """
        assert self.batch_rc is not None
        ref_reactions = [ref_r for ref_r in self.batch_rc.reactions if ref_r.is_reaction_nc_reference]
        ref_od = np.mean([rr.properties['OpticalDensity'] for rr in ref_reactions])
        fom = r.properties['FigureOfMerit']
        od = r.properties['OpticalDensity']
        if r.is_reaction_real:
            info = dict(is_real=True, ref_od=ref_od, od=od, fom=fom)
            return 3.5 > fom > 1e-5 or (pd.isna(fom) and pd.isna(od)) or (abs(od) < 1e-5 and abs(fom) < 1e-5), info
        else:
            info = dict(is_real=False, ref_od=ref_od, od=od, fom=fom)
            return True, info

    @reaction_checker_l1
    def checker__wanted_ligand(self, r: L1XReaction):
        """ if this reaction contains wanted ligand """
        check_pass = True
        if r.is_reaction_real:
            info = dict(ligand=r.ligand)
            if r.ligand.identifier in self.exclude_ligand_identifiers:
                check_pass = False
        else:
            info = dict(ligand=None)
        return check_pass, info
