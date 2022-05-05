import logging
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from monty.json import MSONable

from lsal.schema import Molecule, ReactionOneLigand, ReactionCondition, ReactantSolvent, ReactantSolution, Reactant, \
    NanoCrystal, SolventMolecule
from lsal.tasks.io import ligand_to_ml_input
from lsal.utils import strip_extension, FilePath, padding_vial_label

"""
functions/classes for single ligand campaigns
default to NIMBUS robot
"""

_DefaultRobotInputColumns = (
    'Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)',
    'Reagent5 (ul)', 'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)',
    'Reagent10 (ul)', 'Labware ID:', 'Reaction Parameters', 'Parameter Values', 'Reagents',
    'Reagent Name', 'Reagent Identity', 'Reagent Concentration (uM)', 'Liquid Class',
)
_DefaultReagentColumns = ("Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)")
_VialCol = _DefaultRobotInputColumns[0]
_ConcentrationUnit = "uM"
_VolumeUnit = "ul"
_DefaultSolvents = ["m-xylene", ]


class SingleLigandCampaign(MSONable):

    def __init__(
            self,
            name: str,

            experiment_input_files: list[FilePath],
            experiment_output_files: list[FilePath],
            additional_files: list[FilePath],

            ligand_inventory: list[Molecule],
            reactions: list[ReactionOneLigand],
    ):
        self.experiment_output_files = [os.path.abspath(p) for p in experiment_output_files]
        self.experiment_input_files = [os.path.abspath(p) for p in experiment_input_files]
        self.additional_files = additional_files
        self.ligand_inventory = ligand_inventory
        self.name = name
        self.reactions = reactions
        assert len(self.experiment_input_files) == len(self.experiment_output_files)
        logging.warning("=" * 12)
        logging.warning("{}: {}".format(self.__class__.__name__, self.name))
        logging.warning("------>> PAIRING FILES <<------")
        for i in range(len(self.experiment_input_files)):
            ifile, ofile = experiment_input_files[i], experiment_output_files[i]
            logging.warning("input <--> output: {} <--> {}".format(os.path.basename(ifile), os.path.basename(ofile)))

    def get_ligand_to_reactions(self) -> dict[Molecule, list[ReactionOneLigand]]:
        real_reactions = [r for r in self.reactions if
                          not (r.is_reaction_blank_reference or r.is_reaction_nc_reference)]
        return ReactionOneLigand.group_by_ligand(real_reactions)

    def get_ml_input(self, descriptor_csv: FilePath):

        ligand_to_reactions = self.get_ligand_to_reactions()
        return ligand_to_ml_input(
            ligand_to_data=ligand_to_reactions,
            data_type="reaction",
            ligand_inventory=self.ligand_inventory,
            descriptor_csv=descriptor_csv
        )

    @classmethod
    def from_files(
            cls,
            name: str,

            experiment_input_files: list[FilePath],
            experiment_output_files: list[FilePath],
            additional_files: list[FilePath],

            ligand_inventory: list[Molecule],

            input_columns=_DefaultRobotInputColumns,
            reagent_columns=_DefaultReagentColumns,
    ):
        all_reactions = []
        for ifile, ofile in zip(experiment_input_files, experiment_output_files):
            reactions = SingleLigandCampaign.load_robot_input(ifile, ligand_inventory, input_columns, reagent_columns)
            peak_data = SingleLigandCampaign.load_peak_info(ofile,
                                                            identifier_prefix=strip_extension(os.path.basename(ifile)))
            SingleLigandCampaign.assign_fom(reactions, peak_data)
            all_reactions += reactions

        # remove suspicious reactions
        all_reactions = [r for r in all_reactions if SingleLigandCampaign.sniff_reaction(r)]

        n_real = 0
        n_blank = 0
        n_ref = 0
        for r in all_reactions:
            r: ReactionOneLigand
            logging.warning("Reaction loaded: {}".format(r))
            if r.is_reaction_blank_reference:
                n_blank += 1
            elif r.is_reaction_nc_reference:
                n_ref += 1
            else:
                assert r.properties["fom"] > 1e-7
                n_real += 1
        logging.warning("loaded blank/ref/real: {}/{}/{}".format(n_blank, n_ref, n_real))

        return cls(
            name=name, experiment_input_files=experiment_input_files, experiment_output_files=experiment_output_files,
            additional_files=additional_files, ligand_inventory=ligand_inventory, reactions=all_reactions
        )

    @staticmethod
    def sniff_reaction(r: ReactionOneLigand) -> bool:
        sniff = []
        if "fom" in r.properties:
            sniff.append(r.properties["fom"] > 1e-7)
        else:
            sniff.append(True)
        return all(v for v in sniff)

    @staticmethod
    def default_reagent_df_parser(df: pd.DataFrame, ligand_inventory: list[Molecule]) -> dict[str, Reactant]:
        nc_material, solvent_material = None, None
        reagent_index_to_reactant = dict()
        for record in df.to_dict(orient="records"):

            reagent_index = record["Reagents"]
            reagent_name = record["Reagent Name"]
            reagent_identity = record["Reagent Identity"]
            reagent_concentration = record["Reagent Concentration (uM)"]

            if reagent_name.startswith("CPB") and pd.isnull(reagent_concentration):
                logging.warning("Found nanocrystal: {}".format(reagent_name))
                material = NanoCrystal(identifier="CsPbI3-{}".format(reagent_name), label=reagent_name)
                nc_material = material
                reactant = ReactantSolution(
                    solute=material, volume=np.nan, concentration=np.nan, solvent=None, properties={"definition": "nc"},
                    volume_unit=_VolumeUnit, concentration_unit=_ConcentrationUnit
                )
                reagent_index_to_reactant[reagent_index] = reactant
            elif reagent_name.lower() in _DefaultSolvents and pd.isnull(reagent_concentration):
                material = SolventMolecule(identifier=reagent_name.lower(), label=reagent_name)
                solvent_material = material
                reactant = ReactantSolvent(
                    material=material, volume=np.nan, properties={"definition": "solvent"}, volume_unit=_VolumeUnit
                )
            else:
                reagent_identity = "Ligand-{0:0>4}".format(int(reagent_identity))
                material = Molecule.select_from_inventory(reagent_identity, ligand_inventory, "label")
                reactant = ReactantSolution(
                    solute=material, volume=np.nan, concentration=reagent_concentration, solvent=None,
                    properties={"definition": "ligand_solution"},
                    volume_unit=_VolumeUnit, concentration_unit=_ConcentrationUnit
                )
            reagent_index_to_reactant[reagent_index] = reactant

        for reagent_index, reactant in reagent_index_to_reactant.items():
            if not isinstance(reactant, ReactantSolvent):
                reactant.solvent = solvent_material
        return {"{} (ul)".format(k): v for k, v in reagent_index_to_reactant.items()}

    @staticmethod
    def assign_fom(reactions: list[ReactionOneLigand], peak_data: dict[str, dict]):
        assert len(peak_data) == len(reactions)
        for i in range(len(reactions)):
            r = reactions[i]
            data = peak_data[r.identifier]
            r.properties.update(data)

    @staticmethod
    def load_peak_info(peak_file: FilePath, identifier_prefix="") -> dict[str, dict]:
        peak_df = pd.read_csv(peak_file)
        peak_df.drop(peak_df.filter(regex="Unnamed"), axis=1, inplace=True)
        peak_df.dropna(axis=0, inplace=True, how="all")
        vial_col = [c for c in peak_df.columns if "wellLabel" in c][0]
        for c in peak_df.columns:
            print(c.strip("'"))
        fom_col = [c for c in peak_df.columns if c.strip("'").endswith("PL_FOM")][0]
        data = dict()
        for record in peak_df.to_dict(orient="records"):
            vial = identifier_prefix + "--@--" + padding_vial_label(record[vial_col])
            fom = record[fom_col]
            data[vial] = {"fom": fom}
            data[vial]["peak_file"] = os.path.basename(peak_file)
            data[vial].update(record)
        return data

    @staticmethod
    def load_robot_input(
            f: FilePath, ligand_inventory: list[Molecule], input_columns=_DefaultRobotInputColumns,
            reagent_columns=_DefaultReagentColumns,
    ) -> list[ReactionOneLigand]:
        robot_input_name = strip_extension(os.path.basename(f))

        robotinput_df = pd.read_csv(f)
        robotinput_df.drop(robotinput_df.filter(regex="Unnamed"), axis=1, inplace=True)
        robotinput_df.dropna(axis=0, inplace=True, how="all")

        # sanity check
        assert set(reagent_columns).issubset(
            set(input_columns)), "`reagent_columns` is not a subset of `input_columns`!"
        assert set(robotinput_df.columns) == set(
            input_columns), "`input_columns` is not identical to what we read from: {}".format(f)

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
        reagent_index_to_reactant = SingleLigandCampaign.default_reagent_df_parser(df=reagent_df,
                                                                                   ligand_inventory=ligand_inventory)

        # load volumes
        volume_df = robotinput_df.loc[:, ["Vial Site", ] + list(reagent_index_to_reactant.keys())]
        reactions = []
        for record in volume_df.to_dict(orient="records"):
            # vial
            vial = padding_vial_label(record[_VialCol])
            identifier = "{}--@--{}".format(robot_input_name, vial)

            ligand_reactant, solvent_reactant, nc_reactant = None, None, None
            for reagent_index, reactant in reagent_index_to_reactant.items():
                volume = record[reagent_index]
                if volume < 1e-7:
                    continue
                actual_reactant = deepcopy(reactant)
                actual_reactant.volume = volume
                reactant_def = actual_reactant.properties["definition"]
                if reactant_def == "nc":
                    nc_reactant = actual_reactant
                elif reactant_def == "ligand_solution":
                    ligand_reactant = actual_reactant
                elif reactant_def == "solvent":
                    solvent_reactant = actual_reactant
                else:
                    raise ValueError("wrong definition: {}".format(reactant_def))

            reaction = ReactionOneLigand(
                identifier=identifier, conditions=reaction_conditions, solvent=solvent_reactant,
                nc_solution=nc_reactant, ligand_solution=ligand_reactant, properties=None,
            )
            reactions.append(reaction)
        return reactions
