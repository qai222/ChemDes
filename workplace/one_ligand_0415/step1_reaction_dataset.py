import logging
import os

import pandas as pd

from lsal.one_ligand import ReactionNcOneLigand, Molecule, ReactionCondition, ReactantSolvent, ReactantSolution, \
    categorize_reactions
from lsal.schema import load_inventory, molecule_from_label
from lsal.utils import strip_extension, json_dump, FilePath, padding_vial_label

this_dir = os.path.abspath(os.path.dirname(__file__))

_DefaultRobotInputColumns = ['Vial Site', 'Reagent1 (ul)', 'Reagent2 (ul)', 'Reagent3 (ul)', 'Reagent4 (ul)',
                             'Reagent5 (ul)', 'Reagent6 (ul)', 'Reagent7 (ul)', 'Reagent8 (ul)', 'Reagent9 (ul)',
                             'Reagent10 (ul)', 'Labware ID:', 'Reaction Parameters', 'Parameter Values', 'Reagents',
                             'Reagent Name', 'Reagent Identity', 'Reagent Concentration (uM)', 'Liquid Class', ]
_VialCol = _DefaultRobotInputColumns[0]
_LigandConcentrationUnit = "uM"
_VolumeUnit = "ul"
_DefaultSolvents = ["m-xylene", ]
_LigandInventory = load_inventory("../inventory/inventory.csv", to_mols=True)


def load_robot_input(f: FilePath, ):
    robot_input_name = strip_extension(os.path.basename(f))

    robotinput_df = pd.read_csv(f)
    robotinput_df.drop(robotinput_df.filter(regex="Unnamed"), axis=1, inplace=True)
    robotinput_df.dropna(axis=0, inplace=True, how="all")
    assert set(robotinput_df.columns) == set(_DefaultRobotInputColumns)

    # load general coniditions
    condition_df = robotinput_df.loc[:, ["Reaction Parameters", "Parameter Values"]]
    condition_df = condition_df.dropna(axis=0, how="all")
    reaction_conditions = []
    for k, v in zip(condition_df.iloc[:, 0], condition_df.iloc[:, 1]):
        rc = ReactionCondition(k, v)
        reaction_conditions.append(rc)

    # load reagents
    reagent_df = robotinput_df.loc[:,
                 ["Reagents", "Reagent Name", "Reagent Identity", "Reagent Concentration (uM)"]
                 ]
    reagent_df.dropna(axis=0, inplace=True, how="all")
    reagent_col_name_to_molecule = dict()
    ligand_col_name_to_concentration = dict()
    nanocrystal = None
    solvent_identity = None
    solvent_col_name = None
    nanocrystal_col_name = None
    ligand_col_names = []
    for record in reagent_df.to_dict(orient="records"):
        reagent_col_name = record["Reagents"] + " (ul)"
        assert reagent_col_name in robotinput_df.columns
        reagent_chemical_name = record["Reagent Name"]
        reagent_chemical_label = record["Reagent Identity"]
        reagent_chemical_concentration = record["Reagent Concentration (uM)"]
        if reagent_chemical_name.startswith("CPB"):
            nanocrystal = reagent_chemical_name
            reagent_col_name_to_molecule[reagent_col_name] = nanocrystal
            nanocrystal_col_name = reagent_col_name
        elif pd.isnull(reagent_chemical_concentration) and reagent_chemical_name.lower() in _DefaultSolvents:
            solvent_identity = reagent_chemical_name
            reagent_col_name_to_molecule[reagent_col_name] = solvent_identity
            solvent_col_name = reagent_col_name
        else:
            reagent_chemical_label = int(reagent_chemical_label)
            mol = molecule_from_label(reagent_chemical_label, _LigandInventory)
            reagent_col_name_to_molecule[reagent_col_name] = mol
            ligand_col_names.append(reagent_col_name)
            ligand_col_name_to_concentration[reagent_col_name] = reagent_chemical_concentration
    assert nanocrystal is not None
    assert solvent_identity is not None
    assert solvent_col_name is not None
    assert nanocrystal_col_name is not None

    # load volumes
    reaction_df = robotinput_df.loc[:, ["Vial Site", ] + list(reagent_col_name_to_molecule.keys())]
    reactions = []
    for record in reaction_df.to_dict(orient="records"):
        # vial
        vial = padding_vial_label(record[_VialCol])
        identifier = robot_input_name + "--" + vial

        # solvent
        solvent_volume = record[solvent_col_name]
        solvent = ReactantSolvent(solvent_identity, solvent_volume)

        # nanocrystal
        nc_solution_volume = record[nanocrystal_col_name]
        nc = ReactantSolution(nanocrystal, volume=nc_solution_volume, solvent_identity=solvent_identity,
                              volume_unit=_VolumeUnit,
                              concentration=None, properties={}, concentration_unit=None)

        # ligand solution -- one ligand
        ligand_solution_volume = 0.0
        ligand_concentration = 0.0
        ligand_col = None
        for ligand_col in ligand_col_names:
            ligand_solution_volume = record[ligand_col]
            if ligand_solution_volume > 1e-7:
                ligand_concentration = ligand_col_name_to_concentration[ligand_col]
                break
        assert ligand_col is not None
        # skip invalid entries
        if solvent_volume < 1e-7 and nc_solution_volume < 1e-7 and ligand_solution_volume < 1e-7:
            continue
        # control reactions
        if ligand_concentration < 1e-7 and ligand_solution_volume < 1e-7:
            ligand_mol = Molecule(inchi=robot_input_name)
        else:
            ligand_mol = reagent_col_name_to_molecule[ligand_col]

        ligand = ReactantSolution(
            ligand_mol,
            ligand_solution_volume,
            concentration=ligand_concentration,
            solvent_identity=solvent_identity,
            volume_unit=_VolumeUnit,
            concentration_unit=_LigandConcentrationUnit
        )
        reaction = ReactionNcOneLigand(identifier, nc, ligand, reaction_conditions, solvent,
                                       properties={"vial": vial, "robotinput_file": os.path.basename(f)})
        reactions.append(reaction)
    return reactions, robot_input_name


def categorized_reactions_warning(data: dict):
    for l in data:
        logging.warning(">> ligand: {}".format(l))
        real, ref, blank = data[l]
        logging.warning("# of real reactions: {}".format(len(real)))
        logging.warning("# of ref reactions: {}".format(len(ref)))
        logging.warning("# of blank reactions: {}".format(len(blank)))


def load_peaks(peak_file: FilePath, identifier_prefix="") -> dict[str, dict]:
    peak_df = pd.read_csv(peak_file)
    peak_df.drop(peak_df.filter(regex="Unnamed"), axis=1, inplace=True)
    peak_df.dropna(axis=0, inplace=True, how="all")
    vial_col = [c for c in peak_df.columns if "wellLabel" in c][0]
    fom_col = [c for c in peak_df.columns if "FOM" in c][0]
    data = dict()
    for record in peak_df.to_dict(orient="records"):
        vial = identifier_prefix + "--" + padding_vial_label(record[vial_col])
        fom = record[fom_col]
        data[vial] = {"fom": fom}
        data[vial]["peak_file"] = os.path.basename(peak_file)
        data[vial].update(record)
    return data


def assign_outcome(peak_data: dict, reactions: list[ReactionNcOneLigand]):
    assert len(peak_data) == len(reactions)
    for i in range(len(reactions)):
        r = reactions[i]
        data = peak_data[r.identifier]
        r.properties.update(data)


def load_robotinput_and_peak_info(robotinput_file: FilePath, peak_file: FilePath):
    reactions, identifier_prefix = load_robot_input(robotinput_file)
    peak_data = load_peaks(peak_file, identifier_prefix=identifier_prefix)
    assign_outcome(peak_data, reactions)
    return reactions


if __name__ == '__main__':
    reactions = []

    reactions += load_robotinput_and_peak_info(
        robotinput_file="data/2022_0415_LS001_0021_0001_tip_sorted_robotinput.csv",
        peak_file="data/0415_LS001_peakInfo.csv"
    )
    reactions += load_robotinput_and_peak_info(
        robotinput_file="data/2022_0415_LS002_0014_0009_tip_sorted_robotinput.csv",
        peak_file="data/0415_LS002_peakInfo.csv"
    )
    reactions += load_robotinput_and_peak_info(
        robotinput_file="data/2022_0415_LS003_0008_0020_tip_sorted_robotinput.csv",
        peak_file="data/0415_LS003_peakInfo.csv"
    )
    ligand_to_categorized_reactions = categorize_reactions(reactions)
    categorized_reactions_warning(ligand_to_categorized_reactions)
    json_dump(reactions, "output/reactions.json")
    json_dump({k.__repr__(): v for k, v in ligand_to_categorized_reactions.items()},
              "output/reactions_data.json")
