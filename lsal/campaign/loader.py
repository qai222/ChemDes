from __future__ import annotations

import logging
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Tuple, List

import numpy as np
import pandas as pd
from loguru import logger

from lsal.schema import Molecule, ReactionCondition, ReactantSolution, NanoCrystal, _EPS, L1XReactionCollection, \
    featurize_molecules, L1XReaction
from lsal.utils import FilePath, padding_vial_label, strip_extension, get_extension, file_exists, get_basename


def load_molecules(
        fn: FilePath, col_to_mol_kw: dict[str, str], mol_type: str = 'LIGAND',
) -> list[Molecule]:
    mol_type = mol_type.upper()

    logger.info(f"LOADING: {mol_type} from {fn}")
    assert file_exists(fn)
    extension = get_extension(fn)
    if extension == "csv":
        df = pd.read_csv(fn)
    elif extension == "xlsx":
        ef = pd.ExcelFile(fn)
        assert len(ef.sheet_names) == 1, "there should be only one sheet in the xlsx file"
        df = ef.parse(ef.sheet_names[0])
    else:
        raise ValueError(f"extension not understood: {extension}")

    required_columns = col_to_mol_kw.keys()
    assert set(required_columns).issubset(set(df.columns)), f"csv does not have required columns: {required_columns}"

    df = df[required_columns]
    df = df.dropna(axis=0, how="all")

    assign_label = not 'label' in df.columns
    if assign_label:
        logger.info(f'we WILL assign labels based on row index and mol_type=={mol_type}')

    molecules = []
    mol_kws = ['identifier', 'iupac_name', 'name']
    for irow, row in enumerate(df.to_dict("records")):

        if assign_label:
            int_label = irow
        else:
            label = row['label']
            mol_type, int_label = label.split('-')
            int_label = int(int_label)

        mol_kwargs = dict(
            int_label=int_label,
            mol_type=mol_type,
            properties=OrderedDict({"load_from": get_basename(fn)})
        )

        for colname, value in row.items():
            # TODO parse nested properties
            try:
                mol_kw = col_to_mol_kw[colname]
                assert mol_kw in mol_kws
                mol_kwargs[mol_kw] = value
            except (AssertionError, KeyError) as e:
                pass
        m = Molecule(**mol_kwargs)
        molecules.append(m)
    return molecules


def load_featurized_molecules(
        inv_csv: FilePath,
        des_csv: FilePath,
        col_to_mol_kw: dict[str, str] = None,
        mol_type: str = 'LIGAND',
) -> list[Molecule]:
    # load inv csv
    molecules = load_molecules(inv_csv, col_to_mol_kw, mol_type)
    des_df = pd.read_csv(des_csv)
    assert des_df.shape[0] == len(molecules)
    assert not des_df.isnull().values.any()
    featurize_molecules(molecules, des_df)
    return molecules


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
        logging.info("loading reaction: {}".format(record))
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
            logging.info("Found nanocrystal: {}".format(reagent_name))
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


def load_reactions_from_expt_files(
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
        L1XReactionCollection.assign_reaction_results(reactions, peak_data)
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
    logging.warning("REACTIONS LOADED: blank/ref/real: {}/{}/{}".format(n_blank, n_ref, n_real))
    slc = L1XReactionCollection(reactions=all_reactions, properties=properties)
    slc.properties.update(
        dict(experiment_input_files=experiment_input_files, experiment_output_files=experiment_output_files,
             ligand_inventory=ligand_inventory, solvent_inventory=solvent_inventory, ))
    return slc
