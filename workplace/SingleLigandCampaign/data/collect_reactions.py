import glob
import os.path
from typing import Callable

import pandas as pd

from lsal.campaign.loader import LigandExchangeReaction, Molecule
from lsal.campaign.loader import ReactionCollection
from lsal.campaign.loader import load_reactions_from_expt_files
from lsal.utils import json_load, json_dump, get_basename, FilePath, file_exists

LigandInventory = json_load("../../Inventory/ligand_inventory.json")
SolventInventory = json_load("../../Inventory/solvent_inventory.json")


def collect_reactions(
        folder: FilePath,
        ligand_inventory: list[Molecule],
        solvent_inventory: list[Molecule],
        exclude_reaction: Callable
) -> ReactionCollection:
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
    reaction_collection = load_reactions_from_expt_files(
        experiment_input_files=experiment_input_files,
        experiment_output_files=experiment_output_files,
        ligand_inventory=ligand_inventory,
        solvent_inventory=solvent_inventory,
    )
    return ReactionCollection(
        [r for r in reaction_collection.reactions if not exclude_reaction(r)],
        reaction_collection.properties
    )


def exclude_reaction_in_SL_0519(r: LigandExchangeReaction):
    # ligand 0000 in SL08 should be excluded
    if not r.is_reaction_real:
        return False
    ligand = r.ligands[0]
    ligand: Molecule
    if "SL08" in r.identifier and ligand.int_label == 0:
        print("exclude reaction: {}".format(r.identifier))
        return True
    return False


def collect_all(folders: list[FilePath]):
    for folder in folders:
        if get_basename(folder) == "SL_0519":
            reaction_collection = collect_reactions(folder, LigandInventory, SolventInventory,
                                                    exclude_reaction_in_SL_0519)
        else:
            reaction_collection = collect_reactions(folder, LigandInventory, SolventInventory, lambda x: False)
        json_dump(reaction_collection, f"{get_basename(__file__)}_{get_basename(folder)}.json")


if __name__ == '__main__':
    collect_all(sorted(glob.glob("*/")))
