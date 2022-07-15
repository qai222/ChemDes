from lsal.campaign.loader import LigandExchangeReaction, Molecule
from lsal.campaign.loader import ReactionCollection
from lsal.utils import json_load, json_dump, get_basename

experiment_input_files = [
    "../data/SL_0519/2022_0519_SL01_0008_0015_robotinput.xls",
    "../data/SL_0519/2022_0519_SL02_0007_0012_robotinput.xls",
    "../data/SL_0519/2022_0520_SL03_0017_0004_robotinput.xls",
    "../data/SL_0519/2022_0520_SL04_0006_0009_robotinput.xls",
    "../data/SL_0519/2022_0520_SL05_0018_0014_robotinput.xls",
    "../data/SL_0519/2022_0520_SL06_0003_0010_robotinput.xls",
    "../data/SL_0519/2022_0520_SL07_0013_0021_robotinput.xls",
    "../data/SL_0519/2022_0520_SL08_0023_0000_robotinput.xls",
    "../data/SL_0519/2022_0525_SL09_0000_0001_robotinput.xls",
    "../data/SL_0519/2022_0525_SL10_0020_0002_robotinput.xls",
    "../data/SL_0519/2022_0525_SL11_0005_0022_robotinput.xls",
]

experiment_output_files = [
    "../data/SL_0519/PS0519_SL01_peakInfo.csv",
    "../data/SL_0519/PS0519_SL02_peakInfo.csv",
    "../data/SL_0519/PS0520_SL03_peakInfo.csv",
    "../data/SL_0519/PS0520_SL04_peakInfo.csv",
    "../data/SL_0519/PS0520_SL05_peakInfo.csv",
    "../data/SL_0519/PS0520_SL06_peakInfo.csv",
    "../data/SL_0519/PS0520_SL07_peakInfo.csv",
    "../data/SL_0519/PS0520_SL08_peakInfo.csv",
    "../data/SL_0519/PS0525_SL09_peakInfo.csv",
    "../data/SL_0519/PS0525_SL10_peakInfo.csv",
    "../data/SL_0519/PS0525_SL11_peakInfo.csv",
]


def exclude_reaction(r: LigandExchangeReaction):
    # ligand 0000 in SL08 should be excluded
    if not r.is_reaction_real:
        return False
    ligand = r.ligands[0]
    ligand: Molecule
    if "SL08" in r.identifier and ligand.int_label == 0:
        print("exclude reaction: {}".format(r.identifier))
        return True
    return False


if __name__ == '__main__':
    ligand_inventory = json_load("../../Inventory/ligand_inventory.json")
    solvent_inventory = json_load("../../Inventory/solvent_inventory.json")

    reaction_collection = ReactionCollection.from_files(
        experiment_input_files=experiment_input_files,
        experiment_output_files=experiment_output_files,
        ligand_inventory=ligand_inventory,
        solvent_inventory=solvent_inventory,
    )

    reaction_collection = ReactionCollection(
        [r for r in reaction_collection.reactions if not exclude_reaction(r)],
        reaction_collection.properties
    )

    json_dump(reaction_collection, "../data/" + get_basename(__file__) + ".json")
