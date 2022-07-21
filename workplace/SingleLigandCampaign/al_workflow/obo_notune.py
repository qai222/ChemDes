import time

from lsal.campaign import run_wf
from lsal.campaign.fom import FomCalculator
from lsal.campaign.loader import LoaderLigandDescriptors, ReactionCollection
from lsal.utils import json_load, pkl_dump

inventory = json_load("../../Inventory/ligand_inventory.json")
des_loader = LoaderLigandDescriptors("des_loader")
ligand_to_desrecords = des_loader.load_file("../../MolDescriptors/ligand_descriptors_2022_06_16_expka.csv",
                                            inventory=inventory)
reaction_collection_0519 = json_load("../data/collect_reactions_SL_0519.json")
reaction_collection_0519: ReactionCollection
FomCalculator(reaction_collection_0519).update_foms()

available_metric = ("std", "mu-top2%", "std-top2%mu")
available_fom = ("fom1", "fom2", "fom3")

tuning = False
obo = True
split_in_tune = True
if __name__ == '__main__':
    for fom_type in available_fom:
        for obo_metric in available_metric:
            ts1 = time.perf_counter()
            swf = run_wf(
                fom_type, obo_metric,
                loaded_reaction_collection=reaction_collection_0519,
                ligand_to_desrecords=ligand_to_desrecords,
                tune=tuning, one_by_one=obo, tune_split=split_in_tune,
            )
            ts2 = time.perf_counter()
            print("{} : {:.4f} s".format(swf.swf_name, ts2 - ts1))
            pkl_dump(swf, "models/{}.pkl".format(swf.swf_name))
