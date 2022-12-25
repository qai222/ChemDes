import glob

import pandas as pd
from loguru import logger

from lsal.schema import L1XReactionCollection
from lsal.utils import json_load, get_basename, FilePath

"""
make sure ligands used in AL expt campaign are in the model suggestion list
"""


def verify(reaction_collection_json: FilePath, suggestion_csv_folder: FilePath):
    sugg_dict = dict()
    for sugg_csv in glob.glob(f"{suggestion_csv_folder}/suggestion__*.csv"):
        bname = get_basename(sugg_csv)
        ids = pd.read_csv(sugg_csv)["ligand_identifier"].tolist()
        for i in ids:
            if i not in sugg_dict:
                sugg_dict[i] = bname
            else:
                sugg_dict[i] += " " + bname
    rc = json_load(reaction_collection_json)
    rc: L1XReactionCollection
    for lig in rc.unique_ligands:
        try:
            print(sugg_dict[lig.identifier])
        except KeyError:
            logger.critical(f"ligand not found!: {lig.label}\n\t{lig.identifier}")


if __name__ == '__main__':
    verify(
        "reaction_collection_AL1026.json.gz",
        "../learning_SL0519/suggestion"
    )
    verify(
        "reaction_collection_AL1213.json.gz",
        "../learning_AL1026/suggestion"
    )
