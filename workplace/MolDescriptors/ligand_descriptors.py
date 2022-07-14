import datetime

import pandas as pd

from lsal.schema import Molecule
from lsal.tasks.descal import calculate_cxcalc, calculate_mordred, opera_pka
from lsal.utils import json_load, file_exists

if __name__ == '__main__':

    mols_file = "ligand_descriptors.smi"
    mols = json_load("../Inventory/ligand_inventory.json")
    if not file_exists(mols_file):
        Molecule.write_molecules(mols, mols_file, "smi")  # write smi file
    mordred_df = calculate_mordred(smis=[m.smiles for m in mols])
    cxcalc_df = calculate_cxcalc(mol_file=mols_file)
    pka_df = opera_pka("ligand_descriptors_OPERA2.7Pred.csv")

    des_df = pd.concat([pka_df, cxcalc_df, mordred_df], axis=1)
    des_df["InChI"] = [m.inchi for m in mols]
    des_df["IUPAC Name"] = [m.iupac_name for m in mols]
    des_df.to_csv("ligand_descriptors_{}.csv".format(datetime.datetime.now().strftime("%Y_%m_%d")), index=False)
