from lsal.schema import Molecule
from lsal.tasks.io import load_raw_ligand_inventory
from lsal.utils import json_dump

if __name__ == "__main__":
    ligand_molecules = load_raw_ligand_inventory("../Raw/2022_0217_ligand_InChI_mk.xlsx")
    df_inventory = Molecule.write_molecules(ligand_molecules, "ligand_inventory.csv", output="csv")
    json_dump(ligand_molecules, "ligand_inventory.json")
