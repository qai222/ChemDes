from lsal.schema import Molecule
from lsal.campaign import LoaderInventory
from lsal.utils import json_dump

if __name__ == "__main__":

    ligand_loader = LoaderInventory("ligand", "LIGAND")
    ligands = ligand_loader.load("2022_0217_ligand_InChI_mk.xlsx")

    solvent_loader = LoaderInventory("solvent", "SOLVENT")
    solvents = solvent_loader.load("2022_0217_solvent_InChI.csv")

    df_ligand_inventory = Molecule.write_molecules(ligands, "ligand_inventory.csv", output="csv")
    df_solvent_inventory = Molecule.write_molecules(solvents, "solvent_inventory.csv", output="csv")

    json_dump(ligands, "ligand_inventory.json")
    json_dump(solvents, "solvent_inventory.json")
