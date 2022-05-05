from skopt import dump

from lsal.tasks.io import load_ligand_to_descriptors
from lsal.tasks.slc import SingleLigandCampaign
from lsal.utils import json_load

if __name__ == '__main__':
    LigandInventory = json_load("../../LigandInventory/ligand_inventory.json")
    Ligand_to_Desrecord = load_ligand_to_descriptors(
        fn="../../LigandInventory/ligand_descriptors_2022_03_21.csv",
        inventory=LigandInventory
    )

    experiment_input_files = [
        "data/2022_0415_LS001_0021_0001_tip_sorted_robotinput.csv",
        "data/2022_0415_LS002_0014_0009_tip_sorted_robotinput.csv",
        "data/2022_0415_LS003_0008_0020_tip_sorted_robotinput.csv",
    ]

    experiment_output_files = [
        "data/0415_LS001_peakInfo.csv",
        "data/0415_LS002_peakInfo.csv",
        "data/0415_LS003_peakInfo.csv",
    ]

    SLC = SingleLigandCampaign.from_files(
        name="SL_0415",
        experiment_input_files=experiment_input_files,
        experiment_output_files=experiment_output_files,
        additional_files=[], ligand_inventory=LigandInventory,
    )

    df_ligands, df_X, df_y = SLC.get_ml_input(
        ligand_to_desrecord=Ligand_to_Desrecord
    )

    step1_data = {
        "SLC": SLC,
        "df_X_ligands": df_ligands,
        "df_X": df_X,
        "df_y": df_y,
        "ligand_inventory": LigandInventory,
        "ligand_to_desrecord": Ligand_to_Desrecord,
    }
    data = {"step1": step1_data}
    dump(data, "output/step1.pkl")
