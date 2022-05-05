from skopt import load, dump

from lsal.tasks.io import load_ligand_to_descriptors
from lsal.tasks.slc import SingleLigandCampaign
from lsal.utils import json_load

if __name__ == '__main__':
    last_round_SLC = load("../SL_0415/output/step1.pkl")["step1"]["SLC"]
    last_round_SLC: SingleLigandCampaign
    LigandInventory = json_load("../../LigandInventory/ligand_inventory.json")
    Ligand_to_Desrecord = load_ligand_to_descriptors(
        fn="../../LigandInventory/ligand_descriptors_2022_03_21.csv",
        inventory=LigandInventory
    )

    experiment_input_files = [
        "data/2022_0421_SL1_0004_0000_robotinput.csv",
        "data/2022_0421_SL2_0018_0012_robotinput.csv",
        "data/2022_0422_SL3_0006_0007_robotinput.csv",
        "data/2022_0422_SL4_0002_0003_robotinput.csv",
        "data/2022_0422_SL5_0015_0022_robotinput.csv",
    ]

    experiment_output_files = [
        "data/0421_SL1_peakInfo.csv",
        "data/0421_SL2_peakInfo.csv",
        "data/0421_SL3_peakInfo.csv",
        "data/0421_SL4_peakInfo_fixed.csv",
        "data/0421_SL5_peakInfo.csv",
    ]

    SLC = SingleLigandCampaign.from_files(
        name="SL_0421",
        experiment_input_files=experiment_input_files + last_round_SLC.experiment_input_files,
        experiment_output_files=experiment_output_files + last_round_SLC.experiment_output_files,
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
