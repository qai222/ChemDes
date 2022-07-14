import pandas as pd
from sklearn.metrics import pairwise_distances
from lsal.campaign.loader import LoaderLigandDescriptors
from lsal.utils import json_dump, json_load, FilePath, scale_df

_distance_metric = "manhattan"



def prepare_dimred_data(inv_json: FilePath, des_csv: FilePath):
    inventory = json_load(inv_json)
    des_loader = LoaderLigandDescriptors("des_loader")
    ligand_to_desrecords = des_loader.load_file(des_csv, inventory=inventory)
    df = pd.DataFrame.from_records([r for r in ligand_to_desrecords.values()])
    df = scale_df(df)
    return inventory, df


if __name__ == '__main__':

    molecules, data_df = prepare_dimred_data("ligand_inventory.json", "../MolDescriptors/ligand_descriptors_2022_06_16_expka.csv")

    distance_matrix = pairwise_distances(data_df.values, metric=_distance_metric)

    from lsal.tasks.dimred import umap_run, tune_umap
    # tune_umap(
    #     distance_matrix,
    #     n_neighbors_values=(3, 5, 7, 9),
    #     min_dist_values=(0.1, 0.2, 0.3, 0.5),
    #     wdir="./dimred",
    # )  # nn7 md0.2 looks ok

    data_2d = umap_run(distance_matrix, nn=7, md=0.2, wdir="./")

    dimred_data = {
        "dmat": distance_matrix,
        "data_2d": data_2d,
        "molecules": molecules,
    }
    json_dump(dimred_data, "dimred.json")
