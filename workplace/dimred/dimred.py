from sklearn.metrics import pairwise_distances
from lsal.tasks.preprocess import load_molecular_descriptors, preprocess_descriptor_df
from lsal.tasks.dimred import umap_run, tune_umap
from lsal.utils import json_dump

_distance_metric = "manhattan"

if __name__ == '__main__':
    molecules, df = load_molecular_descriptors("../ligand_descriptors/molecular_descriptors_2022_03_21.csv")
    data_df = preprocess_descriptor_df(df, scale=True, vthreshould=True)

    distance_matrix = pairwise_distances(data_df.values, metric=_distance_metric)

    tune_umap(distance_matrix, wdir="./")  # nn7 md0.2 looks ok

    data_2d = umap_run(distance_matrix, nn=7, md=0.2, wdir="./")

    dimred_data = {
        "dmat": distance_matrix,
        "data_2d": data_2d,
        "molecules": molecules,
    }
    json_dump(dimred_data, "dimred.json")
