from sklearn.metrics import pairwise_distances
from chemdes.tasks.preprocess import load_molecular_descriptors, preprocess_descriptor_df
from chemdes.tasks.dimred import umap_run
from chemdes.utils import json_dump

_distance_metric = "manhattan"

if __name__ == '__main__':
    molecules, df = load_molecular_descriptors("../data/molecular_descriptors_2022_03_09.csv")
    data_df = preprocess_descriptor_df(df)

    distance_matrix = pairwise_distances(data_df.values, metric=_distance_metric)

    # tune_umap(distance_matrix, wdir="./")  # nn5 md0.1 looks ok

    data_2d = umap_run(distance_matrix, nn=5, md=0.1, wdir="./")

    dimred_data = {
        "dmat": distance_matrix,
        "data_2d": data_2d,
        "molecules": molecules,
    }
    json_dump(dimred_data, "dimred.json")
