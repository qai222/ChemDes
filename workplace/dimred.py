import logging

import matplotlib.pyplot as plt
import umap
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances

from chemdes import *

SEED = 42
_distance_metric = "manhattan"

_inchi_to_iupac_name = {m.inchi: m.iupac_name for m in load_inventory("data/2022_0217_ligand_InChI_mk.xlsx")}


def load_molecular_descriptors(fn: typing.Union[pathlib.Path, str]):
    molecules = []
    df = pd.read_csv(fn)
    assert not df.isnull().values.any()
    for r in df.to_dict("records"):
        inchi = r["InChI"]
        mol = Molecule.from_str(inchi, "i", _inchi_to_iupac_name[inchi])
        molecules.append(mol)
    df = df[[c for c in df.columns if c != "InChI"]]
    return molecules, df


def preprocess_descriptor_df(data_df):
    x = data_df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data_df = pd.DataFrame(x_scaled, columns=data_df.columns, index=data_df.index)
    sel = VarianceThreshold(threshold=0.01)
    sel_var = sel.fit_transform(data_df)
    data_df = data_df[data_df.columns[sel.get_support(indices=True)]]
    return data_df


def plot2d(data_2d, saveas):
    plot_kwds = {'alpha': 0.5, 's': 80, 'linewidth': 0}
    plt.scatter(data_2d.T[0], data_2d.T[1], color='gray', **plot_kwds)
    plt.xticks([])
    plt.yticks([])
    plt.box(on=None)
    plt.savefig("{}.png".format(saveas))
    plt.clf()


def umap_run(dmat, nn, md, wdir="./"):
    saveas = "nn{}-md{}".format(nn, md)
    logging.warning("working on: {}".format(saveas))
    transformer = umap.UMAP(
        n_neighbors=nn, min_dist=md, metric="precomputed", random_state=SEED)
    data_2d = transformer.fit_transform(dmat)
    plot2d(data_2d, os.path.join(wdir, saveas))
    return data_2d


def tune_umap(
        dmat,
        n_neighbors_values=[3, 5, 7, ],
        min_dist_values=[0.1, 0.2, 0.3],
        wdir="./",
):
    for nn in n_neighbors_values:
        for md in min_dist_values:
            umap_run(dmat, nn, md, wdir)


if __name__ == '__main__':
    molecules, df = load_molecular_descriptors("data/molecular_descriptors_2022_03_09.csv")
    data_df = preprocess_descriptor_df(df)

    distance_matrix = pairwise_distances(data_df.values, metric=_distance_metric)

    # tune_umap(distance_matrix, wdir="./dimred")  # nn5 md0.1 looks ok

    data_2d = umap_run(distance_matrix, nn=5, md=0.1, wdir="./")

    dimred_data = {
        "dmat": distance_matrix,
        "data_2d": data_2d,
        "molecules": molecules,
    }
    json_dump(dimred_data, "dimred/dimred.json")
