import logging
import pathlib
import typing

from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

from chemdes.schema import pd, Molecule


def load_molecular_descriptors(fn: typing.Union[pathlib.Path, str], warning=False):
    if warning:
        logging.warning("loading file: {}".format(fn))
    molecules = []
    df = pd.read_csv(fn)
    assert not df.isnull().values.any()
    for r in df.to_dict("records"):
        inchi = r["InChI"]
        iupac_name = r["IUPAC Name"]
        mol = Molecule.from_str(inchi, "i", iupac_name)
        molecules.append(mol)
    df = df[[c for c in df.columns if c not in ["InChI", "IUPAC Name"]]]
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
