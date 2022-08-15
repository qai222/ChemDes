import math
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from lsal.utils import FilePath, json_dump, createdir, smiles2inchi

sns.set_style("whitegrid")


def domain_range(lib_descriptor_csv: FilePath, features: list[str]):
    df = pd.read_csv(lib_descriptor_csv)
    df = df[features]
    lim = dict()
    for f in features:
        lim[f] = (min(df[f]), max(df[f]))
    return lim


def delta_feature_screen(delta: float, feature_lim_dict: dict, smis: list[str], feature_df: pd.DataFrame,
                         screen_features: list[str], logger=None):
    df = feature_df
    assert len(smis) == df.shape[0]
    screened_records = []
    screened_smis = []

    current_feature_lim_dict = {k: (v[0] * (1 - delta), v[1] * (1 + delta)) for k, v in feature_lim_dict.items()}
    if logger is not None:
        logger.info(str(current_feature_lim_dict))
    for smi, r in tqdm(zip(smis, df.to_dict(orient="records"))):
        in_domain = True
        for f in screen_features:
            fmin, fmax = current_feature_lim_dict[f]
            v = r[f]
            if not fmin <= v <= fmax:
                in_domain = False
                break
        if in_domain:
            screened_records.append(r)
            screened_smis.append(smi)
    return screened_records, screened_smis, smis



def smi2poolinv(smis):
    records = []
    for i, smi in tqdm(enumerate(smis)):
        label = "POOL-{0:08d}".format(i)
        inchi = smiles2inchi(smi)
        r = {
            "label": label,
            "smiles": smi,
            "identifier": inchi,
        }
        records.append(r)
    return pd.DataFrame.from_records(records)


def delta_plot(
        initial_smis, domain_lim, feature_df, out: FilePath,
        available_features: list[str], xmin=1e-5, xmax=0.8, nxs=9, wdir: FilePath = "./", logger=None,
):
    xs = np.linspace(xmin, xmax, nxs)
    ys = []
    createdir(wdir)
    for delta in xs:
        rs, smis, all_smis = delta_feature_screen(delta, domain_lim, initial_smis, feature_df, available_features, logger)
        delta_str = "{:.2f}".format(delta)
        delta_data = rs, smis
        json_dump(delta_data, os.path.join(wdir, delta_str + ".json"))
        ys.append(len(rs))
    fig, ax = plt.subplots()
    ax.plot(xs, ys, "ro-", label="# in domain")
    # ax.set_yscale("log")
    minmax_x = xmax - xmin
    ax.hlines(len(all_smis), xmin-0.05*minmax_x, xmax+0.05*minmax_x, colors=["k"], linestyles="dotted", label="# total")
    ax.set_xlim([xmin - 0.05 * minmax_x, xmax + 0.05 * minmax_x])

    ax.set_ylabel("# of molecules")
    ax.set_xlabel(r"$\Delta ({\mathrm{Feature}})$")
    ax.legend(loc="lower right")
    fig.savefig(out, dpi=600)

# smi 2 record
def get_smi2record(smis: list[str], df1: pd.DataFrame, df2: pd.DataFrame or None):
    smi2recrod = dict()
    if df2 is None:
        bunch = zip(smis, df1.to_dict(orient="records"), [{} for _ in range(len(smis))])
    else:
        bunch = zip(smis, df1.to_dict(orient="records"), df2.to_dict(orient="records"))
    for smi, r1, r2 in bunch:
        r = dict()
        r.update(r1)
        r.update(r2)
        smi2recrod[smi] = r
    return smi2recrod
