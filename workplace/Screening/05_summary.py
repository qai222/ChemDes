import pandas as pd
from tqdm import tqdm

from lsal.utils import smiles2inchi, json_load, get_basename, createdir


def smi2inv(smis):
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


if __name__ == '__main__':
    screened_data = json_load("results/04_screen_cm/0.40.json")
    records, smis = screened_data

    df_inv = smi2inv(smis)
    df_des = pd.DataFrame.from_records(records)
    createdir("results/{}".format(get_basename(__file__)))
    df_inv.to_csv("results/{}/inv.csv".format(get_basename(__file__)), index=False)
    df_des.to_csv("results/{}/des.csv".format(get_basename(__file__)), index=False)
