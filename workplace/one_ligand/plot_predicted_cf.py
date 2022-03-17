import matplotlib.pyplot as plt
import pandas as pd
from chemdes.utils import sort_and_group


def plot_predictions(df:pd.DataFrame, xfield:str, yfield:str, title:str, x_unit):
    fig, ax = plt.subplots()
    x = df[xfield]
    y = df[yfield]

    ax.scatter(x, y, label="sample")
    ax.set_xlabel("Ligand Amount ({})".format(x_unit))
    ax.set_ylabel("Figure of Merit (a.u.)")
    ax.set_title(title)
    ax.legend()
    fig.savefig("output/{}.png".format(title), dpi=300)




prediction_df = pd.read_csv("RandomForestRegressor.csv")
ligand_iupac_names = sorted(set(prediction_df["ligand"].tolist()))
for name in ligand_iupac_names:
    ligand_df = prediction_df.loc[prediction_df['ligand'] == name]
    xfield = "amount"


    print(ligand_df)
