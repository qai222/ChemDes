import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt import load

from chemdes.one_ligand import Molecule
from chemdes.tasks.preprocess import preprocess_descriptor_df, load_descriptors_and_fom
from chemdes.twinsk.estimator import TwinRegressor
from chemdes.utils import SEED


def ligands_from_df(df: pd.DataFrame) -> list[Molecule]:
    assert "ligand_inchi" in df.columns
    assert "ligand_iupac_name" in df.columns
    ligands = []
    for r in df.to_dict("records"):
        l = Molecule.from_str(r["ligand_inchi"], iupac_name=r["ligand_iupac_name"])
        ligands.append(l)
    return ligands


def unique_ligand_to_indices(df: pd.DataFrame):
    ligands = ligands_from_df(df)
    unique_ligands = sorted(set(ligands))
    data = dict()
    for l in unique_ligands:
        indices = np.where(df["ligand_inchi"] == l.inchi)[0].tolist()
        data[l] = indices
    return unique_ligands, data


def load_opt():
    opt_data = load("output/tune-data.pkl")
    opt = opt_data["opt"]
    opt: BayesSearchCV
    logging.warning("best params:")
    logging.warning(opt.best_params_)
    return opt


def lolo(df_X_labelled: pd.DataFrame, df_y_labelled: pd.DataFrame, opt: BayesSearchCV):
    # rm low var cols
    unique_ligands, ligand_to_indices = unique_ligand_to_indices(df_X_labelled)
    df_X_labelled = preprocess_descriptor_df(df_X_labelled, scale=False, vthreshould=False)

    fig, total_axes = plt.subplots(nrows=len(unique_ligands), ncols=len(unique_ligands),
                                   figsize=(4 * len(unique_ligands), 3 * len(unique_ligands)), )
    # leave a ligand out, train the twin reg, check how well it predicts the left-out ligand
    for i in range(len(unique_ligands)):
        # define a left out ligand
        left_out_ligand = unique_ligands[i]
        train_ligands = [l for l in unique_ligands if l != left_out_ligand]

        # training data with n-1 ligands
        train_indices = []
        for train_ligand in train_ligands:
            train_indices += ligand_to_indices[train_ligand]
        X_train = df_X_labelled.iloc[train_indices].values
        y_train = df_y_labelled.iloc[train_indices].values

        # recreate regressor using tuned params
        reg = TwinRegressor(RandomForestRegressor(n_estimators=100, random_state=SEED))
        reg.set_params(**opt.best_params_)
        reg.fit(X_train, y_train)

        axes = total_axes[i]
        for iax, ax in enumerate(axes):
            current_ligand = unique_ligands[iax]
            current_indices = ligand_to_indices[current_ligand]
            amounts = df_X_labelled.iloc[current_indices]["ligand_amount"].values
            y_real = df_y_labelled.values[current_indices]
            y_pred, y_pred_uncertainty = reg.twin_predict(df_X_labelled.iloc[current_indices].values)

            ax.scatter(amounts, y_real, marker="x", c="k", label="Real")
            ax.errorbar(amounts, y_pred, yerr=3 * y_pred_uncertainty, fmt="o", c="r",
                        label=r"Prediction ($3\sigma$)", alpha=0.5)
            if current_ligand != left_out_ligand:
                ax.set_title("{}-Train: {}".format(i, current_ligand.iupac_name))
            else:
                ax.set_title("{}-Test: {}".format(i, current_ligand.iupac_name))
            ax.set_xlim([-1, 24])
            ax.set_ylim([-0.5, 2.0])
            if iax == 0:
                ax.legend()

        fig.subplots_adjust(hspace=0.6)
    fig.savefig("lolo.png")


if __name__ == '__main__':
    opt = load_opt()
    labelled_ligands, df_X_labelled, df_y_labelled = load_descriptors_and_fom(
        mdes_csv="../ligand_descriptors/molecular_descriptors_2022_03_21.csv",
        reactions_json="output/2022_0304_LS001_MK003_reaction_data.json",
    )
    lolo(df_X_labelled, df_y_labelled, opt)
