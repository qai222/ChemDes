import logging

import matplotlib.pyplot as plt
import pandas as pd
from skopt import BayesSearchCV
from skopt import load

from chemdes.one_ligand import Molecule
from chemdes.twinsk.estimator import TwinRegressor


def ligands_from_df(df: pd.DataFrame) -> list[Molecule]:
    assert "ligand_inchi" in df.columns
    assert "ligand_iupac_name" in df.columns
    ligands = []
    for r in df.to_dict("records"):
        l = Molecule.from_str(r["ligand_inchi"], iupac_name=r["ligand_iupac_name"])
        ligands.append(l)
    return ligands


def vis_ligand(l: Molecule, X, y, X_train, y_train, X_test, y_test, ligands):
    indices = [i for i in range(len(ligands)) if ligands[i] == l]

    X_train_indices = sorted(set(indices).intersection(set(X_train.index)))
    y_train_indices = sorted(set(indices).intersection(set(y_train.index)))
    X_test_indices = sorted(set(indices).intersection(set(X_test.index)))
    y_test_indices = sorted(set(indices).intersection(set(y_test.index)))

    X_train = X.values[X_train_indices, :]
    y_train = y.values[y_train_indices]
    y_train_pred, y_train_uncertainty = reg.twin_predict(X_train)
    X_test = X.values[X_test_indices, :]
    y_test = y.values[y_test_indices]
    y_test_pred, y_test_uncertainty = reg.twin_predict(X_test)

    X_train_amount = X.iloc[X_train_indices, :]["ligand_amount"].tolist()
    X_test_amount = X.iloc[X_test_indices, :]["ligand_amount"].tolist()

    fig, (ax1, ax2) = plt.subplots(nrows=2)

    # training
    ax1.scatter(X_train_amount, y_train, marker="x", c="k", label="Train: Real")
    ax1.errorbar(X_train_amount, y_train_pred, yerr=3 * y_train_uncertainty, fmt="o", c="r",
                 label=r"Train: Prediction ($3\sigma$)", alpha=0.5)
    ax1.set_title("Ligand: {}".format(l.iupac_name))
    ax1.set_xticks([])
    ax1.legend()
    ax1.set_ylim([-0.2, 1.8])
    ax1.set_xlim([0, 23])
    ax1.set_ylabel("PL Figure of Merit (a.u.)")

    # test
    ax2.scatter(X_test_amount, y_test, marker="x", c="k", label="Test: Real")
    ax2.errorbar(X_test_amount, y_test_pred, yerr=3 * y_test_uncertainty, fmt="o", c="b",
                 label=r"Test: Prediction ($3\sigma$)", alpha=0.5)
    ax2.set_ylim([-0.2, 1.8])
    ax2.legend()
    ax2.set_xlim([0, 23])
    ax2.set_xlabel(r"Ligand Amount $(\mu M \times L)$")
    ax2.set_ylabel("PL Figure of Merit (a.u.)")
    fig.subplots_adjust(hspace=0.1)
    fig.savefig("vis_cv/{}.png".format(l.iupac_name))


if __name__ == '__main__':

    data = load("output/tune-data.pkl")
    opt = data["opt"]
    opt: BayesSearchCV
    logging.warning("best params:")
    logging.warning(opt.best_params_)
    reg = opt.best_estimator_
    reg: TwinRegressor

    X_ligands, X, y, X_train, y_train, X_test, y_test = [data[key] for key in
                                                         ["X_ligands", "X", "y", "X_train", "y_train", "X_test",
                                                          "y_test"]]

    ligands = ligands_from_df(X_ligands)
    unique_ligands = sorted(set(ligands))
    for l in unique_ligands:
        vis_ligand(l, X, y, X_train, y_train, X_test, y_test, ligands)
