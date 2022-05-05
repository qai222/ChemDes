import logging

import pandas as pd
from skopt import BayesSearchCV
from skopt import load, dump

from lsal.tasks.slc import Molecule
from lsal.tasks.tune import train_twin_rf_with_tuned_params
from lsal.utils import df_all_numbers


def lolo_input(df_ligands: list[Molecule], df_X: pd.DataFrame, df_y: pd.DataFrame, ) -> list[dict]:
    assert df_all_numbers(df_X)
    assert df_all_numbers(df_y)
    assert df_X.shape[0] == df_y.shape[0] == len(df_ligands)
    assert len(df_y.dropna()) == len(df_y)

    unique_ligand_to_indices = dict()
    for iligand, ligand in enumerate(df_ligands):
        if ligand not in unique_ligand_to_indices:
            unique_ligand_to_indices[ligand] = [iligand, ]
        else:
            unique_ligand_to_indices[ligand].append(iligand)
    lolo_input_data = []

    for leftout_ligand, leftout_indices in unique_ligand_to_indices.items():
        df_X_leftout = df_X.iloc[leftout_indices, :]
        df_y_leftout = df_y.iloc[leftout_indices]
        remain_indices = [i for i in range(df_X.shape[0]) if i not in leftout_indices]
        assert len(remain_indices) + len(leftout_indices) == df_X.shape[0]
        df_X_remain = df_X.iloc[remain_indices, :]
        df_y_remain = df_y.iloc[remain_indices]
        data = dict(
            leftout_ligand=leftout_ligand,
            df_X_leftout=df_X_leftout,
            df_y_leftout=df_y_leftout,
            df_X_remain=df_X_remain,
            df_y_remain=df_y_remain,
        )
        lolo_input_data.append(data)
    return lolo_input_data


def lolo_calculate(lolo_input_data: list[dict], predictor_params: dict):
    lolo_calculate_data = dict()
    for data in lolo_input_data:
        leftout_ligand = data["leftout_ligand"]
        logging.warning("calculate leaving out: {}".format(leftout_ligand))
        df_X_leftout = data["df_X_leftout"]
        df_y_leftout = data["df_y_leftout"]
        df_X_remain = data["df_X_remain"]
        df_y_remain = data["df_y_remain"]
        reg = train_twin_rf_with_tuned_params(df_X_remain.values, df_y_remain.values, predictor_params)
        y_pred, y_pred_uncertainty = reg.twin_predict(df_X_leftout.values)
        y_real = df_y_leftout.values
        amounts = df_X_leftout["ligand_amount"].values
        cal_data = dict(
            y_real=y_real,
            y_pred=y_pred,
            y_pred_uncertainty=y_pred_uncertainty,
            amounts=amounts,
            reg=reg,
        )
        cal_data.update(data)
        lolo_calculate_data[leftout_ligand] = cal_data
    return lolo_calculate_data


if __name__ == '__main__':
    data = load("output/step4.pkl")
    ligand_inventory = data["step1"]["ligand_inventory"]
    df_ligands = data["step1"]["df_X_ligands"]
    df_X = data["step1"]["df_X"]
    df_y = data["step1"]["df_y"]
    opt = data["step2"]["opt"]
    opt: BayesSearchCV
    input_data = lolo_input(df_ligands, df_X, df_y)
    lolo_data = lolo_calculate(input_data, opt.best_params_)
    data["step5"] = {"lolo_output": lolo_data, "lolo_input": input_data}
    dump(data, "output/step5.pkl")
