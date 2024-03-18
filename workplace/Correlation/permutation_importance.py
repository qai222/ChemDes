import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import r_regression
from sklearn.utils import check_random_state
from tqdm import tqdm

from lsal.alearn.one_ligand import SingleLigandLearner, SingleLigandPrediction
from lsal.schema import Molecule
from lsal.twinsk import TwinRegressor
from lsal.utils import json_load, json_dump, pkl_load


def actual_feature_names(fns: list[str]):
    afns = []
    for i in range(len(fns)):
        afns.append(f"p1@@{fns[i]}")
    for i in range(len(fns)):
        afns.append(f"p2@@{fns[i]}")
    for i in range(len(fns)):
        afns.append(f"DELTA@@{fns[i]}")
    return afns


def permutation_importance_(
        twin: TwinRegressor,
        col_idx: int,
        X: pd.DataFrame,
        n_repeats=5,
):
    rs = check_random_state(42)
    X_permuted = X.copy()
    shuffling_idx = np.arange(X_permuted.shape[0])
    for _ in range(n_repeats):
        rs.shuffle(shuffling_idx)
        col = X_permuted.iloc[shuffling_idx, col_idx]
        col.index = X_permuted.index
        X_permuted.iloc[:, col_idx] = col
    mu_permuted, std_permuted = twin.twin_predict(X_permuted.values)
    mu, std = twin.twin_predict(X.values)
    return float(np.mean(mu_permuted - mu)), float(np.mean(std_permuted - std)), mu, std


def permutation_importance(twin: TwinRegressor, X: pd.DataFrame):
    scores_mu = []
    scores_std = []
    for col_idx in tqdm(range(X.shape[1])):
        mu, std, y_mu, y_std = permutation_importance_(twin, col_idx, X)
        scores_mu.append(mu)
        scores_std.append(std)
    return dict(zip(X.columns, scores_mu)), dict(zip(X.columns, scores_std)), y_mu, y_std


def plot_pi(pi: dict, target: str, filename: str):
    pi_abs = {k: abs(v) for k, v in pi.items()}
    pi_series = pd.Series(pi_abs)
    pi_series.sort_values(inplace=True)
    # pi_series = pi_series.tail(20)
    fig, ax = plt.subplots()
    pi_series.plot.bar(ax=ax)
    ax.set_title("permutation feature importance")
    ax.set_ylabel(fr"$\Delta\{target}$")
    fig.tight_layout()
    fig.savefig(filename, dpi=600)


def get_r_df(X, y) -> pd.DataFrame:
    corr = r_regression(X, y)
    records = []
    for fn, rv in zip(X.columns.tolist(), corr):
        records.append(
            {"feature": fn, "r": rv}
        )
    return pd.DataFrame.from_records(records)



def main_analysis(learner_name: str, expt_only_r: bool = True):
    """
    perform feature importance analysis

    :param learner_name: name of the learning iteration
    :param expt_only_r: True if only experimental data is used in r reg, otherwise the entire pool is used
    :return:
    """
    learner_path = f"E:/workplace_data/OneLigand/{learner_name}/learner.json.gz"
    learner = json_load(learner_path, gz=True)
    learner: SingleLigandLearner

    output_json = f"PI_{learner_name}.json"
    output_mu_png = f"PI_{learner_name}_mu.png"
    output_std_png = f"PI_{learner_name}_std.png"

    X, y = learner.teaching_records[-1].X, learner.teaching_records[-1].y
    try:
        pi_mu, pi_std = json_load(output_json)
    except FileNotFoundError:
        learner.load_model(model_path=f"E:/workplace_data/OneLigand/{learner_name}/TwinRF_model.pkl")
        pi_mu, pi_std, y_mu, y_std = permutation_importance(learner.current_model, X)
        json_dump([pi_mu, pi_std], output_json)
    plot_pi(pi_mu, "mu", output_mu_png)
    plot_pi(pi_std, "sigma", output_std_png)

    if expt_only_r:
        suffix = "expt"
        X, y = learner.teaching_records[-1].X, learner.teaching_records[-1].y
        corr = X.corr()
        corr.to_csv(f"corr_{learner_name}_{suffix}.csv", index=False)
    else:
        suffix = "all"
        import glob
        y = []
        ligands = []
        amounts = None
        for pkl in tqdm(sorted(glob.glob(f"E:/workplace_data/OneLigand/{learner_name}/prediction/prediction_*.pkl"))):
            slps = pkl_load(pkl, print_timing=False)
            slps: list[SingleLigandPrediction]
            for slp in slps:
                ligands.append(slp.ligand)
                y += slp.pred_mu.tolist()
                amounts = slp.amounts
        ligand_col, X = Molecule.l1_input(ligands, amounts)
    df_r = get_r_df(X, y)
    df_r.to_csv(f"r_{learner_name}_{suffix}.csv", index=False)


if __name__ == '__main__':
    # LEARNER_NAME = """learning_AL0503"""
    LEARNER_NAME = """learning_SL0519"""
    main_analysis(LEARNER_NAME, expt_only_r=True)
    # main_analysis(LEARNER_NAME, expt_only_r=False)
