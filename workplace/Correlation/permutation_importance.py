import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import r_regression
from sklearn.utils import check_random_state
from tqdm import tqdm

from lsal.alearn.one_ligand import SingleLigandLearner
from lsal.twinsk import TwinRegressor
from lsal.utils import json_load, json_dump


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
    return float(np.mean(mu_permuted - mu)), float(np.mean(std_permuted - std))


def permutation_importance(twin: TwinRegressor, X: pd.DataFrame):
    scores_mu = []
    scores_std = []
    for col_idx in tqdm(range(X.shape[1])):
        mu, std = permutation_importance_(twin, col_idx, X)
        scores_mu.append(mu)
        scores_std.append(std)
    return dict(zip(X.columns, scores_mu)), dict(zip(X.columns, scores_std))


def plot_pi(pi: dict, name: str):
    pi_abs = {k: abs(v) for k, v in pi.items()}
    pi_series = pd.Series(pi_abs)
    pi_series.sort_values(inplace=True)
    # pi_series = pi_series.tail(20)
    fig, ax = plt.subplots()
    pi_series.plot.bar(ax=ax)
    ax.set_title("permutation feature importance")
    ax.set_ylabel(fr"$\Delta\{name}$")
    fig.tight_layout()
    fig.savefig("pi_{}.png".format(name), dpi=600)


def print_univariate(X, y):
    corr = r_regression(X, y)
    for fn, rv in zip(X.columns.tolist(), corr):
        print(fn, rv)


if __name__ == '__main__':
    learner_folder = """learning_AL0503"""
    learner_json = f"E:/workplace_data/OneLigand/{learner_folder}/learner.json.gz"
    learner = json_load(learner_json, gz=True)
    learner: SingleLigandLearner
    try:
        pi_mu, pi_std = json_load(f"{learner_folder}_PI.json")
    except FileNotFoundError:
        learner.load_model(model_path=f"E:/workplace_data/OneLigand/{learner_folder}/TwinRF_model.pkl")
        pi_mu, pi_std = permutation_importance(learner.current_model, learner.teaching_records[-1].X)
        json_dump([pi_mu, pi_std], learner_json + "_FI.json")
    plot_pi(pi_mu, 'mu')
    plot_pi(pi_std, 'sigma')

    X, y = learner.teaching_records[-1].X, learner.teaching_records[-1].y
    print_univariate(X, y)
