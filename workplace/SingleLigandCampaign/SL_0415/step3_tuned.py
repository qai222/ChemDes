from skopt import BayesSearchCV
from skopt import load, dump

from lsal.tasks.tune import train_twin_rf_with_tuned_params

if __name__ == '__main__':
    data = load("output/step2.pkl")

    opt = data["step2"]["opt"]
    opt: BayesSearchCV
    X = data["step1"]["df_X"]
    y = data["step1"]["df_y"]
    reg = train_twin_rf_with_tuned_params(X.values, y.values, opt.best_params_)
    step3_data = {
        "tuned_regressor": reg
    }
    data["step3"] = step3_data
    data["step1"] = load("output/step1.pkl")["step1"]  # to be removed
    dump(data, filename="output/step3.pkl")
