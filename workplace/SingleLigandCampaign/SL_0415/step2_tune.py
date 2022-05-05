from skopt import dump, load

from lsal.tasks.tune import tune_twin_rf

if __name__ == '__main__':
    # load step1 data
    data = load("output/step1.pkl")
    X = data["step1"]["df_X"]
    y = data["step1"]["df_y"]

    new_data = tune_twin_rf(X, y)
    data["step2"] = new_data
    dump(data, "output/step2.pkl")
