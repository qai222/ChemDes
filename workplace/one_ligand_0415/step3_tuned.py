from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt import load

from lsal.tasks.preprocess import Union, Path
from lsal.twinsk.estimator import TwinRegressor
from lsal.utils import SEED


def train_with_tuned_params(X, y, opt: BayesSearchCV, dumpto: Union[Path, str]):
    reg = TwinRegressor(RandomForestRegressor(n_estimators=100, random_state=SEED))
    reg.set_params(**opt.best_params_)
    reg.fit(X, y)
    dump(reg, filename=dumpto)
    return reg


if __name__ == '__main__':
    data = load("output/step2_tune-data.pkl")
    opt = data["opt"]
    opt: BayesSearchCV
    X = data["X"]
    y = data["y"]
    reg = train_with_tuned_params(X.values, y.values, opt, dumpto="output/tuned.joblib")
