import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Integer

from lsal.twinsk.estimator import TwinRegressor
from lsal.utils import SEED


def tune_twin_rf(X: pd.DataFrame, y: pd.DataFrame, ):
    n_features = X.shape[1]

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=SEED)

    # RF for twin regressor
    base_estimator = RandomForestRegressor(n_estimators=100, random_state=SEED)
    reg = TwinRegressor(base_estimator)

    # the search space defined for the `base_estimator`, in this case RFG
    space = {
        "max_depth": Integer(1, 50, prior="uniform"),
        "max_features": Integer(1, n_features, prior="uniform"),
        "min_samples_split": Integer(2, 10, prior="uniform"),
        "min_samples_leaf": Integer(1, 10, prior="uniform"),
    }

    # use mse to score the regressor
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # create skopt optimizer
    opt = BayesSearchCV(
        estimator=reg,
        search_spaces=space,
        scoring=scorer,
        n_jobs=1,
        cv=5,
        verbose=10,
        return_train_score=True,
        n_iter=50,
        random_state=SEED,
    )
    opt.fit(X_train.values, y_train.values)

    return X_train, y_train, X_test, y_test, opt


def train_twin_rf_with_tuned_params(X, y, opt_params: dict):
    reg = TwinRegressor(RandomForestRegressor(n_estimators=100, random_state=SEED))
    reg.set_params(**opt_params)
    reg.fit(X, y)
    return reg
