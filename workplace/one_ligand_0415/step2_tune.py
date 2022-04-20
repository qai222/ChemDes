import logging
import os
import random

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split
from skopt import dump, BayesSearchCV
from skopt.space import Integer

from lsal.tasks.preprocess import preprocess_descriptor_df, load_descriptors_and_fom
from lsal.twinsk.estimator import TwinRegressor
from lsal.utils import SEED
from lsal.utils import strip_extension

labelled_ligands, df_X_labelled, df_y_labelled = load_descriptors_and_fom(
    mdes_csv="../ligand_descriptors/molecular_descriptors_2022_03_21.csv",
    reactions_json="output/reactions_data.json",
)

# prepare ML input
X_ligands = df_X_labelled[["ligand_inchi", "ligand_iupac_name"]]
X = preprocess_descriptor_df(df_X_labelled, scale=False, vthreshould=False)  # select numbers
y = df_y_labelled
n_features = X.shape[1]

if __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)

    # setup logging
    filebasename = strip_extension(os.path.basename(__file__))
    logging.basicConfig(filename='{}.log'.format(filebasename), filemode="w")

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

    data = {
        "X": X,
        "df_X_labelled": df_X_labelled,
        "X_ligands": X_ligands,
        "y": y,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "opt": opt,
    }

    print(opt.score(X_test.values, y_test.values))
    dump(data, "output/{}-data.pkl".format(filebasename))
