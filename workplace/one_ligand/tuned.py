from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt import load

from lsal.tasks.preprocess import load_descriptors_and_fom, preprocess_descriptor_df, Union, Path
from lsal.twinsk.estimator import TwinRegressor
from lsal.utils import SEED


def train_with_tuned_params(X, y, opt: BayesSearchCV, dumpto: Union[Path, str]):
    reg = TwinRegressor(RandomForestRegressor(n_estimators=100, random_state=SEED))
    reg.set_params(**opt.best_params_)
    reg.fit(X, y)
    dump(reg, filename=dumpto)
    return reg


if __name__ == '__main__':
    labelled_ligands, df_X_labelled, df_y_labelled = load_descriptors_and_fom(
        mdes_csv="../ligand_descriptors/molecular_descriptors_2022_03_21.csv",
        reactions_json="output/2022_0304_LS001_MK003_reaction_data.json",
    )

    data = load("output/tune-data.pkl")
    opt = data["opt"]
    opt: BayesSearchCV
    df_X_labelled = preprocess_descriptor_df(df_X_labelled, scale=False, vthreshould=False)
    reg = train_with_tuned_params(df_X_labelled.values, df_y_labelled.values, opt, dumpto="output/tuned.joblib")
