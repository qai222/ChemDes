from lsal.alearn.one_ligand import TwinRegressor
from lsal.utils import pkl_load

model = pkl_load("../../../workplace_data/OneLigand/learning_AL0331/TwinRF_model.pkl")
model: TwinRegressor
print(model.twin_base_estimator.get_params(deep=True))
