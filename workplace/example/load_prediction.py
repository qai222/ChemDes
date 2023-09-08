from lsal.utils import pkl_load
from lsal.alearn.one_ligand import SingleLigandPrediction
from loguru import logger

prediction_chunk = pkl_load("../../workplace_data/OneLigand/learning_AL0503/prediction/prediction_chunk_000000.pkl")

for pred in prediction_chunk:
    # predictions are saved as a `SingleLigandPrediction` object
    pred: SingleLigandPrediction
    logger.info("here are the predictions for:", pred.ligand.label)
    for i, amount in enumerate(pred.amounts):
        logger.info(f"concentration: {amount}, averaged predicted FOM: {pred.pred_mu[i]}, std of this prediction: {pred.pred_std[i]}")
        logger.info(f"actual predictions for this concentration form an array of size: {pred.prediction_values[i].size}")
    break