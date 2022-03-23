import logging
from pathlib import Path
from typing import Union

import numpy as np

from chemdes.twinsk.estimator import BaseEstimator, TwinClassifier, TwinRegressor, RegressorMixin, \
    upper_confidence_interval
from chemdes.utils import SEED


def valid_float_indices(a: np.ndarray) -> np.ndarray:
    assert a.ndim == 1
    return np.argwhere(~np.isnan(a)).T[0]


def invalid_float_indices(a: np.ndarray) -> np.ndarray:
    assert a.ndim == 1
    return np.argwhere(np.isnan(a)).T[0]


def multi_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Selects the indices of the n_instances highest values.

    Args:
        values: Contains the values to be selected from.
        n_instances: Specifies how many indices to return.

    Returns:
        The indices of the n_instances largest values.
    """
    assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility'

    max_idx = np.argpartition(-values, n_instances - 1, axis=0)[:n_instances]
    return max_idx


class TwinActiveLearnerError(Exception): pass


class TwinActiveLearner:
    _available_prediction_outcomes = ["std", "mu", "uci"]

    def __init__(
            self,
            name: str,
            base_estimator: BaseEstimator,
            X_raw: np.ndarray,
            y_raw: np.ndarray,
            properties: dict = None,
            taught_indices: list[int] = None,
    ):
        self.name = name
        self._X_raw = X_raw
        self._y_raw = y_raw

        if isinstance(base_estimator, RegressorMixin):
            self.twin_estimator = TwinRegressor(base_estimator)
        else:
            self.twin_estimator = TwinClassifier(base_estimator)

        if properties is None:
            properties = dict()
        self.properties = properties
        if taught_indices is None:
            taught_indices = []
        self._taught_indices = taught_indices

        assert self.n_raw > 1
        assert self.y_raw.ndim == 1

    @property
    def n_raw(self):
        return self.X_raw.shape[0]

    @property
    def details(self):
        # TODO print out details
        return

    @property
    def n_taught(self):
        return len(self.taught_indices)

    @property
    def raw_data(self):
        return self.X_raw, self.y_raw

    @property
    def taught_indices(self):
        return self._taught_indices

    @taught_indices.setter
    def taught_indices(self, indices: list[int]):
        logging.critical("UPDATING `taught_indices`...")
        assert isinstance(indices, list)
        self._taught_indices = indices

    @property
    def raw_indices(self):
        return list(range(self.n_raw))

    @property
    def labelled_indices(self):
        return tuple(valid_float_indices(self.y_raw))

    @property
    def unlabelled_indices(self):
        return tuple(invalid_float_indices(self.y_raw))

    @property
    def teachable_indices(self):
        indices = [i for i in self.labelled_indices if i not in self.taught_indices]
        return tuple(indices)

    @property
    def X_raw(self):
        return self._X_raw

    @X_raw.setter
    def X_raw(self, new_X_raw):
        raise TwinActiveLearnerError("DO NOT SET `X_raw`, CREATE A NEW LEARNER!")

    @property
    def y_raw(self):
        return self._y_raw

    @y_raw.setter
    def y_raw(self, new_y_raw):
        logging.critical("UPDATING `y_raw`...")
        assert new_y_raw.ndim == 1
        assert new_y_raw.shape[0] == self.y_raw.shape[0] == self.X_raw.shape[0]
        self.y_raw = new_y_raw
        logging.critical("`y_raw` UPDATED!")

    @property
    def X_taught(self):
        return self.X_raw[self.taught_indices]

    @property
    def y_taught(self):
        return self.y_raw[self.taught_indices]

    def __hash__(self):
        # TODO hmmm...
        return hash(self.name + self.twin_estimator.base_estimator.__class__.__name__)

    def teach(self, indices: list[int] or int):
        """
        fit the twin estimator, indices of `self.X_raw` will be **appended** to the existing `self.taught_indices`

        :param indices: a list of indices which should be a subset of `self.teachable_indices`
        :return:
        """
        if isinstance(indices, int):
            indices = [indices, ]
        assert not set(indices).intersection(set(self.taught_indices)), "some teaching indices already taught"
        assert set(indices).issubset(set(self.teachable_indices)), "teaching indices must be labelled!"

        logging.critical("teaching indices UPDATING:")
        logging.critical("OLD: {}".format(self.taught_indices))
        self.taught_indices = list(self.taught_indices) + [ii for ii in indices]
        logging.critical("NEW: {}".format(self.taught_indices))
        X_teach = self.X_taught
        y_teach = self.y_taught
        self.twin_estimator.fit(X_teach, y_teach)
        logging.critical("teaching finished")

    def first_lesson(self, n=2, seed=SEED):
        """ teach first n labelled data points """
        random_state = np.random.RandomState(seed)
        indices = random_state.choice(self.teachable_indices, n, replace=False)
        self.teach(indices)

    def prediction_outcomes(self, X: np.ndarray = None) -> dict[str, np.ndarray]:
        if X is None:
            X = self.X_raw
        y_distribution = self.twin_estimator.twin_predict_distribution(X)
        mu = y_distribution.mean(axis=1)
        std = y_distribution.std(axis=1)
        uci = np.apply_along_axis(upper_confidence_interval, 1, y_distribution)
        return {"mean": mu, "std": std, "uci": uci}

    def query(self, outcome_type: str = "uci", strategy="max", k=1, scope="teachable") -> list[int]:
        """ select indices from `self.teachable_indices` """
        assert scope in ["teachable", "taught", "raw", "unlabelled", "labelled"]
        scope_indices = getattr(self, "{}_indices".format(scope))
        X_query = self.X_raw[scope_indices]
        if strategy == "random":
            rs = np.random.RandomState(len(self.teachable_indices))
            query_indices = rs.choice(scope_indices, k, replace=False).tolist()
        elif strategy == "max":
            assert outcome_type in self._available_prediction_outcomes
            outcomes = self.prediction_outcomes(X=X_query)[outcome_type]
            query_indices = multi_argmax(outcomes, k).tolist()
            query_indices = [scope_indices[i] for i in query_indices]
        else:
            raise TwinActiveLearnerError("strategy not implemented: {}".format(strategy))
        return query_indices

    def save(self, saveas: Union[Path, str]):
        from joblib import dump
        data = dict(
            name=self.name,
            X_raw=self.X_raw,
            y_raw=self.y_raw,
            properties=self.properties,
            taught_indices=self.taught_indices,
            twin_estimator=self.twin_estimator
        )
        dump(data, saveas)

    @classmethod
    def load(cls, loadfrom: Union[Path, str]):
        from joblib import load
        data = load(loadfrom)
        return cls(**data)
