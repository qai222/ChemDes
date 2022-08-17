import abc

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


def pair_augment_x(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """ augment two input arrays using their pairwise differences """
    assert x1.ndim == x2.ndim == 2
    n1, m1 = x1.shape
    n2, m2 = x2.shape
    x1 = x1[:, np.newaxis, :].repeat(n2, axis=1)
    x2 = x2[np.newaxis, :, :].repeat(n1, axis=0)
    combined = np.concatenate([x1, x2, x1 - x2], axis=2)
    combined = combined.reshape(n1 * n2, -1)
    return combined


def pair_augment_y(y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """ augmented reg/clf target """
    return (y1[:, np.newaxis] - y2[np.newaxis, :]).flatten()


class TwinEstimator(BaseEstimator, abc.ABC):
    def __init__(self, twin_base_estimator: BaseEstimator, **kwargs):
        """
        a twin estimator built based on a base estimator

        :param twin_base_estimator:
        """
        self.twin_base_estimator = twin_base_estimator
        self.twin_base_estimator.set_params(**kwargs)
        self._train_X = None
        self._train_y = None

    def get_params(self, deep=True):
        d = self.twin_base_estimator.get_params(deep=deep)
        d["twin_base_estimator"] = self.twin_base_estimator
        return d

    def set_params(self, **d):
        base_d = dict()
        for k, v in d.items():
            if k == "twin_base_estimator":
                self.twin_base_estimator = v
            else:
                base_d[k] = v
        self.twin_base_estimator.set_params(**base_d)
        return self

    def fit(self, X, y):
        X_pair = pair_augment_x(X, X)
        y_pair = pair_augment_y(y, y)
        self._train_X = X
        self._train_y = y
        self.twin_base_estimator.fit(X_pair, y_pair)
        return self

    def twin_predict_distribution(self, X) -> np.ndarray:
        n1 = X.shape[0]
        n2 = self._train_X.shape[0]
        X1X2 = pair_augment_x(X, self._train_X)
        y_pair_pred = self.twin_base_estimator.predict(X1X2)
        y_pred_distribution = y_pair_pred.reshape(n1, n2) + self._train_y[np.newaxis, :]
        return y_pred_distribution

    def twin_predict(self, X):
        y_pred_distribution = self.twin_predict_distribution(X)
        mu = y_pred_distribution.mean(axis=1)
        std = y_pred_distribution.std(axis=1)
        return mu, std

    @abc.abstractmethod
    def predict(self, X):
        pass


class TwinRegressor(TwinEstimator, RegressorMixin):

    def predict(self, X):
        mu, _ = self.twin_predict(X)
        return mu
