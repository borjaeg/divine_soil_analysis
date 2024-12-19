from typing import Optional
from pprint import pprint

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)
from ConfigSpace.conditions import EqualsCondition

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import (
    SPARSE,
    DENSE,
    SIGNED_DATA,
    UNSIGNED_DATA,
    PREDICTIONS,
)


class XGBRegression(AutoSklearnRegressionAlgorithm):
    def __init__(self, n_estimators, max_depth):
        self.n_estimators = float(n_estimators)
        self.max_depth = float(max_depth)
        from xgboost import XGBRegressor

        self.estimator = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
        )

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "XGB",
            "name": "XGBRegression",
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": False,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter(
             name="n_estimators", lower=50, upper=1000, default_value=100, log=False
        )
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=1, log=False
        )
        cs.add_hyperparameters([n_estimators, max_depth])
        return cs
