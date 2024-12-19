from typing import Optional

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)

import sklearn.metrics

from autosklearn.askl_typing import FEAT_TYPE_TYPE
#import autosklearn.regression
#import autosklearn.pipeline.components.regression
from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import (
    SPARSE,
    DENSE,
    SIGNED_DATA,
    UNSIGNED_DATA,
    PREDICTIONS,
)


class BayesianRidgeRegression(AutoSklearnRegressionAlgorithm):
    def __init__(self):
        self.epsilon = float(epsilon)
        import sklearn.linear_model

        self.estimator = sklearn.linear_model.BayesianRidge()

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
            "shortname": "BRR",
            "name": "Bayesian Ridge Regression",
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

        #cs.add_hyperparameters()
        return cs
