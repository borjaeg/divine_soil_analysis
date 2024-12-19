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


class PoissonRegression(AutoSklearnRegressionAlgorithm):
    def __init__(self, alpha):
        self.alpha = float(alpha)
        import sklearn.linear_model

        self.estimator = sklearn.linear_model.PoissonRegressor(
            alpha=self.alpha,
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
            "shortname": "PR",
            "name": "PoissonRegression",
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
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.1, upper=2.0, log=True, default_value=1.0
        )
        cs.add_hyperparameters([alpha])
        return cs
