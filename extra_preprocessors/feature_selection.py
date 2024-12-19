from typing import Optional

import autosklearn.pipeline.components.data_preprocessing
import sklearn.metrics
from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.askl_typing import FEAT_TYPE_TYPE
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, UNSIGNED_DATA, INPUT

class FeatureSelection_3(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, **kwargs):
        """This preprocessors does not change the data"""
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression
        #for key, val in kwargs.items():
        #    setattr(self, key, val)
        self.transformer = SelectKBest(f_regression, k=3)

    def fit(self, X, Y=None):
        self.transformer.fit(X, y)

    def transform(self, X):
        return self.transformer.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "3FS",
            "name": "3-Feature Selection",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        return ConfigurationSpace()

    
class FeatureSelection_1(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, **kwargs):
        """This preprocessors does not change the data"""
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression
        #for key, val in kwargs.items():
        #    setattr(self, key, val)
        self.transformer = SelectKBest(f_regression, k=1)

    def fit(self, X, Y=None):
        self.transformer.fit(X, y)

    def transform(self, X):
        return self.transformer.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "1FS",
            "name": "1-Feature Selection",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        return ConfigurationSpace()    
    
class FeatureSelection_5(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, **kwargs):
        """This preprocessors does not change the data"""
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression
        #for key, val in kwargs.items():
        #    setattr(self, key, val)
        self.transformer = SelectKBest(f_regression, k=5)

    def fit(self, X, Y=None):
        self.transformer.fit(X, y)

    def transform(self, X):
        return self.transformer.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            "shortname": "5FS",
            "name": "5-Feature Selection",
            "handles_regression": True,
            "handles_classification": True,
            "handles_multiclass": True,
            "handles_multilabel": True,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA),
            "output": (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(
        feat_type: Optional[FEAT_TYPE_TYPE] = None, dataset_properties=None
    ):
        return ConfigurationSpace()