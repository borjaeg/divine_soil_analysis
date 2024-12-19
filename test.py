import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

X, y = sklearn.datasets.make_regression(
    n_samples=100, n_features=5, n_informative=1, random_state=2023
)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    ensemble_kwargs={"ensemble_size": 3},
    resampling_strategy="cv",
    resampling_strategy_arguments={"folds": 3},
    metric=autosklearn.metrics.root_mean_squared_error,
    include={
        "regressor": ["libsvm_svr", "ard_regression", "sgd"],
        "feature_preprocessor": ["select_percentile_regression"],
        "data_preprocessor": ["feature_type"],
    },
    ensemble_nbest=3,
    initial_configurations_via_metalearning=0,
    tmp_folder=None,
    n_jobs=1,
)

automl.fit(X_train, y_train)
