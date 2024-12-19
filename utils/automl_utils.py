import autosklearn
import autosklearn.regression
import autosklearn.pipeline.components.regression
from extra_regressors.theilsen import TheilSenRegression
from extra_regressors.huber import HuberRegression
from extra_regressors.bayesian_ridge import BayesianRidgeRegression
from extra_regressors.passive_aggressive_regression import PassiveAggressiveRegression
from extra_regressors.poisson_regression import PoissonRegression
from extra_regressors.gamma_regression import GammaRegression
from extra_regressors.xgb import XGBRegression

from extra_preprocessors.feature_selection import FeatureSelection_1
from extra_preprocessors.no_preprocessing import NoPreprocessing


autosklearn.pipeline.components.regression.add_regressor(TheilSenRegression)
autosklearn.pipeline.components.regression.add_regressor(HuberRegression)
autosklearn.pipeline.components.regression.add_regressor(BayesianRidgeRegression)

autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(
    FeatureSelection_1
)

autosklearn.pipeline.components.data_preprocessing.add_preprocessor(NoPreprocessing)


def fit_automl(train_ndvi_input, train_yield, ix_search_space, ensemble_size, k_splits):
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=TIME_LEFT_FOR_THIS_TASK,
        per_run_time_limit=PER_RUN_TIME_LIMIT,
        ensemble_size=ensemble_size,
        memory_limit=8000,
        resampling_strategy="cv",
        resampling_strategy_arguments={"folds": k_splits},
        metric=autosklearn.metrics.root_mean_squared_error,
        include=SEARCH_SPACES[ix_search_space],
        ensemble_nbest=ensemble_size,
        ensemble_class="default",
        initial_configurations_via_metalearning=0,
        tmp_folder=None,
        dataset_compression=False,
        dataset_compression={
            "memory_allocation": 0.5,
            "methods": ["precision"],
        },
        n_jobs=1,
    )
    print(
        f"[Before Training] train_ndvi_input: {train_ndvi_input[:,0]}, Type: {train_ndvi_input[:,0].dtype}"
    )
    print(
        f"[Before Training] test_ndvi_input: {test_ndvi_input[:,0]}, Type: {test_ndvi_input[:,0].dtype}"
    )
    print(f"[Before Training] train_ndvi_input type: {train_ndvi_input[:,0].dtype}")
    automl.fit(
        train_ndvi_input,
        train_yield,
        dataset_name=f"all_bands-tomatoes-{feature_extracion_mode}-{k_splits}-{fill_mode}-{ensemble_size}",
    )

    selected_models_weights = " - ".join(
        weight.strip()
        for weight in str(automl.leaderboard().to_dict()["ensemble_weight"]).split(",")
    )
    selected_models_type = " - ".join(
        weight.strip()
        for weight in str(automl.leaderboard().to_dict()["type"]).split(",")
    )
    print("=" * 50)
    (
        pre_train_r2_score,
        pre_test_r2_score,
        pre_adjusted_train_r2_score,
        pre_adjusted_test_r2_score,
        pre_mse,
    ) = compute_metrics(
        automl,
        train_yield,
        train_ndvi_input,
        test_yield,
        test_ndvi_input,
    )
    print(f"[INFO] Before Refitting")
    automl.refit(
        train_ndvi_input.copy(),
        train_yield.copy(),
    )
    (
        train_r2_score,
        test_r2_score,
        adjusted_train_r2_score,
        adjusted_test_r2_score,
        mse,
    ) = compute_metrics(
        automl,
        train_yield,
        train_ndvi_input,
        test_yield,
        test_ndvi_input,
    )

    pprint(
        automl.show_models(),
        indent=4,
    )
    print(automl.leaderboard())

    return (
        selected_models_type,
        selected_models_weights,
        [
            pre_train_r2_score,
            pre_test_r2_score,
            pre_adjusted_train_r2_score,
            pre_adjusted_test_r2_score,
            pre_mse,
        ],
        [
            train_r2_score,
            test_r2_score,
            adjusted_train_r2_score,
            adjusted_test_r2_score,
            mse,
        ],
    )
