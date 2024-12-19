import os
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
from pprint import pprint
import traceback
import sys
from typing import List
import sys
import glob

# sys.set_int_max_str_digits(0)


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn
import pandas as pd

import numpy as np
from numpy.random import randn
from collections import Counter
import random
import math
from sklearn.model_selection import StratifiedKFold

import csv

from utils.feature_extraction_utils import extract_features
from utils.logging_utils import logging_config
from utils.input_data_utils import get_train_test_data_all
from utils.input_data_utils import get_train_test_data_sampled
from utils.report_utils import write_results
from utils.baseline_results import compute_baseline_results
from utils.scaling_utils import scale_input_data
from utils.feature_selection_utils import select_features

from sklearn.preprocessing import PolynomialFeatures

from scipy import signal

from subset_bands import subset_bands

from search_spaces import SEARCH_SPACES

from automl_config import MINS_SEARCH
from automl_config import TIME_LEFT_FOR_THIS_TASK
from automl_config import PER_RUN_TIME_LIMIT

from config import NDVI_FIRST_COLUMN
from config import OUTPUT_COLUMN_NAME
from config import ONLY_USE_SUBSET
from config import HOLDOUT_MODE
from config import USE_AUTOML

# from config import TRAIN_DATA
from config import TEST_DATA
from config import MODE
from config import SMOOTHING_ORDER
from config import SEED
from config import POLYNOMIAL_DEGREE
from config import NUM_FOLDS
from config import SAMPLING_MODE
from config import HOLDOUT_MODE
from config import USE_HOLDOUT_SCALING
from config import FORCED_COLUMN_IX


def experiment(
    file_name: str,
    current_nutrient,
    other_nutrients,
    initial_experiment: int,
    final_experiment: int,
    initial_fold: int,
    subsample_rate: float,
    feature_size: int,
):
    print(f"Feature Size {feature_size}")
    SAMPLING_RATIOS = [1.0]  # 0.75,
    FILTER_OUTLIERS_MODE = [False]  # , True]
    FILL_MODES = [
        "median",
        # "nearest-interpolation",
        # "mean",
    ]  # , "mean", , "linear-interpolation"]
    FEATURES_TO_SELECT = [75, 24, 12]  # , 24, 16]
    FEATURE_SELECTION_ALGORITHMS = [
        # "intersectison",
        # "union",
        "none",
        "random",
        "permutation",
        "f-regression",
        "mutual-info-regression"
    ]
    FEATURE_EXTRACTION_ALGORITHMS = [
        "autoencoder",
        "random",
        "umap-15",
        "umap-5",
        "pca",
        "all",
    ]  # ["autoencoder"
    SCALING_MODES = ["minmax", "none", "standard"]  # ["standard", "minmax", "none"]
    SMOOTHING_WINDOW_SIZES = [0, 8]
    RFE_PERCENT_FEATURES_TO_SELECT = [1.0, 0.5]
    # file_index = file_name.split("/")[-1].split(".")[0]
    file_index = ""
    # RESULTS_FILE_NAME = f"results_file_individual_{feature_size}_{file_index}_{TEST_DATA}_{HOLDOUT_MODE}"
    RESULTS_FILE_NAME = f"results_file_individual_{feature_size}_{file_name}_{current_nutrient}_{HOLDOUT_MODE}"

    training_data = pd.read_csv(f"./data/{file_name}", header=0, sep=",")
    print("Source Training Data")
    print(training_data.head())
    training_data_processed = training_data.dropna(subset=[current_nutrient])
    training_data_processed.reset_index(drop=True, inplace=True)

    if ONLY_USE_SUBSET:
        training_data_processed = training_data_processed[subset_bands]
    period_names = (
        training_data_processed.columns[NDVI_FIRST_COLUMN:].to_numpy().tolist()
    )
    if HOLDOUT_MODE:
        # test_data = pd.read_csv(f"data/{BANDS}-2022.csv", header=0, sep=",")
        test_data = pd.read_csv(f"data/{TEST_DATA}.csv", header=0, sep=",")
        print("Source Test Data")
        print(test_data.head())
        test_data_processed = test_data.dropna(subset=[current_nutrient])
        test_data_processed.reset_index(drop=True, inplace=True)
        if ONLY_USE_SUBSET:
            test_data_processed = test_data_processed[subset_bands]
    else:
        test_data_processed = None
        # period_names = test_data_processed.columns[NDVI_FIRST_COLUMN:].to_numpy().tolist()
    # print(f"Period Names: {period_names}")
    # open the file in the write mode
    subsample_rate_naming = str(subsample_rate).replace(".", "_")
    with open(
        f"{RESULTS_PATH}{os.path.sep}{RESULTS_FILE_NAME}-{initial_experiment}-{initial_fold}-{subsample_rate_naming}.csv",
        "w",
    ) as results_file:
        # create the csv writer
        writer = csv.writer(results_file)
        np.random.seed(SEED)
        for num_experiment in range(initial_experiment, final_experiment):
            skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=num_experiment)
            for num_fold, (train_idx, test_idx) in enumerate(skf.split(training_data_processed, training_data_processed["Depth"])):
                if num_fold >= initial_fold:
                    for sampling_ratio in SAMPLING_RATIOS:
                        for filter_outliers in FILTER_OUTLIERS_MODE:
                            for fill_mode in FILL_MODES:
                                if HOLDOUT_MODE:
                                    (
                                        train_ndvi_input_raw,
                                        train_yield_raw,
                                        test_ndvi_input_raw,
                                        test_yield,
                                    ) = get_train_test_data_all(
                                        training_data_processed.copy(),
                                        test_data_processed.copy(),
                                        fill_mode,
                                        filter_outliers,
                                        train_idx,
                                        test_idx,
                                        num_experiment,
                                    )
                                else:
                                    if SAMPLING_MODE == "constant":
                                        (
                                            train_ndvi_input_raw,
                                            train_yield_raw,
                                            test_ndvi_input_raw,
                                            test_yield,
                                        ) = get_train_test_data_all(
                                            training_data_processed.copy(),
                                            None,
                                            fill_mode,
                                            filter_outliers,
                                            train_idx,
                                            test_idx,
                                            current_nutrient,
                                            other_nutrients,
                                            num_experiment,
                                        )
                                    elif SAMPLING_MODE == "incremental":
                                        (
                                            train_ndvi_input_raw,
                                            train_yield_raw,
                                            test_ndvi_input_raw,
                                            test_yield,
                                        ) = get_train_test_data_sampled(
                                            training_data_processed.copy(),
                                            train_idx,
                                            test_idx,
                                            subsample_rate=subsample_rate,
                                            num_experiment=num_experiment,
                                        )
                                print(train_ndvi_input_raw.shape)
                                print(train_yield_raw.shape)
                                print(test_ndvi_input_raw.shape)
                                print(test_yield.shape)
                                print("Output Sample")
                                print(train_yield_raw[:10])
                                sampling_ix = sorted(
                                    np.random.choice(
                                        np.arange(0, train_ndvi_input_raw.shape[0], 1),
                                        size=int(
                                            train_ndvi_input_raw.shape[0] * sampling_ratio
                                        ),
                                        replace=False,
                                    )
                                )
                                if feature_size == -1:
                                    feature_window = train_ndvi_input_raw.shape[1]
                                else:
                                    FEATURE_EXTRACTION_ALGORITHMS = ["all"]
                                    FEATURES_TO_SELECT = [1]  # , 24, 16]
                                    feature_window = 1

                                for vi_idx in range(
                                    0, train_ndvi_input_raw.shape[1], feature_window
                                ):
                                    try:
                                        train_ndvi_filtered = train_ndvi_input_raw[
                                            sampling_ix,
                                            vi_idx : min(
                                                vi_idx + feature_window,
                                                train_ndvi_input_raw.shape[1],
                                            ),
                                        ]
                                        print(train_ndvi_filtered.shape)
                                        train_yield = train_yield_raw[sampling_ix]
                                        print(train_yield.shape)
                                        test_ndvi_filtered = test_ndvi_input_raw[
                                            :,
                                            vi_idx : min(
                                                vi_idx + feature_window,
                                                test_ndvi_input_raw.shape[1],
                                            ),
                                        ]
                                        for smoothing_window in SMOOTHING_WINDOW_SIZES:
                                            if smoothing_window > 0:
                                                train_ndvi_smoothed = signal.savgol_filter(
                                                    train_ndvi_filtered,
                                                    window_length=smoothing_window,
                                                    polyorder=SMOOTHING_ORDER,
                                                    mode="nearest",
                                                )
                                                test_ndvi_smoothed = signal.savgol_filter(
                                                    test_ndvi_filtered,
                                                    window_length=smoothing_window,
                                                    polyorder=SMOOTHING_ORDER,
                                                    mode="nearest",
                                                )
                                            else:
                                                train_ndvi_smoothed = train_ndvi_filtered
                                                test_ndvi_smoothed = test_ndvi_filtered
                                            for features_to_select in FEATURES_TO_SELECT:
                                                for scaling_mode in SCALING_MODES:
                                                    try:
                                                        (
                                                            train_ndvi_input_scaled,
                                                            test_ndvi_input_scaled,
                                                        ) = scale_input_data(
                                                            train_ndvi_smoothed,
                                                            test_ndvi_smoothed,
                                                            scaling_mode,
                                                            USE_HOLDOUT_SCALING,
                                                        )
                                                        # print(f"Scaled train_ndvi_input: {train_ndvi_input[:, 0]}")
                                                        # print(f"Scaled test_ndvi_input: {test_ndvi_input[:, 0]}")
                                                        # print(f"[After Scaling] train_ndvi_input: {train_ndvi_input[:,0]}, Type: {train_ndvi_input[:,0].dtype}")
                                                        # print(f"[After Scaling] test_ndvi_input: {test_ndvi_input[:,0]}, Type: {test_ndvi_input[:,0].dtype}")
                                                        for (
                                                            rfe_percent
                                                        ) in RFE_PERCENT_FEATURES_TO_SELECT:
                                                            for (
                                                                feature_selection_algorithm
                                                            ) in FEATURE_SELECTION_ALGORITHMS:
                                                                try:
                                                                    # print(f"[After Selection] train_ndvi_input: {train_ndvi_input[:,0]}, Type: {train_ndvi_input[:,0].dtype}")
                                                                    # print(f"[After Selection] test_ndvi_input: {test_ndvi_input[:,0]}, Type: {test_ndvi_input[:,0].dtype}")
                                                                    (
                                                                        train_ndvi_input_sel,
                                                                        test_ndvi_input_sel,
                                                                        best_features_ix,
                                                                        best_features_ix_2,
                                                                    ) = select_features(
                                                                        train_ndvi_input_scaled,
                                                                        test_ndvi_input_scaled,
                                                                        train_yield,
                                                                        period_names,
                                                                        FORCED_COLUMN_IX,
                                                                        feature_selection_algorithm,
                                                                        features_to_select,
                                                                        rfe_percent,
                                                                    )
                                                                    if best_features_ix == []:
                                                                        best_features_ix = [
                                                                            *range(
                                                                                vi_idx,
                                                                                min(
                                                                                    vi_idx
                                                                                    + feature_window,
                                                                                    train_ndvi_input_raw.shape[
                                                                                        1
                                                                                    ],
                                                                                ),
                                                                            )
                                                                        ]
                                                                    else:
                                                                        print(
                                                                            "Best Features Selected"
                                                                        )
                                                                    # poly = PolynomialFeatures(
                                                                    #    degree=2, include_bias=False
                                                                    # )
                                                                    # train_ndvi_input_poly = (
                                                                    #    poly.fit_transform(
                                                                    #        train_ndvi_input_sel
                                                                    #    )
                                                                    # )
                                                                    # test_ndvi_input_poly = (
                                                                    #    poly.transform(
                                                                    #        test_ndvi_input_sel
                                                                    #    )
                                                                    # )
                                                                    for (
                                                                        feature_extracion_mode
                                                                    ) in FEATURE_EXTRACTION_ALGORITHMS:
                                                                        if (
                                                                            feature_extracion_mode
                                                                            == "all"
                                                                        ):
                                                                            FEATURES_TO_EXTRACT = [
                                                                                features_to_select
                                                                            ]
                                                                        else:
                                                                            FEATURES_TO_EXTRACT = [
                                                                                -1,
                                                                                0.95,
                                                                                0.99,
                                                                                features_to_select
                                                                                // 2,
                                                                            ]
                                                                        for (
                                                                            features_to_extract
                                                                        ) in (
                                                                            FEATURES_TO_EXTRACT
                                                                        ):
                                                                            print(
                                                                                "[INFO] Before Feature Extraction"
                                                                            )
                                                                            if (
                                                                                feature_extracion_mode
                                                                                != "all"
                                                                                and features_to_select
                                                                                > 1
                                                                            ):
                                                                                (
                                                                                    train_ndvi_input,
                                                                                    test_ndvi_input,
                                                                                    history,
                                                                                ) = extract_features(
                                                                                    train_ndvi_input_sel,
                                                                                    test_ndvi_input_sel,
                                                                                    # train_ndvi_input_poly,
                                                                                    # test_ndvi_input_poly,
                                                                                    features_to_extract,
                                                                                    # train_ndvi_input_sel.shape[
                                                                                    #    1
                                                                                    # ],
                                                                                    mode=feature_extracion_mode,
                                                                                )
                                                                            else:
                                                                                train_ndvi_input = train_ndvi_input_sel
                                                                                test_ndvi_input = test_ndvi_input_sel
                                                                                history = {
                                                                                    "val_loss": [
                                                                                        0.0
                                                                                    ],
                                                                                    "loss": [
                                                                                        0.0
                                                                                    ],
                                                                                }

                                                                            print(
                                                                                "[INFO] After Feature Extraction"
                                                                            )

                                                                            # print(f"[After Extraction] train_ndvi_input: {train_ndvi_input[:,0]}, Type: {train_ndvi_input[:,0].dtype}")
                                                                            # print(f"[After Extraction] test_ndvi_input: {test_ndvi_input[:,0]}, Type: {test_ndvi_input[:,0].dtype}")

                                                                            train_ndvi_input = train_ndvi_input.astype(
                                                                                np.float16
                                                                            )
                                                                            test_ndvi_input = test_ndvi_input.astype(
                                                                                np.float16
                                                                            )

                                                                            num_evidences = train_ndvi_input.shape[
                                                                                0
                                                                            ]

                                                                            configuration = "_".join(
                                                                                [
                                                                                    fill_mode,
                                                                                    str(
                                                                                        filter_outliers
                                                                                    ),
                                                                                    str(
                                                                                        rfe_percent
                                                                                    ),
                                                                                    str(
                                                                                        feature_selection_algorithm
                                                                                    ),
                                                                                    str(
                                                                                        subsample_rate
                                                                                    ),
                                                                                    str(-1),
                                                                                    str(
                                                                                        NUM_FOLDS
                                                                                    ),
                                                                                    str(
                                                                                        features_to_select
                                                                                    ),
                                                                                    str(
                                                                                        scaling_mode
                                                                                    ),
                                                                                    str(
                                                                                        feature_extracion_mode
                                                                                    ),
                                                                                    str(
                                                                                        features_to_extract
                                                                                    ),
                                                                                    str(
                                                                                        smoothing_window
                                                                                    ),
                                                                                    str(
                                                                                        POLYNOMIAL_DEGREE
                                                                                    ),
                                                                                    str(
                                                                                        MINS_SEARCH
                                                                                    ),
                                                                                ]
                                                                            )

                                                                            for (
                                                                                ix_search_space
                                                                            ) in range(
                                                                                len(
                                                                                    SEARCH_SPACES
                                                                                )
                                                                            ):
                                                                                print(
                                                                                    f"""[INFO] Num. experiment: {num_experiment}, K-Splits {subsample_rate}, 
                                                                                Ensemble Size: -1, Fill Mode: {fill_mode},
                                                                                Feature Selection Mode: {feature_selection_algorithm},
                                                                                Filter Mode: {filter_outliers},
                                                                                Features to select: {features_to_select}, Scaling Mode: {scaling_mode},
                                                                                Search Space: {SEARCH_SPACES[ix_search_space]}"""
                                                                                )
                                                                                if USE_AUTOML:
                                                                                    (
                                                                                        selected_models_type,
                                                                                        selected_models_weights,
                                                                                        pre_auto_results,
                                                                                        auto_results,
                                                                                    ) = fit_automl(
                                                                                        train_ndvi_input,
                                                                                        train_yield,
                                                                                        ix_search_space,
                                                                                        3,
                                                                                        2,
                                                                                    )
                                                                                else:
                                                                                    selected_models_type = (
                                                                                        ""
                                                                                    )
                                                                                    selected_models_weights = (
                                                                                        ""
                                                                                    )
                                                                                    pre_auto_results = [
                                                                                        0.0,
                                                                                        0.0,
                                                                                        0.0,
                                                                                        0.0,
                                                                                        0.0,
                                                                                    ]
                                                                                    auto_results = [
                                                                                        0.0,
                                                                                        0.0,
                                                                                        0.0,
                                                                                        0.0,
                                                                                        0.0,
                                                                                    ]
                                                                                start_time = (
                                                                                    time.time()
                                                                                )

                                                                                baseline_results = compute_baseline_results(
                                                                                    train_ndvi_input,
                                                                                    train_yield,
                                                                                    test_ndvi_input,
                                                                                    test_yield,
                                                                                )
                                                                                end_time = (
                                                                                    time.time()
                                                                                )

                                                                                write_results(
                                                                                    writer,
                                                                                    results_file,
                                                                                    # file_index,
                                                                                    current_nutrient,
                                                                                    current_nutrient,
                                                                                    round(
                                                                                        np.min(
                                                                                            train_yield
                                                                                        ),
                                                                                        5,
                                                                                    ),
                                                                                    round(
                                                                                        np.max(
                                                                                            train_yield
                                                                                        ),
                                                                                        5,
                                                                                    ),
                                                                                    round(
                                                                                        np.min(
                                                                                            test_yield
                                                                                        ),
                                                                                        5,
                                                                                    ),
                                                                                    round(
                                                                                        np.max(
                                                                                            test_yield
                                                                                        ),
                                                                                        5,
                                                                                    ),
                                                                                    str(
                                                                                        num_experiment
                                                                                    ),
                                                                                    str(
                                                                                        num_fold
                                                                                    ),
                                                                                    configuration,
                                                                                    str(
                                                                                        ix_search_space
                                                                                    ),
                                                                                    selected_models_type,
                                                                                    selected_models_weights,
                                                                                    train_ndvi_input.shape,
                                                                                    np.array(
                                                                                        period_names
                                                                                    )[
                                                                                        best_features_ix
                                                                                    ][
                                                                                        :features_to_select
                                                                                    ],
                                                                                    np.array(
                                                                                        period_names
                                                                                    )[
                                                                                        best_features_ix_2
                                                                                    ][
                                                                                        :features_to_select
                                                                                    ],
                                                                                    # np.array(
                                                                                    #    period_names
                                                                                    # )[
                                                                                    #    best_features_ix
                                                                                    # ][
                                                                                    #    np.argsort(
                                                                                    #        baseline_results[
                                                                                    #            -1
                                                                                    #        ]
                                                                                    #    )[::-1]
                                                                                    # ],
                                                                                    # np.array(
                                                                                    #    period_names
                                                                                    # )[
                                                                                    #    best_features_ix
                                                                                    # ][
                                                                                    #    baseline_results[
                                                                                    #        -2
                                                                                    #    ]
                                                                                    # ][
                                                                                    #    0
                                                                                    # ],
                                                                                    # history["loss"][
                                                                                    #    -1
                                                                                    # ],
                                                                                    # history[
                                                                                    #    "val_loss"
                                                                                    # ][-1],
                                                                                    # pre_auto_results,
                                                                                    # auto_results,
                                                                                    baseline_results,
                                                                                    end_time
                                                                                    - start_time,
                                                                                )

                                                                                print("=" * 100)

                                                                except Exception as e2:
                                                                    print("Internal Exception")
                                                                    print(
                                                                        traceback.format_exc()
                                                                    )
                                                                    print(e2)
                                                    except Exception as e1:
                                                        print("Exception 1")
                                                        print(traceback.format_exc())
                                                        print(e1)
                                    except Exception as e3:
                                        print("Exception 2")
                                        print(traceback.format_exc())
                                        print(e3)


if __name__ == "__main__":
    print("hello")
    RANDOM_STATE = 2024
    RESULTS_PATH = "results"
    inital_experiment = int(sys.argv[1])
    final_experiment = int(sys.argv[2])
    initial_fold = int(sys.argv[3])
    subsample_rate = float(sys.argv[4])  # 2, 3
    feature_size = int(sys.argv[5])  # 1, -1
    init_file_ix = int(sys.argv[6])  # 1, 3
    end_file_ix = int(sys.argv[7])

    micro_nutrient_columns = ["pH", "EC", "Total CaCO3", "Organic Matter"] + [
        "B",
        "Total N",
        "P",
        "Κ",
        "Ca",
        "Mg",
        "Fe",
        "Cu",
        "Mn",
        "Zn",
    ]

    for ix in range(len(micro_nutrient_columns)):
        print(ix)
        if ix >= init_file_ix and ix <= end_file_ix:
            current_nutrient = micro_nutrient_columns[ix]
            print(current_nutrient)
            other_nutrients = list(
                set(micro_nutrient_columns) - set([current_nutrient])
            )
            file_name = "soil_all_depths.csv"
            experiment(
                file_name,
                current_nutrient,
                other_nutrients,
                inital_experiment,
                final_experiment,
                initial_fold,
                subsample_rate,
                feature_size,
            )
