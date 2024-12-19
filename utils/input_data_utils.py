from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# from config import OUTPUT_COLUMN_NAME
from config import SEED
from config import COLUMNS_TO_REMOVE

import numpy as np
import pandas as pd


def clean_data(data, column_names: str):
    print(column_names)
    train_data = data.drop(column_names, axis=1)

    return train_data


# def load_data(train_data, test_data, test_size, num_experiment: int):


def load_data(
    train_data,
    test_data,
    train_idx,
    test_idx,
    OUTPUT_COLUMN_NAME,
    columns_to_remove,
    num_experiment: int,
):

    preprocessed_train_data = clean_data(
        train_data, COLUMNS_TO_REMOVE + columns_to_remove
    )
    if test_data is None:
        print("HOULDOUT deactivated")
        x_train = preprocessed_train_data.drop([OUTPUT_COLUMN_NAME], axis=1).iloc[train_idx]
        x_test = preprocessed_train_data.drop([OUTPUT_COLUMN_NAME], axis=1).iloc[test_idx]
        y_train = preprocessed_train_data[OUTPUT_COLUMN_NAME].iloc[train_idx]
        y_test = preprocessed_train_data[OUTPUT_COLUMN_NAME].iloc[test_idx]
        #x_train, x_test, y_train, y_test = train_test_split(
        #    preprocessed_train_data.drop([OUTPUT_COLUMN_NAME], axis=1),
        #    preprocessed_train_data[OUTPUT_COLUMN_NAME],
        #    test_size=test_size,
        #    shuffle=True,
        #    random_state=SEED // (num_experiment + 1),
        #)
    else:
        print("HOULDOUT activated")
        preprocessed_test_data = clean_data(test_data, COLUMNS_TO_REMOVE)
        x_train = preprocessed_train_data.drop([OUTPUT_COLUMN_NAME], axis=1)
        x_test = preprocessed_test_data.drop([OUTPUT_COLUMN_NAME], axis=1)
        y_train = preprocessed_train_data[OUTPUT_COLUMN_NAME]
        y_test = preprocessed_test_data[OUTPUT_COLUMN_NAME]

    return x_train, y_train, x_test, y_test


def fill_data(x_train, x_test, fill_method: str):
    if fill_method == "mean":
        train_data_processed = x_train.fillna(x_train.mean())
        test_data_processed = x_test.fillna(x_test.mean())
        # print(test_data_processed)
    elif fill_method == "median":
        train_data_processed = x_train.fillna(x_train.median())
        test_data_processed = x_test.fillna(x_test.median())
    elif fill_method == "ffill":
        train_data_processed = x_train.fillna(method="ffill")
        test_data_processed = x_test.fillna(method="ffill")
    elif fill_method == "nearest-interpolation":
        train_data_processed = x_train.fillna(
            x_train.interpolate(method="nearest").median()
        )
        test_data_processed = x_test.fillna(
            x_test.interpolate(method="nearest").median()
        )
    elif fill_method == "linear-interpolation":
        train_data_processed = x_train.fillna(
            x_train.interpolate(method="linear").median()
        )
        test_data_processed = x_test.fillna(
            x_test.interpolate(method="linear").median()
        )
    else:
        raise NotImplementedError

    return train_data_processed, test_data_processed


def dataframe_to_numpy(train_data_processed, test_data_processed):
    DATA_START_IX = 0

    train_data_array = train_data_processed.to_numpy()
    train_ndvis = train_data_array[:, DATA_START_IX:].astype(np.float32)
    test_data_array = test_data_processed.to_numpy()
    test_ndvis = test_data_array[:, DATA_START_IX:].astype(np.float32)

    return train_ndvis, test_ndvis


def get_train_test_data_all(
    train_data_processed,
    test_data_processed,
    fill_method: str,
    filter_outliers: bool,
    train_idx,
    test_idx,
    output_column_name: str,
    columns_to_remove: str,
    num_experiment: int,
):
    CONTAMINATION = 0.1
    # train_data = data_processed.drop(labels=test_indexes, axis=0)
    x_train, y_train, x_test, y_test = load_data(
        train_data_processed,
        test_data_processed,
        train_idx,
        test_idx,
        output_column_name,
        columns_to_remove,
        num_experiment,
    )
    if test_data_processed is not None:
        y_train, y_test = np.array(y_train), np.array(y_test)
    else:
        y_train, y_test = np.array(y_train), np.array(y_test)

    print("Filling data")
    for column in x_train.columns:
        x_train[column] = pd.to_numeric(x_train[column], errors="coerce")
    for column in x_train.columns:
        x_test[column] = pd.to_numeric(x_test[column], errors="coerce")
    print(x_train.head())
    print(x_train.median())
    print(x_test.head())
    print(x_test.median())
    train_data_processed, test_data_processed = fill_data(x_train, x_test, fill_method)
    train_ndvis, test_ndvis = dataframe_to_numpy(
        train_data_processed, test_data_processed
    )

    if filter_outliers:
        iso = IsolationForest(contamination=CONTAMINATION)
        is_outlier = iso.fit_predict(y_train.reshape(len(y_train), 1))
        mask = is_outlier != -1
        return (
            train_ndvis[mask, :],
            y_train[mask],
            test_ndvis[:, :],
            y_test,
        )

    return (
        train_ndvis.astype(np.float32),
        y_train.astype(np.float32),
        test_ndvis.astype(np.float32),
        y_test.astype(np.float32),
    )


def get_train_test_data_sampled(
    data: pd.DataFrame,
    test_size: float,
    subsample_rate: int,
    num_experiment: int,
):
    upper_leek = 288
    upper_mushroom = 539
    upper_broccoli = 789
    upper_apple = 1040
    AVG_SAMPLING = 239

    resampled_data = np.concatenate(
        [
            data.values[:upper_leek, :][
                np.random.choice(
                    AVG_SAMPLING, int(AVG_SAMPLING * subsample_rate), replace=False
                )
            ],
            data.values[upper_leek:upper_mushroom, :][
                np.random.choice(
                    AVG_SAMPLING, int(AVG_SAMPLING * subsample_rate), replace=False
                )
            ],
            data.values[upper_mushroom:upper_broccoli, :][
                np.random.choice(
                    AVG_SAMPLING, int(AVG_SAMPLING * subsample_rate), replace=False
                )
            ],
            data.values[upper_broccoli:upper_apple, :][
                np.random.choice(
                    AVG_SAMPLING, int(AVG_SAMPLING * subsample_rate), replace=False
                )
            ],
        ]
    )

    x = resampled_data[:, 2:]
    y = resampled_data[:, 1].reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=SEED // (num_experiment + 1)
    )

    return (
        x_train.astype(np.float32),
        y_train.astype(np.float32),
        x_test.astype(np.float32),
        y_test.astype(np.float32),
    )
