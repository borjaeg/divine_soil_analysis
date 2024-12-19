from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression
import heapq
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


def select_features(
    train_input_scaled,
    test_input_scaled,
    train_yield,
    period_names,
    forced_column_ix,
    feature_selection_algorithm: str,
    features_to_select: int,
    rfe_percent_features_to_select: float,
):
    print("[INFO] Before Feature Selection")
    if train_input_scaled.shape[1] > 1:
        rfe = RFECV(
            estimator=SVR(kernel="linear"),
            cv=3,
            min_features_to_select=int(
                train_input_scaled.shape[1] * rfe_percent_features_to_select
            ),
            step=1,
            scoring="r2",
            verbose=0,
        )

        train_input_rfe = rfe.fit_transform(train_input_scaled, train_yield)
        test_input_rfe = rfe.transform(test_input_scaled)
    else:
        train_input_rfe = train_input_scaled
        test_input_rfe = test_input_scaled

    if feature_selection_algorithm == "mutual-info-regression":
        selector = SelectKBest(
            mutual_info_regression,
            k=features_to_select,
        )
        scores = mutual_info_regression(train_input_rfe, train_yield)
        best_features_mask = [
            i
            for x, i in heapq.nlargest(
                features_to_select,
                ((x, i) for i, x in enumerate(scores)),
            )
        ]
    elif feature_selection_algorithm == "f-regression":
        selector = SelectKBest(
            f_regression,
            k=features_to_select,
        )
        scores = f_regression(train_input_rfe, train_yield)[0]
        best_features_mask = [
            i
            for x, i in heapq.nlargest(
                features_to_select,
                ((x, i) for i, x in enumerate(scores)),
            )
        ]
    elif feature_selection_algorithm == "permutation":
        ard = ARDRegression()
        ard.fit(train_input_rfe, train_yield)
        ard_importances = permutation_importance(
            ard, train_input_rfe, train_yield, n_repeats=10, random_state=2024
        )
        scores = ard_importances.importances_mean
        best_features_mask = [
            i
            for x, i in heapq.nlargest(
                features_to_select,
                ((x, i) for i, x in enumerate(scores)),
            )
        ]
    elif feature_selection_algorithm == "random":
        best_features_mask = np.random.choice(train_input_rfe.shape[1], features_to_select, replace=False).tolist()
        
    elif feature_selection_algorithm == "union":
        selector = SelectKBest(
            f_regression,
            k=features_to_select,
        )
        mi = mutual_info_regression(train_input_rfe, train_yield)
        f_regr = f_regression(train_input_rfe, train_yield)[0]
        mi /= np.max(mi)
        f_regr /= np.max(f_regr)

        mi_best_features_mask = [
            i
            for x, i in heapq.nlargest(
                features_to_select,
                ((x, i) for i, x in enumerate(mi)),
            )
        ]
        f_regr_best_features_mask = [
            i
            for x, i in heapq.nlargest(
                features_to_select,
                ((x, i) for i, x in enumerate(f_regr)),
            )
        ]
        best_features_mask = list(
            set(mi_best_features_mask).union(set(f_regr_best_features_mask))
        )
    elif feature_selection_algorithm == "intersection":
        selector = SelectKBest(
            f_regression,
            k=features_to_select,
        )
        mi = mutual_info_regression(train_input_rfe, train_yield)
        f_regr = f_regression(train_input_rfe, train_yield)[0]
        mi /= np.max(mi)
        f_regr /= np.max(f_regr)

        mi_best_features_mask = [
            i
            for x, i in heapq.nlargest(
                features_to_select,
                ((x, i) for i, x in enumerate(mi)),
            )
        ]
        f_regr_best_features_mask = [
            i
            for x, i in heapq.nlargest(
                features_to_select,
                ((x, i) for i, x in enumerate(f_regr)),
            )
        ]
        best_features_mask = list(
            set(mi_best_features_mask).intersection(set(f_regr_best_features_mask))
        )

    if feature_selection_algorithm != "none" and train_input_scaled.shape[1] != 1:
        param_grid = {
            "n_estimators": [100, 500],
            "learning_rate": [0.1, 0.01],
            "max_depth": [3, 5],
            "subsample": [1.0, 0.6],
            "max_features": ["auto", "sqrt", "log2"],
        }
        gbr = GradientBoostingRegressor(random_state=2024)
        grid_search = GridSearchCV(
            estimator=gbr,
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=2,
            n_jobs=-1,
            verbose=1,
        )

        # Perform grid search on the training data
        grid_search.fit(train_input_rfe, train_yield)

        feature_importances = grid_search.best_estimator_.feature_importances_
        best_features_mask_2 = [
            i
            for x, i in heapq.nlargest(
                features_to_select,
                ((x, i) for i, x in enumerate(feature_importances)),
            )
        ]

        train_input_sel = train_input_rfe[:, sorted(best_features_mask)]
        test_input_sel = test_input_rfe[:, sorted(best_features_mask)]

    else:
        best_indices = []

        train_input_sel = train_input_rfe
        test_input_sel = test_input_rfe
        best_features_mask = []
        best_features_mask_2 = []

    if forced_column_ix is not None:
        train_input_sel = np.concatenate(
            [train_input_sel, train_input_scaled[:, forced_column_ix].reshape(-1, 1)],
            axis=1,
        )
        test_input_sel = np.concatenate(
            [test_input_sel, test_input_scaled[:, forced_column_ix].reshape(-1, 1)],
            axis=1,
        )

    print("[INFO] After Feature Selection")

    return (
        train_input_sel,
        test_input_sel,
        best_features_mask,
        best_features_mask_2,
    )


def get_best_indices(best_features_mask, vi_names):

    return list(np.array(vi_names)[best_features_mask])
