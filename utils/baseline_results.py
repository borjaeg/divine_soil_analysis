from sklearn.linear_model import ARDRegression
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance

from utils.metrics_utils import compute_metrics
from config import POLYNOMIAL_DEGREE
from config import USE_POLYNOMIAL
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import SGDRegressor
import numpy as np


def round_coefficients(original_coefficients):
    coefs = []
    for coef in original_coefficients:
        if coef < 0.001:
            coefs.append(0)
        else:
            coefs.append(round(coef, 4))
    return coefs


def compute_baseline_results(
    train_ndvi_input, train_yield, test_ndvi_input, test_yield
):
    
    #print("Number of NaNs:", np.isnan(test_ndvi_input).sum())

    # Check for infinity values
    print("Number of infinities:", np.isinf(test_ndvi_input).sum())
    if np.isinf(test_ndvi_input).sum() > 0:
        test_ndvi_input = np.nan_to_num(test_ndvi_input.copy(), nan=0.0, posinf=0.0, neginf=0.0)
    if np.isinf(train_ndvi_input).sum() > 0:
        train_ndvi_input = np.nan_to_num(test_ndvi_input.copy(), nan=0.0, posinf=0.0, neginf=0.0)

    # Check for extremely large values
    #print("Max value:", np.max(test_ndvi_input))
    #print("Min value:", np.min(test_ndvi_input))

    if USE_POLYNOMIAL:
        poly = PolynomialFeatures(degree=POLYNOMIAL_DEGREE, include_bias=False)
        poly_train_ndvi_input = poly.fit_transform(train_ndvi_input.copy())
        poly_test_ndvi_input = poly.transform(test_ndvi_input.copy())

    else:
        poly = PolynomialFeatures(degree=1, include_bias=False)
        poly_train_ndvi_input = poly.fit_transform(train_ndvi_input.copy())
        poly_test_ndvi_input = poly.transform(test_ndvi_input.copy())

    ard = ARDRegression()
    ard.fit(train_ndvi_input.copy(), train_yield.copy())

    (ard_results) = compute_metrics(
        ard,
        train_yield,
        train_ndvi_input,
        test_yield,
        test_ndvi_input,
    )

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)

    mlp.fit(train_ndvi_input.copy(), train_yield.copy())

    (mlp_results) = compute_metrics(
        mlp,
        train_yield,
        train_ndvi_input,
        test_yield,
        test_ndvi_input,
    )

    poly_ard = ARDRegression()
    poly_ard.fit(poly_train_ndvi_input, train_yield)

    (poly_ard_results) = compute_metrics(
        poly_ard,
        train_yield,
        poly_train_ndvi_input,
        test_yield,
        poly_test_ndvi_input,
    )

    ard_with_transformed_target = TransformedTargetRegressor(
        regressor=ARDRegression(),
        transformer=QuantileTransformer(
            n_quantiles=len(train_yield) // 4, output_distribution="normal"
        ),
    ).fit(train_ndvi_input, train_yield)

    (trans_ard_results) = compute_metrics(
        ard_with_transformed_target,
        train_yield,
        train_ndvi_input,
        test_yield,
        test_ndvi_input,
    )

    poly_ard_with_transformed_target = TransformedTargetRegressor(
        regressor=ARDRegression(),
        transformer=QuantileTransformer(
            n_quantiles=len(train_yield) // 4, output_distribution="normal"
        ),
    ).fit(poly_train_ndvi_input, train_yield)

    (trans_poly_ard_results) = compute_metrics(
        poly_ard_with_transformed_target,
        train_yield,
        poly_train_ndvi_input,
        test_yield,
        poly_test_ndvi_input,
    )

    if train_ndvi_input.shape[1] > 1:
        poly_ard_pipe = Pipeline(
            [
                ("select", SelectKBest(mutual_info_regression)),
                ("pca", PCA()),
                ("ard", ARDRegression()),
            ]
        )

        params = [
            {
                "select__k": [
                    poly_train_ndvi_input.shape[1] // 2,
                    poly_train_ndvi_input.shape[1] // 3,
                ],
                "pca__n_components": [
                    poly_train_ndvi_input.shape[1] // 4,
                    poly_train_ndvi_input.shape[1] // 6,
                ],
            }
        ]
        poly_ard_pipe_grid = GridSearchCV(
            poly_ard_pipe, param_grid=params, scoring="r2", cv=3, refit=True
        )

        pls_pipe = Pipeline([("pls", PLSRegression())])

        params = [
            {
                "pls__n_components": [
                    train_ndvi_input.shape[1] // 2,
                    train_ndvi_input.shape[1],
                ]
            }
        ]
        pls_pipe_grid = GridSearchCV(
            pls_pipe, param_grid=params, scoring="r2", cv=3, refit=True
        )

        pls_pipe_grid.fit(train_ndvi_input, train_yield)

        (pls_results) = compute_metrics(
            pls_pipe_grid.best_estimator_,
            train_yield,
            train_ndvi_input,
            test_yield,
            test_ndvi_input,
        )

        poly_pls_pipe = Pipeline(
            [("poly", PolynomialFeatures(POLYNOMIAL_DEGREE)), ("pls", PLSRegression())]
        )

        params = [
            {
                "pls__n_components": [
                    train_ndvi_input.shape[1] // 2,
                    train_ndvi_input.shape[1],
                ]
            }
        ]
        poly_pls_pipe_grid = GridSearchCV(
            poly_pls_pipe, param_grid=params, scoring="r2", cv=3, refit=True
        )

        poly_pls_pipe_grid.fit(train_ndvi_input, train_yield)

        (poly_pls_results) = compute_metrics(
            poly_pls_pipe_grid.best_estimator_,
            train_yield,
            train_ndvi_input,
            test_yield,
            test_ndvi_input,
        )

        # spline_pls_pipe = Pipeline(
        #    [("spline", SplineTransformer()), ("pls", PLSRegression())]
        # )

        params = [
            {
                "spline__n_knots": [4, 6, 8, 20],
                "spline__degree": [2, 3],
                "pls__n_components": [
                    train_ndvi_input.shape[1] // 2,
                    train_ndvi_input.shape[1],
                ],
            }
        ]

        bin_pls = make_pipeline(
            KBinsDiscretizer(n_bins=10, encode="onehot-dense"),
            PLSRegression(n_components=train_ndvi_input.shape[1] // 2),
        )
        bin_pls.fit(train_ndvi_input, train_yield)

        (bin_pls_results) = compute_metrics(
            bin_pls,
            train_yield,
            train_ndvi_input,
            test_yield,
            test_ndvi_input,
        )

        pls_aux = PLSRegression(n_components=train_ndvi_input.shape[1])
        pls_aux.fit(train_ndvi_input, train_yield)
        best_pls_indexes = np.argsort(np.abs(pls_aux.coef_))

        ard_importances = permutation_importance(
            ard, train_ndvi_input, train_yield, n_repeats=10, random_state=2024
        )
        ard_mean_importances = (ard_importances.importances_mean,)
        # spline_pls_pipe_grid = GridSearchCV(
        #    spline_pls_pipe, param_grid=params, scoring="r2", cv=3, refit=True
        # )

        # spline_pls_pipe_grid.fit(train_ndvi_input.copy(), train_yield.copy())

        # (spline_pls_results) = compute_metrics(
        #    spline_pls_pipe_grid.best_estimator_,
        #    train_yield,
        #    train_ndvi_input,
        #    test_yield,
        #    test_ndvi_input,
        # )

        # poly_ard_pipe_grid.fit(poly_train_ndvi_input.copy(), train_yield.copy())

        # (poly_ard_pipe_results) = compute_metrics(
        #    poly_ard_pipe_grid.best_estimator_,
        #    train_yield,
        #    poly_train_ndvi_input,
        #    test_yield,
        #    poly_test_ndvi_input,
        # )

        # spline_ard_pipe = Pipeline(
        #    [("spline", SplineTransformer()), ("ard", ARDRegression())]
        # )

        # params = [{"spline__n_knots": [4, 6, 8, 20], "spline__degree": [2, 3]}]
        # spline_ard_pipe_grid = GridSearchCV(
        #    spline_ard_pipe, param_grid=params, scoring="r2", cv=3, refit=True
        # )

        # spline_ard_pipe_grid.fit(train_ndvi_input.copy(), train_yield.copy())

        # (spline_ard_results) = compute_metrics(
        #    spline_ard_pipe_grid.best_estimator_,
        #    train_yield,
        #    train_ndvi_input,
        #    test_yield,
        #    test_ndvi_input,
        # )
    else:
        pls_results = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        poly_pls_results = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        bin_pls_results = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        best_pls_indexes = None
        ard_mean_importances = None

    ridge = Ridge()
    ridge.fit(train_ndvi_input, train_yield)

    (ridge_results) = compute_metrics(
        ridge,
        train_yield,
        train_ndvi_input,
        test_yield,
        test_ndvi_input,
    )

    poly_svr = SVR(kernel="linear")
    poly_svr.fit(train_ndvi_input, train_yield)

    (poly_svr_results) = compute_metrics(
        poly_svr,
        train_yield,
        train_ndvi_input,
        test_yield,
        test_ndvi_input,
    )

    bayesian = BayesianRidge()
    bayesian.fit(train_ndvi_input, train_yield)

    (bayesian_results) = compute_metrics(
        bayesian,
        train_yield,
        train_ndvi_input,
        test_yield,
        test_ndvi_input,
    )

    bin_ard = make_pipeline(
        KBinsDiscretizer(n_bins=10, encode="onehot-dense"), ARDRegression()
    )
    bin_ard.fit(train_ndvi_input, train_yield)

    (bin_ard_results) = compute_metrics(
        bin_ard,
        train_yield,
        train_ndvi_input,
        test_yield,
        test_ndvi_input,
    )

    FIRST_SAMPLE = 50
    SECOND_SAMPLE = 100
    THIRD_SAMPLE = 150

    try:
        partial_learner = make_pipeline(
            PolynomialFeatures(degree=POLYNOMIAL_DEGREE), ARDRegression()
        )
        partial_learner.fit(train_ndvi_input, train_yield)

        (partial_learner_1_results) = compute_metrics(
            partial_learner,
            train_yield,
            train_ndvi_input,
            test_yield,
            test_ndvi_input,
        )
        x_train = np.vstack([train_ndvi_input, test_ndvi_input[:FIRST_SAMPLE]])
        y_train = np.vstack(
            [train_yield.reshape(-1, 1), test_yield[:FIRST_SAMPLE].reshape(-1, 1)]
        )
        partial_learner.fit(x_train, y_train)
        print(f"1: {x_train.shape}-{y_train.shape}")

        (partial_learner_2_results) = compute_metrics(
            partial_learner,
            y_train,
            x_train,
            test_yield[FIRST_SAMPLE:],
            test_ndvi_input[FIRST_SAMPLE:],
        )

        x_train = np.vstack([train_ndvi_input, test_ndvi_input[:SECOND_SAMPLE]])
        y_train = np.vstack(
            [train_yield.reshape(-1, 1), test_yield[:SECOND_SAMPLE].reshape(-1, 1)]
        )
        partial_learner.fit(x_train, y_train)
        print(f"2: {x_train.shape}-{y_train.shape}")

        (partial_learner_3_results) = compute_metrics(
            partial_learner,
            y_train,
            x_train,
            test_yield[SECOND_SAMPLE:],
            test_ndvi_input[SECOND_SAMPLE:],
        )

        x_train = np.vstack([train_ndvi_input, test_ndvi_input[:THIRD_SAMPLE]])
        y_train = np.vstack(
            [train_yield.reshape(-1, 1), test_yield[:THIRD_SAMPLE].reshape(-1, 1)]
        )
        partial_learner.fit(x_train, y_train)
        print(f"3: {x_train.shape}-{y_train.shape}")

        (partial_learner_4_results) = compute_metrics(
            partial_learner,
            y_train,
            x_train,
            test_yield[THIRD_SAMPLE:],
            test_ndvi_input[THIRD_SAMPLE:],
        )
    except Exception as e:
        print(f"[Catched Exception] {e}")
        partial_learner_1_results = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        partial_learner_2_results = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        partial_learner_3_results = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        partial_learner_4_results = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # write a row to the csv file

    ard_coefs = round_coefficients(ard.coef_)
    ard_coefs.append(round(ard.intercept_, 4))

    # ridge_coefs = []  # round_coefficients(ridge.coef_)
    # ridge_coefs.append(round(ridge.intercept_, 4))
    # lasso_coefs = round_coefficients(lasso.coef_)
    # lasso_coefs.append(round(lasso.intercept_, 4))
    # bayesian_coefs = round_coefficients(bayesian.coef_)
    # bayesian_coefs.append(round(bayesian.intercept_, 4))
    # poly_ard_coefs = round_coefficients(poly_ard.coef_)
    # poly_ard_coefs.append(round(poly_ard.intercept_, 4))
    # pls_coefs = round_coefficients(pls2.coef_)
    # pls_coefs.append(round(pls2.intercept_, 4))
    # poly_pls_coefs = round_coefficients(poly_pls.coef_)
    # poly_pls_coefs.append(round(poly_pls.intercept_, 4))

    return [
        *ard_results,
        ard_coefs,
        *mlp_results,
        *poly_ard_results,
        # poly_ard_coefs,
        # *poly_ard_pipe_results,
        *trans_poly_ard_results,
        *trans_ard_results,
        *ridge_results,
        # ridge_coefs,
        *poly_svr_results,
        [],
        *bayesian_results,
        # bayesian_coefs,
        *pls_results,
        # *poly_pls_results,
        # *spline_ard_results,
        # *spline_pls_results,
        # *bin_ard_results,
        # *bin_pls_results,
        # *partial_learner_1_results,
        # *partial_learner_2_results,
        # *partial_learner_3_results,
        # *partial_learner_4_results,
        # poly_ard_pipe_grid.best_params_,
        # spline_ard_pipe_grid.best_params_,
        # spline_pls_pipe_grid.best_params_,
        # poly_train_ndvi_input.shape,
        # round(poly_ard_pipe_grid.best_score_, 4),
        # best_pls_indexes,
        # ard_mean_importances
    ]
