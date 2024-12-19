from sklearn import metrics
import numpy as np


def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def adjusted_r2(y_test, y_pred, x):
    return 1 - (
        (1 - metrics.r2_score(y_test, y_pred))
        * ((x.shape[0] - 1) / (x.shape[0] - x.shape[1] - 1))
    )


def compute_metrics(model, train_yield, train_ndvi_input, test_yield, test_ndvi_input):

    print("Computing Metrics")
    print(
        f"Max. Test Yield: {np.max(test_yield):.3f}; Min. Test Yield: {np.min(test_yield):.3f}"
    )
    train_predictions = model.predict(train_ndvi_input)
    train_r2_score = metrics.r2_score(train_yield, train_predictions)
    print(f"Train R2 score: {train_r2_score:.3f}")
    test_predictions = model.predict(test_ndvi_input)
    test_r2_score = metrics.r2_score(test_yield, test_predictions)
    print(f"Test R2 score: {test_r2_score:.3f}")
    adjusted_train_r2_score = adjusted_r2(
         train_yield,
         train_predictions,
         train_ndvi_input,
     )
    
    adjusted_test_r2_score = adjusted_r2(
        test_yield,
        test_predictions,
        test_ndvi_input,
    )

    train_rmse = metrics.root_mean_squared_error(
        train_yield, train_predictions
    )
    test_rmse = metrics.root_mean_squared_error(test_yield, test_predictions)

    # alt_train_rmse = compute_rmse(train_yield, train_predictions)
    # alt_test_rmse = compute_rmse(test_yield, test_predictions)

    return [
        round(train_r2_score, 5),
        round(test_r2_score, 5),
        round(adjusted_train_r2_score, 5),
        round(adjusted_test_r2_score, 5),
        round(train_rmse, 5),
        round(test_rmse, 5),
        # round(alt_train_rmse, 5),
        # round(alt_test_rmse, 5),
    ]
