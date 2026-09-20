"""Small helpers for regression evaluation (Table 8).

Every metric returned here is computed on the original gold-price scale; mape_percent
is a percentage (2.5 means 2.5%)."""

import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def _one_dimensional_finite(values, name):
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def regression_metrics_original_price(
    y_true_price,
    predictions,
    *,
    prediction_scaler=None,
):
    """Return original-price predictions, RMSE (USD/oz) and MAPE (%).

    If the predictions are standardized, pass the prediction_scaler fit on the matching
    training fold. Leave it as None when they are already in price units.
    """
    true_price = _one_dimensional_finite(y_true_price, "y_true_price")
    prediction_values = _one_dimensional_finite(predictions, "predictions")

    if prediction_scaler is None:
        predicted_price = prediction_values
    else:
        predicted_price = prediction_scaler.inverse_transform(
            prediction_values.reshape(-1, 1)
        ).reshape(-1)

    predicted_price = _one_dimensional_finite(predicted_price, "predicted_price")
    if true_price.shape != predicted_price.shape:
        raise ValueError("True prices and predictions have different lengths")

    rmse_usd_per_oz = float(np.sqrt(mean_squared_error(true_price, predicted_price)))
    mape_percent = float(
        mean_absolute_percentage_error(true_price, predicted_price) * 100.0
    )
    return predicted_price, rmse_usd_per_oz, mape_percent
