import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def calculate_mae(actual, predicted):
    return mean_absolute_error(actual, predicted)


def calculate_rmse(actual, predicted):
    error = np.asarray(actual) - np.asarray(predicted)
    return np.sqrt(np.mean(error ** 2))


def calculate_bias(actual, predicted):
    return np.mean(np.asarray(predicted) - np.asarray(actual))


def calculate_mase(actual, predicted, training_data, seasonality=24):
    training_data = pd.Series(training_data).astype(float)
    seasonal_error = np.abs(
        training_data.iloc[seasonality:].values -
        training_data.iloc[:-seasonality].values
    )
    scale = seasonal_error.mean()
    if scale == 0:
        return np.nan
    return np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))) / scale


def evaluate_forecast(model_name, actual, predicted, training_data):
    return {
        "model": model_name,
        "MAE": calculate_mae(actual, predicted),
        "RMSE": calculate_rmse(actual, predicted),
        "MASE": calculate_mase(actual, predicted, training_data),
        "Bias": calculate_bias(actual, predicted),
    }
