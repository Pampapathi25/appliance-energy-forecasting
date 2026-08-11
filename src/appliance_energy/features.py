import numpy as np


def add_time_features(data):
    data = data.copy()
    data["hour"] = data.index.hour
    data["dayofweek"] = data.index.dayofweek
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7)
    return data


def add_lag_features(data, target="Appliances"):
    data = data.copy()
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        data[f"lag_{lag}"] = data[target].shift(lag)
    return data


def add_rolling_features(data, target="Appliances"):
    data = data.copy()
    shifted = data[target].shift(1)
    for window in [3, 6, 12, 24, 168]:
        data[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        data[f"roll_std_{window}"] = shifted.rolling(window).std()
    return data


def create_ml_dataset(data):
    data = add_time_features(data)
    data = add_lag_features(data)
    data = add_rolling_features(data)
    return data.dropna()


def get_feature_columns(data):
    return [c for c in data.columns if c != "Appliances"]
