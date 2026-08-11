import pandas as pd


def mean_forecast(train, horizon, index):
    return pd.Series(train.mean(), index=index, name="mean")


def naive_forecast(train, horizon, index):
    return pd.Series(train.iloc[-1], index=index, name="naive")


def seasonal_naive_forecast(train, horizon, index, seasonality):
    history = list(train.values)
    predictions = []
    for _ in range(horizon):
        prediction = history[-seasonality]
        predictions.append(prediction)
        history.append(prediction)
    return pd.Series(predictions, index=index, name=f"seasonal_naive_{seasonality}")


def drift_forecast(train, horizon, index):
    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    predictions = [train.iloc[-1] + slope * step for step in range(1, horizon + 1)]
    return pd.Series(predictions, index=index, name="drift")
