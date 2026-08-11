import pandas as pd

from appliance_energy.models.benchmarks import naive_forecast, seasonal_naive_forecast


def test_naive_forecast_length():
    train = pd.Series([1, 2, 3, 4])
    index = pd.date_range("2026-01-01", periods=3, freq="h")
    forecast = naive_forecast(train, 3, index)
    assert len(forecast) == 3


def test_naive_forecast_value():
    train = pd.Series([1, 2, 3, 4])
    index = pd.date_range("2026-01-01", periods=3, freq="h")
    forecast = naive_forecast(train, 3, index)
    assert (forecast == 4).all()


def test_seasonal_forecast_length():
    train = pd.Series(range(48))
    index = pd.date_range("2026-01-01", periods=24, freq="h")
    forecast = seasonal_naive_forecast(train, 24, index, 24)
    assert len(forecast) == 24
