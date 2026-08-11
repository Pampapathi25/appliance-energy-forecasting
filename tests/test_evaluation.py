import pandas as pd

from appliance_energy.evaluation import calculate_mae, calculate_rmse, calculate_bias


def test_metrics_zero_error():
    actual = pd.Series([1, 2, 3])
    predicted = pd.Series([1, 2, 3])
    assert calculate_mae(actual, predicted) == 0
    assert calculate_rmse(actual, predicted) == 0
    assert calculate_bias(actual, predicted) == 0
