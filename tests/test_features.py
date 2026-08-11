import pandas as pd

from appliance_energy.features import add_lag_features, add_rolling_features


def test_lag_feature_uses_previous_value():
    index = pd.date_range("2026-01-01", periods=30, freq="h")
    data = pd.DataFrame({"Appliances": range(30)}, index=index)
    result = add_lag_features(data)
    assert result.loc[index[10], "lag_1"] == 9


def test_rolling_feature_is_shifted():
    index = pd.date_range("2026-01-01", periods=30, freq="h")
    data = pd.DataFrame({"Appliances": range(30)}, index=index)
    result = add_rolling_features(data)
    assert pd.isna(result.loc[index[23], "roll_mean_24"])
