import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(series):
    series = pd.Series(series).dropna()
    result = adfuller(series)
    return {
        "test": "ADF",
        "statistic": result[0],
        "p_value": result[1],
        "lags": result[2],
        "observations": result[3],
    }


def kpss_test(series):
    series = pd.Series(series).dropna()
    result = kpss(series, regression="c", nlags="auto")
    return {
        "test": "KPSS",
        "statistic": result[0],
        "p_value": result[1],
        "lags": result[2],
    }
