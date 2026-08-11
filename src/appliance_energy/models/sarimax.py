import itertools
import warnings
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def search_sarimax_parameters(y, seasonal_period=24):
    results = []
    for p, d, q in itertools.product(range(7), range(3), range(7)):
        order = (p, d, q)
        seasonal_order = (1, 1, 1, seasonal_period)
        try:
            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            results.append({"p": p, "d": d, "q": q, "AIC": fitted.aic})
        except Exception as exc:
            print(f"Failed for {order}: {exc}")
    return pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)


def fit_best_sarimax(y, search_results, seasonal_period=24):
    best = search_results.iloc[0]
    order = (int(best["p"]), int(best["d"]), int(best["q"]))
    seasonal_order = (1, 1, 1, seasonal_period)
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)
