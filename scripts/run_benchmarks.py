import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from appliance_energy.config import DAILY_SEASONALITY, FORECAST_HORIZON, TEST_STEPS, WEEKLY_SEASONALITY
from appliance_energy.models.benchmarks import drift_forecast, mean_forecast, naive_forecast, seasonal_naive_forecast
from appliance_energy.evaluation import evaluate_forecast

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "appliance_hourly.csv"


def main():
    data = pd.read_csv(DATA_FILE, index_col="date", parse_dates=True)
    target = data["Appliances"]
    test = target.iloc[-TEST_STEPS:]
    train = target.iloc[:-TEST_STEPS]
    horizon = FORECAST_HORIZON
    index = test.index[:horizon]
    actual = test.iloc[:horizon]

    forecasts = {
        "mean": mean_forecast(train, horizon, index),
        "naive": naive_forecast(train, horizon, index),
        "daily_seasonal_naive": seasonal_naive_forecast(train, horizon, index, DAILY_SEASONALITY),
        "weekly_seasonal_naive": seasonal_naive_forecast(train, horizon, index, WEEKLY_SEASONALITY),
        "drift": drift_forecast(train, horizon, index),
    }

    results = [
        evaluate_forecast(name, actual, pred, train)
        for name, pred in forecasts.items()
    ]

    output = pd.DataFrame(results)
    output_path = PROJECT_ROOT / "outputs" / "metrics" / "benchmark_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(output.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
