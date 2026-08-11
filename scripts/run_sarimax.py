import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from appliance_energy.config import DAILY_SEASONALITY, FORECAST_HORIZON, TEST_STEPS
from appliance_energy.models.sarimax import search_sarimax_parameters, fit_best_sarimax

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "appliance_hourly.csv"


def main():
    data = pd.read_csv(DATA_FILE, index_col="date", parse_dates=True)
    target = data["Appliances"]
    train = target.iloc[:-TEST_STEPS]

    print("Searching 147 SARIMAX configurations...")
    results = search_sarimax_parameters(train, DAILY_SEASONALITY)
    results.to_csv(PROJECT_ROOT / "outputs" / "metrics" / "sarimax_aic_search.csv", index=False)

    print("\nTop AIC models:")
    print(results.head(10).round(3).to_string(index=False))

    model = fit_best_sarimax(train, results, DAILY_SEASONALITY)
    print("\nBest order:", model.model.order)
    print("Best seasonal order:", model.model.seasonal_order)
    print("AIC:", model.aic)

    forecast = model.get_forecast(steps=FORECAST_HORIZON)
    mean = forecast.predicted_mean
    conf = forecast.conf_int()

    forecast_df = pd.DataFrame({
        "forecast": mean,
        "lower": conf.iloc[:, 0],
        "upper": conf.iloc[:, 1],
    })
    forecast_df.to_csv(PROJECT_ROOT / "outputs" / "forecasts" / "sarimax_forecast.csv")

    residuals = model.resid.dropna()
    residuals.to_csv(PROJECT_ROOT / "outputs" / "metrics" / "sarimax_residuals.csv")

    lb = acorr_ljungbox(residuals, lags=[24, 48], return_df=True)
    lb.to_csv(PROJECT_ROOT / "outputs" / "metrics" / "sarimax_ljung_box.csv")

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_acf(residuals, lags=72, ax=ax)
    ax.set_title("ACF of SARIMAX Residuals")
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "outputs" / "figures" / "sarimax_residual_acf.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    mean.plot(ax=ax, label="SARIMAX forecast")
    ax.fill_between(conf.index, conf.iloc[:, 0], conf.iloc[:, 1], alpha=0.2, label="95% CI")
    ax.set_title("SARIMAX 24-Hour Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliance Energy Use")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "outputs" / "figures" / "sarimax_forecast.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
