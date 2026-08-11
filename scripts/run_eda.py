import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from appliance_energy.stationarity import adf_test, kpss_test

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "appliance_hourly.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


def main():
    data = pd.read_csv(DATA_FILE, index_col="date", parse_dates=True)
    series = data["Appliances"]

    print("Shape:", data.shape)
    print("\nMissing values:")
    print(data.isna().sum())
    print("\nDescriptive statistics:")
    print(series.describe())
    print("\nADF:")
    print(adf_test(series))
    print("\nKPSS:")
    print(kpss_test(series))

    fig, ax = plt.subplots(figsize=(15, 6))
    series.plot(ax=ax)
    ax.set_title("Hourly Appliance Energy Consumption")
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliance Energy Use")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "appliance_hourly_series.png", dpi=300)
    plt.close(fig)

    hourly_profile = data.groupby(data.index.hour)["Appliances"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly_profile.plot(marker="o", ax=ax)
    ax.set_title("Average Appliance Energy Use by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Appliance Energy Use")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "hourly_profile.png", dpi=300)
    plt.close(fig)

    decomposition = seasonal_decompose(series, model="additive", period=24)
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "seasonal_decomposition.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_acf(series, lags=24 * 7, ax=ax)
    ax.set_title("ACF of Hourly Appliance Energy Use")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "acf.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_pacf(series, lags=72, ax=ax, method="ywm")
    ax.set_title("PACF of Hourly Appliance Energy Use")
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "pacf.png", dpi=300)
    plt.close(fig)

    first_difference = series.diff().dropna()
    print("\nADF after first difference:")
    print(adf_test(first_difference))
    print("\nKPSS after first difference:")
    print(kpss_test(first_difference))


if __name__ == "__main__":
    main()
