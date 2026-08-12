import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)


# ============================================================
# PROJECT CONFIG
# ============================================================

from appliance_energy.config import (
    FORECAST_HORIZON,
    TEST_STEPS,
)


# ============================================================
# PATHS
# ============================================================

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "appliance_hourly.csv"
)

FORECAST_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "forecasts"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)


# ============================================================
# MASE
# ============================================================

def calculate_mase(
    actual,
    predicted,
    training_data,
    seasonality=24,
):
    """
    Calculate Mean Absolute Scaled Error.

    The scale is based on the in-sample seasonal naive
    forecast error.
    """

    training_data = pd.Series(
        training_data
    ).astype(float)

    seasonal_error = np.abs(
        training_data.iloc[seasonality:].values
        -
        training_data.iloc[:-seasonality].values
    )

    scale = seasonal_error.mean()

    if scale == 0:
        return np.nan

    forecast_error = np.mean(
        np.abs(
            np.asarray(actual)
            -
            np.asarray(predicted)
        )
    )

    return forecast_error / scale


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("CHRONOS FOUNDATION MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. IMPORT CHRONOS
    # --------------------------------------------------------

    print("\n[1/10] Importing Chronos...")

    try:
        import torch

        from chronos import ChronosPipeline

    except ImportError as error:

        print("\nChronos/PyTorch could not be imported.")
        print("\nInstall using:")
        print("pip install torch chronos-forecasting")

        print("\nOriginal error:")
        print(error)

        sys.exit(1)

    print("Chronos imported successfully.")


    # --------------------------------------------------------
    # 2. CHECK INPUT DATA
    # --------------------------------------------------------

    print("\n[2/10] Checking processed dataset...")

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            "\nProcessed dataset does not exist:\n"
            f"{DATA_FILE}\n\n"
            "Run this first:\n"
            "python scripts/prepare_data.py"
        )


    # --------------------------------------------------------
    # 3. LOAD DATA
    # --------------------------------------------------------

    print("\n[3/10] Loading hourly data...")

    data = pd.read_csv(
        DATA_FILE,
        index_col="date",
        parse_dates=True,
    )

    data = data.sort_index()

    if "Appliances" not in data.columns:

        raise KeyError(
            "'Appliances' column is missing "
            "from appliance_hourly.csv."
        )

    target = (
        data["Appliances"]
        .astype(float)
        .dropna()
    )

    print(
        f"Observations: {len(target)}"
    )

    print(
        f"Start: {target.index.min()}"
    )

    print(
        f"End:   {target.index.max()}"
    )


    # --------------------------------------------------------
    # 4. TRAIN / TEST SPLIT
    # --------------------------------------------------------

    print("\n[4/10] Creating train/test split...")

    if len(target) <= TEST_STEPS:

        raise ValueError(
            "There are not enough observations "
            "for the configured test period."
        )

    train = target.iloc[
        :-TEST_STEPS
    ]

    test = target.iloc[
        -TEST_STEPS:
    ]

    actual = test.iloc[
        :FORECAST_HORIZON
    ]

    print(
        f"Training observations: {len(train)}"
    )

    print(
        f"Test observations: {len(test)}"
    )

    print(
        f"Forecast horizon: {FORECAST_HORIZON} hours"
    )


    # --------------------------------------------------------
    # 5. CREATE CHRONOS CONTEXT
    # --------------------------------------------------------

    print("\n[5/10] Preparing Chronos context...")

    # Use the latest 7 days of hourly history.
    # 7 days x 24 hours = 168 observations.

    CONTEXT_LENGTH = min(
        7 * 24,
        len(train),
    )

    context = train.iloc[
        -CONTEXT_LENGTH:
    ]

    print(
        f"Context length: {CONTEXT_LENGTH} hours"
    )


    # --------------------------------------------------------
    # 6. LOAD PRETRAINED CHRONOS MODEL
    # --------------------------------------------------------

    print("\n[6/10] Loading pretrained Chronos model...")

    MODEL_NAME = (
        "amazon/chronos-t5-small"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    pipeline = (
        ChronosPipeline
        .from_pretrained(
            MODEL_NAME,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    )

    print(
        "Chronos model loaded successfully."
    )


    # --------------------------------------------------------
    # 7. PREPARE INPUT TENSOR
    # --------------------------------------------------------

    context_tensor = torch.tensor(
        context.values,
        dtype=torch.float32,
    )

    print(
        f"Input tensor shape: "
        f"{context_tensor.shape}"
    )


    # --------------------------------------------------------
    # 8. GENERATE PROBABILISTIC FORECAST
    # --------------------------------------------------------

    print(
        "\n[7/10] Generating "
        "24-hour probabilistic forecast..."
    )

    NUM_SAMPLES = 100

    forecast = pipeline.predict(
        context_tensor,
        prediction_length=FORECAST_HORIZON,
        num_samples=NUM_SAMPLES,
    )

    print(
        f"Raw Chronos forecast shape: "
        f"{tuple(forecast.shape)}"
    )


    # --------------------------------------------------------
    # IMPORTANT
    #
    # ChronosPipeline.predict returns:
    #
    # (batch_size, num_samples, prediction_length)
    #
    # We have one time series, therefore:
    #
    # forecast[0]
    #
    # becomes:
    #
    # (num_samples, prediction_length)
    # --------------------------------------------------------

    forecast_samples = (
        forecast[0]
        .detach()
        .cpu()
        .numpy()
    )

    print(
        f"Forecast samples shape: "
        f"{forecast_samples.shape}"
    )


    # --------------------------------------------------------
    # 9. POINT FORECAST + PREDICTION INTERVAL
    # --------------------------------------------------------

    print(
        "\n[8/10] Calculating "
        "median and prediction intervals..."
    )

    # Median used as point forecast.
    point_forecast = np.median(
        forecast_samples,
        axis=0,
    )

    # 90% prediction interval.
    lower_90 = np.quantile(
        forecast_samples,
        0.05,
        axis=0,
    )

    upper_90 = np.quantile(
        forecast_samples,
        0.95,
        axis=0,
    )


    # --------------------------------------------------------
    # ALIGN WITH TEST INDEX
    # --------------------------------------------------------

    forecast_index = actual.index

    forecast_df = pd.DataFrame(
        {
            "actual":
                actual.values,

            "forecast":
                point_forecast,

            "lower_90":
                lower_90,

            "upper_90":
                upper_90,
        },
        index=forecast_index,
    )


    # --------------------------------------------------------
    # 10. EVALUATION METRICS
    # --------------------------------------------------------

    print(
        "\n[9/10] Calculating evaluation metrics..."
    )

    mae = mean_absolute_error(
        actual.values,
        point_forecast,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual.values,
            point_forecast,
        )
    )

    mase = calculate_mase(
        actual=actual.values,
        predicted=point_forecast,
        training_data=train,
        seasonality=24,
    )

    bias = np.mean(
        point_forecast
        -
        actual.values
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHRONOS RESULTS")
    print("=" * 70)

    print(
        f"Model:              {MODEL_NAME}"
    )

    print(
        f"Context length:     "
        f"{CONTEXT_LENGTH} hours"
    )

    print(
        f"Forecast horizon:   "
        f"{FORECAST_HORIZON} hours"
    )

    print(
        f"Forecast samples:   "
        f"{NUM_SAMPLES}"
    )

    print(
        f"MAE:                "
        f"{mae:.4f}"
    )

    print(
        f"RMSE:               "
        f"{rmse:.4f}"
    )

    print(
        f"MASE:               "
        f"{mase:.4f}"
    )

    print(
        f"Bias:               "
        f"{bias:.4f}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------------

    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # SAVE FORECAST
    # --------------------------------------------------------

    forecast_path = (
        FORECAST_DIR
        / "chronos_forecast.csv"
    )

    forecast_df.to_csv(
        forecast_path
    )


    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        {
            "model": [
                "Chronos"
            ],

            "model_name": [
                MODEL_NAME
            ],

            "MAE": [
                mae
            ],

            "RMSE": [
                rmse
            ],

            "MASE": [
                mase
            ],

            "Bias": [
                bias
            ],

            "forecast_horizon": [
                FORECAST_HORIZON
            ],

            "context_length": [
                CONTEXT_LENGTH
            ],

            "num_samples": [
                NUM_SAMPLES
            ],
        }
    )

    metrics_path = (
        METRICS_DIR
        / "chronos_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )


    # --------------------------------------------------------
    # SAVE FORECAST PLOT
    # --------------------------------------------------------

    print(
        "\n[10/10] Creating Chronos forecast plot..."
    )

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        forecast_df.index,
        forecast_df["actual"],
        label="Actual",
        linewidth=2,
    )

    ax.plot(
        forecast_df.index,
        forecast_df["forecast"],
        label="Chronos forecast",
        linewidth=2,
    )

    ax.fill_between(
        forecast_df.index,
        forecast_df["lower_90"],
        forecast_df["upper_90"],
        alpha=0.2,
        label="90% prediction interval",
    )

    ax.set_title(
        "Chronos 24-Hour "
        "Appliance Energy Forecast"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Appliance Energy Use"
    )

    ax.legend()

    plt.tight_layout()

    plot_path = (
        FIGURE_DIR
        / "chronos_forecast.png"
    )

    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHRONOS FOUNDATION MODEL COMPLETE")
    print("=" * 70)

    print(
        "\nForecast saved:"
    )

    print(
        forecast_path
    )

    print(
        "\nMetrics saved:"
    )

    print(
        metrics_path
    )

    print(
        "\nFigure saved:"
    )

    print(
        plot_path
    )

    print("\nPerformance:")

    print(
        f"MAE  = {mae:.4f}"
    )

    print(
        f"RMSE = {rmse:.4f}"
    )

    print(
        f"MASE = {mase:.4f}"
    )

    print(
        f"Bias = {bias:.4f}"
    )

    print("=" * 70)


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()