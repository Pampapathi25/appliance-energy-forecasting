from pathlib import Path

RANDOM_STATE = 42
TARGET = "Appliances"
FORECAST_HORIZON = 24
TEST_DAYS = 14
TEST_STEPS = TEST_DAYS * FORECAST_HORIZON
DAILY_SEASONALITY = 24
WEEKLY_SEASONALITY = 168

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "model_objects"

for directory in [PROCESSED_DIR, FIGURE_DIR, FORECAST_DIR, METRICS_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
