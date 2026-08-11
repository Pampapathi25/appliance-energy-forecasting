import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from appliance_energy.config import FORECAST_HORIZON, TEST_STEPS, RANDOM_STATE
from appliance_energy.features import create_ml_dataset, get_feature_columns
from appliance_energy.models.feature_models import create_xgboost_model
from appliance_energy.evaluation import evaluate_forecast

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "appliance_hourly.csv"


def main():
    data = pd.read_csv(DATA_FILE, index_col="date", parse_dates=True)
    featured = create_ml_dataset(data)

    test_start = featured.index[-TEST_STEPS]
    train = featured.loc[featured.index < test_start]
    test = featured.loc[featured.index >= test_start].iloc[:FORECAST_HORIZON]

    features = get_feature_columns(featured)
    X_train = train[features]
    y_train = train["Appliances"]
    X_test = test[features]
    y_test = test["Appliances"]

    model = create_xgboost_model(RANDOM_STATE)
    model.fit(X_train, y_train)
    prediction = pd.Series(model.predict(X_test), index=X_test.index, name="xgboost")

    metrics = evaluate_forecast("xgboost", y_test, prediction, y_train)
    pd.DataFrame([metrics]).to_csv(
        PROJECT_ROOT / "outputs" / "metrics" / "xgboost_metrics.csv",
        index=False,
    )

    pd.DataFrame({"actual": y_test, "forecast": prediction}).to_csv(
        PROJECT_ROOT / "outputs" / "forecasts" / "xgboost_forecast.csv"
    )

    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    importance.head(20).sort_values().plot(kind="barh", figsize=(10, 8))
    plt.title("Top XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs" / "figures" / "xgboost_feature_importance.png", dpi=300)
    plt.close()

    print(pd.DataFrame([metrics]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
