import pandas as pd

TARGET = "Appliances"


def load_raw_data(file_path):
    return pd.read_csv(file_path)


def prepare_datetime(data):
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date").set_index("date")


def convert_numeric_columns(data):
    data = data.copy()
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def check_missing_values(data):
    return data.isna().sum()


def resample_hourly(data):
    return data.resample("1h").mean()


def interpolate_missing(data):
    return data.interpolate(method="time")


def prepare_dataset(file_path):
    data = load_raw_data(file_path)
    data = prepare_datetime(data)
    data = convert_numeric_columns(data)
    data = data.dropna(subset=[TARGET])
    hourly = resample_hourly(data)
    hourly = interpolate_missing(hourly)
    return hourly.dropna()
