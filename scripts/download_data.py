from pathlib import Path
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RAW_DIR / "energydata_complete.csv"
    print("Downloading Appliance Energy Prediction dataset...")
    data = pd.read_csv(URL)
    data.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    print(f"Shape: {data.shape}")


if __name__ == "__main__":
    main()
