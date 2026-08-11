import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from appliance_energy.data import prepare_dataset

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "energydata_complete.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "appliance_hourly.csv"


def main():
    data = prepare_dataset(RAW_FILE)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_FILE)
    print(f"Hourly dataset saved to: {OUTPUT_FILE}")
    print(f"Shape: {data.shape}")


if __name__ == "__main__":
    main()
