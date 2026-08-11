import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ["scripts/download_data.py"],
    ["scripts/prepare_data.py"],
    ["scripts/run_eda.py"],
    ["scripts/run_benchmarks.py"],
    ["scripts/run_sarimax.py"],
    ["scripts/run_feature_model.py"],
    ["scripts/run_foundation_model.py"],
]


def main():
    for step in STEPS:
        print(f"\n===== Running {step[0]} =====")
        result = subprocess.run(
            [sys.executable, *step],
            cwd=ROOT,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
