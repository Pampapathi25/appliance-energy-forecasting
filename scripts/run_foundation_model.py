"""
Foundation-model stage.

This file is intentionally a separate adapter so the project can use a
supported model such as Chronos, TimesFM or TimeGPT without changing the
rest of the pipeline.

Do not claim a foundation-model result until a real model has been installed,
run and evaluated.
"""

def main():
    print(
        "Foundation-model adapter is not configured yet. "
        "Choose the model supported by your course environment, then implement "
        "the adapter and save a 24-hour forecast to outputs/forecasts/."
    )


if __name__ == "__main__":
    main()
