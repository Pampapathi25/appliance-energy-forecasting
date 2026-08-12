# Appliance Energy Forecasting

## Project Overview

This project presents an end-to-end time-series forecasting analysis of **household appliance energy consumption** using the **UCI Appliances Energy Prediction dataset**.

The aim of the project is to forecast appliance energy consumption for the **next 24 hours** and compare forecasting approaches of increasing complexity. The analysis progresses from simple benchmark models to statistical time-series modelling, feature-based machine learning, and a pretrained time-series foundation model.

The main modelling approaches used are:

1. Benchmark forecasting models
2. SARIMA/SARIMAX
3. XGBoost feature-based regression
4. Chronos-T5-small foundation model

The complete workflow includes:

- Data acquisition and preprocessing
- Conversion from 10-minute to hourly observations
- Exploratory Data Analysis (EDA)
- Seasonal analysis
- Stationarity testing
- ACF and PACF analysis
- Benchmark forecasting
- SARIMA/SARIMAX parameter selection
- Residual diagnostics
- Feature engineering
- XGBoost modelling
- Chronos foundation-model forecasting
- 24-hour forecasting
- Model evaluation and comparison
- Forecast-origin and data-leakage analysis
- Practical model recommendation

---

# Dataset

The project uses the **Appliances Energy Prediction dataset** from the UCI Machine Learning Repository.

The dataset was introduced by Candanedo, Feldheim and Deramaix (2017) and contains measurements from a low-energy residential building.

The original dataset is sampled every **10 minutes**.

It contains appliance energy consumption together with environmental and sensor measurements such as:

- Appliance energy consumption
- Indoor temperature
- Indoor relative humidity
- Outdoor temperature
- Outdoor humidity
- Atmospheric pressure
- Wind speed
- Visibility
- Dew point
- Date and time

The main target variable used in this project is:

```text
Appliances
```

The original 10-minute observations are resampled into **hourly observations** before modelling.

---

# Forecasting Problem

The forecasting problem is defined as follows:

| Component | Definition |
|---|---|
| Target variable | Appliance energy consumption |
| Original frequency | 10 minutes |
| Modelling frequency | Hourly |
| Forecast horizon | 24 hours |
| Feature-model test period | Final 14 days |
| Main evaluation metrics | MAE, RMSE, MASE and Bias |
| Daily seasonal period | 24 hours |
| Weekly seasonal period | 168 hours |

The main objective is to predict appliance energy consumption for the following **24 hours**.

The models are compared not only according to predictive accuracy but also according to:

- Interpretability
- Forecast uncertainty
- Computational requirements
- Data availability
- Risk of data leakage
- Ease of practical deployment

---

# Project Workflow

## 1. Data Acquisition and Preparation

The first stage of the project prepares the UCI Appliances Energy Prediction dataset for time-series analysis.

The preprocessing workflow includes:

- Retrieving the original dataset
- Parsing the date/time column
- Checking the dataset structure
- Checking missing values
- Converting the timestamp to a time-series index
- Resampling 10-minute measurements to hourly observations
- Saving the processed hourly dataset

The processed data are stored inside:

```text
data/processed/
```

This hourly dataset is subsequently used by the forecasting models.

---

# 2. Exploratory Data Analysis

Exploratory Data Analysis is performed before model development to understand the temporal characteristics of appliance energy consumption.

The EDA includes:

- Full appliance-energy time-series plot
- Hourly consumption profile
- Seasonal decomposition
- Autocorrelation Function (ACF)
- Partial Autocorrelation Function (PACF)
- Distribution and variability analysis
- Investigation of daily patterns
- Investigation of weekly patterns

The analysis shows that appliance energy consumption contains important temporal structure rather than behaving as independent observations.

Daily patterns are visible because household appliance usage changes considerably according to the hour of the day.

Weekly recurrence is also important, as demonstrated by the strong performance of the weekly seasonal benchmark.

---

# 3. Stationarity Analysis

Stationarity is important for autoregressive time-series modelling.

The project assesses stationarity using:

- Augmented Dickey-Fuller (ADF) test
- KPSS test
- Autocorrelation Function
- Partial Autocorrelation Function
- Differencing assessment

The ADF and KPSS tests are interpreted together to determine whether differencing is necessary.

ACF and PACF plots are additionally used to understand serial dependence and support the specification of the autoregressive model.

---

# 4. Benchmark Forecasting Models

Several simple forecasting approaches are implemented to establish baseline performance.

The benchmark models include:

### Mean Forecast

Predicts future observations using the historical average of the training series.

### Naive Forecast

Uses the most recent available observation as the forecast for future observations.

### Daily Seasonal Naive

Uses appliance consumption from the same hour on the previous day.

The seasonal period is:

```text
24 hours
```

### Weekly Seasonal Naive

Uses appliance consumption from the corresponding hour of the previous week.

The seasonal period is:

```text
168 hours
```

### Drift Forecast

Extends the average historical trend between the first and final observations of the training period.

---

## Strongest Benchmark

Among the relevant benchmark models, the **weekly seasonal naive model** provides the strongest performance.

Its final reported performance is:

```text
MASE = 0.798
```

This result indicates that appliance energy consumption has a meaningful **weekly seasonal structure**.

The poor performance of the simple naive and drift approaches also demonstrates that forecasting the next day from a single observation at the train/test boundary can be unreliable when that observation is atypical.

---

# 5. SARIMA and SARIMAX Modelling

The next stage applies seasonal autoregressive modelling.

SARIMA/SARIMAX is appropriate because appliance energy consumption demonstrates:

- Serial correlation
- Daily seasonal behaviour
- Repeated temporal patterns

The project investigates both:

1. Target-only SARIMA
2. SARIMAX using exogenous variables

---

## SARIMA Parameter Search

The assignment requires an exhaustive search over:

```text
p = 0 to 6
d = 0 to 2
q = 0 to 6
```

Therefore, the total number of non-seasonal parameter combinations evaluated is:

```text
7 × 3 × 7 = 147
```

Candidate models are compared using information criteria such as:

- AIC
- BIC

The purpose of the search is to identify a model that provides a strong fit while penalising unnecessary complexity.

---

## SARIMA/SARIMAX Diagnostics

After model fitting, residual diagnostics are performed.

These include:

- Residual time-series analysis
- Residual ACF
- Ljung-Box test
- Residual distribution inspection

The Ljung-Box test and residual correlogram indicate that the fitted model captures much of the important autocorrelation structure.

The model also produces:

- 24-hour point forecasts
- Forecast confidence intervals

---

## SARIMA vs SARIMAX Results

The final comparison gives:

```text
Target-only SARIMA MASE = 0.665

SARIMAX with exogenous variables MASE = 0.687
```

The target-only SARIMA model performs better than the SARIMAX model containing the selected exogenous variables.

The target-only model also performs better according to the model-selection criteria considered in the analysis.

This suggests that adding the selected weather/exogenous variables did not provide sufficient additional predictive information to improve the forecast.

---

# 6. Feature Engineering

The feature-based machine-learning stage adds information that may help explain nonlinear appliance-demand behaviour.

The engineered features include:

### Appliance Lag Features

Examples include:

```text
lag_1
lag_24
lag_168
```

These represent:

- Recent consumption
- Consumption at the same time on the previous day
- Consumption at the same time during the previous week

---

## Rolling Features

Rolling-window statistics are generated from historical appliance consumption.

These include features such as:

- Rolling mean
- Rolling standard deviation
- Short-term demand averages

Rolling features help represent recent demand levels and variability.

---

## Time Features

Calendar variables are extracted from the timestamp.

These include:

- Hour of day
- Day of week
- Weekend information

Cyclical representations of time are also used.

For example:

```text
sin(hour)
cos(hour)
```

Cyclical encoding allows the model to understand that hour 23 and hour 0 are temporally close.

---

## Sensor Features

Indoor environmental measurements are included, such as:

- Temperature
- Relative humidity

Measurements from different rooms provide information about household conditions and possible occupancy/activity patterns.

---

## Weather Features

External weather variables are also considered, including:

- Outdoor temperature
- Outdoor humidity
- Pressure
- Wind speed
- Visibility
- Dew point

---

# 7. XGBoost Feature-Based Model

XGBoost is used as the main feature-based machine-learning model.

XGBoost is suitable because it can model:

- Nonlinear relationships
- Feature interactions
- Lagged demand effects
- Sensor effects
- Weather relationships
- Calendar patterns

The final 14 days are used as the test period for the feature-based modelling stage.

The next 24 hours are evaluated as the main forecasting horizon.

---

## XGBoost Results

The final XGBoost result is:

```text
MASE = 0.597
```

This is the **best MASE obtained among the compared models**.

---

## Important XGBoost Features

Feature importance indicates that the most useful predictors are primarily:

1. Recent lag values
2. Time-of-day features
3. Cyclical hour features
4. Short rolling-window statistics
5. Longer seasonal lag features
6. Sensor/weather information

In particular:

```text
lag_1
```

is highly influential.

This demonstrates that the immediately preceding appliance consumption contains substantial information about near-future consumption.

Time-of-day features are also important, supporting the daily seasonal behaviour identified during EDA.

The importance of lag and rolling variables shows that both recent demand and local demand trends are useful for forecasting.

---

# 8. Chronos Foundation Model

The final modelling stage evaluates a pretrained time-series foundation model.

The project uses:

```text
amazon/chronos-t5-small
```

Chronos is a pretrained model designed to perform time-series forecasting across different datasets.

Unlike XGBoost, Chronos can perform **zero-shot forecasting** without fitting a household-specific supervised model.

---

## Chronos Configuration

The Chronos configuration used in this project is:

```text
Model: amazon/chronos-t5-small

Forecast horizon: 24 hours

Context length: 168 hours

Number of forecast samples: 100
```

A context length of 168 hours represents one week of historical hourly appliance consumption.

The model generates multiple forecast samples to represent predictive uncertainty.

---

## Chronos Results

The final Chronos result is:

```text
MASE = 0.799
```

Chronos therefore performs slightly worse than the strongest weekly seasonal benchmark:

```text
Weekly Seasonal Naive = 0.798
Chronos = 0.799
```

It also performs worse than:

```text
Target-only SARIMA = 0.665
XGBoost = 0.597
```

Chronos therefore does not provide an accuracy improvement sufficient to justify its additional complexity for this particular forecasting task.

However, its ability to perform **zero-shot forecasting** remains useful, particularly when considering forecasting for new households where little household-specific historical training data are available.

---

# Final Model Comparison

The final model comparison according to MASE is:

| Model | MASE | Interpretation |
|---|---:|---|
| XGBoost | **0.597** | Best overall predictive accuracy |
| Target-only SARIMA | **0.665** | Strong accuracy and interpretability |
| SARIMAX with exogenous variables | **0.687** | Exogenous variables did not improve performance |
| Weekly Seasonal Naive | **0.798** | Strongest relevant benchmark |
| Chronos-T5-small | **0.799** | Zero-shot foundation model |

Lower MASE indicates better forecasting performance.

Therefore, the final ranking is:

```text
1. XGBoost                  MASE = 0.597
2. Target-only SARIMA       MASE = 0.665
3. SARIMAX with exogenous   MASE = 0.687
4. Weekly Seasonal Naive    MASE = 0.798
5. Chronos-T5-small         MASE = 0.799
```

---

# Forecast-Origin and Data-Leakage Considerations

An important part of this project is determining which predictors would genuinely be available when a real forecast is produced.

Variables genuinely known at forecast origin include:

- Historical appliance consumption
- Historical lag values
- Rolling statistics calculated from historical observations
- Hour of day
- Day of week
- Other deterministic calendar features

However, realised future values of:

- Indoor temperature
- Indoor humidity
- Outdoor temperature
- Weather measurements

would not necessarily be known when generating a real 24-hour forecast.

If realised future sensor/weather observations from the test dataset are supplied to a model, the resulting forecast should be interpreted as a **conditional forecast**.

It is not a completely realistic ex-ante operational forecast.

A practical deployment would therefore need to:

- Use weather forecasts instead of observed future weather
- Forecast future indoor sensor conditions
- Or restrict the model to predictors genuinely available at forecast origin

This distinction is important when interpreting XGBoost and SARIMAX models that use exogenous variables.

---

# Practical Model Recommendation

No model is best according to every practical criterion.

## XGBoost

Advantages:

- Best predictive accuracy
- MASE = 0.597
- Captures nonlinear relationships
- Effectively uses lag, rolling and temporal features

Limitations:

- Less interpretable than SARIMA
- Does not naturally provide classical confidence intervals
- Current results can depend on future sensor/weather variables
- Requires feature engineering

---

## Target-Only SARIMA

Advantages:

- Strong predictive performance
- MASE = 0.665
- More interpretable
- Uses historical target information
- Provides forecast confidence intervals
- Easier to explain and deploy
- Does not require realised future environmental measurements

Limitations:

- Assumes a relatively structured statistical relationship
- May struggle with nonlinear demand behaviour
- Parameter searching can be computationally expensive

---

## Chronos

Advantages:

- Zero-shot forecasting
- No household-specific supervised training required
- Produces probabilistic forecasts
- Potentially useful for new forecasting environments

Limitations:

- MASE = 0.799
- Does not outperform the specialised models in this experiment
- Greater computational/dependency requirements
- Less interpretable

---

## Recommended Deployment Model

Although XGBoost provides the highest accuracy, the **target-only SARIMA model is recommended for the initial practical smart-home deployment**.

It provides a strong balance between:

- Forecast accuracy
- Interpretability
- Forecast uncertainty
- Data availability
- Computational requirements
- Ease of deployment

XGBoost would be a strong future upgrade if reliable forecasts of the required weather and environmental covariates are available.

Chronos remains useful as a zero-shot forecasting alternative, particularly when applying the system to a new household with limited household-specific historical training data.

---

# Repository Structure

```text
appliance-energy-forecasting/
│
├── data/
│   ├── raw/
│   │   └── energydata_complete.csv
│   │
│   └── processed/
│       └── appliance_hourly.csv
│
├── outputs/
│   │
│   ├── figures/
│   │   ├── EDA figures
│   │   ├── stationarity figures
│   │   ├── SARIMA/SARIMAX forecast figures
│   │   ├── residual diagnostic figures
│   │   ├── XGBoost figures
│   │   └── Chronos forecast figures
│   │
│   ├── forecasts/
│   │   ├── SARIMA/SARIMAX forecasts
│   │   ├── XGBoost forecasts
│   │   └── Chronos forecasts
│   │
│   └── metrics/
│       ├── benchmark metrics
│       ├── stationarity results
│       ├── SARIMA/SARIMAX parameter-search results
│       ├── residual diagnostics
│       ├── XGBoost metrics
│       └── Chronos metrics
│
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── run_eda.py
│   ├── run_benchmarks.py
│   ├── run_sarimax.py
│   ├── run_feature_model.py
│   ├── run_foundation_model.py
│   └── run_pipeline.py
│
├── src/
│   └── appliance_energy/
│       ├── config.py
│       ├── data.py
│       └── models/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Pampapathi25/appliance-energy-forecasting.git
```

Enter the project directory:

```bash
cd appliance-energy-forecasting
```

---

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

---

# 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

For XGBoost on macOS, OpenMP may also be required:

```bash
brew install libomp
```

---

# Running the Project

All commands should be executed from the project root directory.

## Step 1 — Download Data

```bash
python scripts/download_data.py
```

## Step 2 — Prepare Hourly Data

```bash
python scripts/prepare_data.py
```

## Step 3 — Run EDA and Stationarity Analysis

```bash
python scripts/run_eda.py
```

## Step 4 — Run Benchmark Models

```bash
python scripts/run_benchmarks.py
```

## Step 5 — Run SARIMA/SARIMAX

```bash
python scripts/run_sarimax.py
```

The exhaustive parameter search evaluates all required:

```text
147 parameter combinations
```

Depending on the computer, this stage may take considerably longer than the other modelling stages.

## Step 6 — Run XGBoost

```bash
python scripts/run_feature_model.py
```

## Step 7 — Run Chronos

```bash
python scripts/run_foundation_model.py
```

## Run the Complete Pipeline

The complete workflow can also be executed using:

```bash
python scripts/run_pipeline.py
```

---

# Project Outputs

Generated results are organised into three main directories.

## Figures

```text
outputs/figures/
```

Contains:

- Time-series plots
- Hourly demand profile
- Seasonal decomposition
- ACF/PACF plots
- SARIMA/SARIMAX forecasts
- Residual diagnostics
- XGBoost results
- Feature importance
- Chronos forecast plots

## Forecasts

```text
outputs/forecasts/
```

Contains the generated 24-hour forecast values from the main models.

## Metrics

```text
outputs/metrics/
```

Contains:

- Benchmark evaluation metrics
- Stationarity-test results
- SARIMA/SARIMAX parameter search
- AIC/BIC information
- Residual diagnostic results
- XGBoost metrics
- Chronos metrics

---

# Main Findings

The main conclusions from the project are:

1. Appliance energy consumption contains important **daily and weekly temporal structure**.

2. The **weekly seasonal naive** model is the strongest relevant benchmark, demonstrating the importance of weekly recurrence.

3. **SARIMA** substantially improves on the seasonal benchmark while maintaining good interpretability.

4. Adding the selected exogenous variables to SARIMAX does not improve forecasting accuracy.

5. **XGBoost provides the best overall predictive accuracy**, achieving a MASE of 0.597.

6. Recent appliance-use lags, especially `lag_1`, are the most useful XGBoost predictors.

7. Time-of-day and rolling-window features also provide important predictive information.

8. **Chronos-T5-small successfully demonstrates zero-shot foundation-model forecasting**, but it does not outperform the specialised models in this experiment.

9. Increasing model complexity does not automatically produce better forecasting accuracy.

10. Future sensor/weather availability must be considered carefully when evaluating whether a forecast is genuinely deployable.

---

# Reproducibility

The repository is designed so that the complete analysis can be reproduced from the source code.

The generated:

- Metrics
- Forecasts
- Figures
- Model diagnostics

are stored under the `outputs/` directory.

To reproduce the analysis:

1. Clone the repository
2. Create and activate the virtual environment
3. Install `requirements.txt`
4. Run the scripts in sequence
5. Inspect the generated files under `outputs/`

Chronological train/test splitting should always be preserved because randomly shuffling time-series observations would introduce data leakage.

---

# References

Ansari, A.F. *et al.* (2024) ‘Chronos: Learning the language of time series’, *Transactions on Machine Learning Research*.

Box, G.E.P., Jenkins, G.M., Reinsel, G.C. and Ljung, G.M. (2015) *Time Series Analysis: Forecasting and Control*. 5th edn. Hoboken, NJ: Wiley.

Candanedo, L.M., Feldheim, V. and Deramaix, D. (2017) ‘Data driven prediction models of energy use of appliances in a low-energy house’, *Energy and Buildings*, 140, pp. 81–97.

Chen, T. and Guestrin, C. (2016) ‘XGBoost: A scalable tree boosting system’, in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. New York: ACM, pp. 785–794.

Hyndman, R.J. and Athanasopoulos, G. (2021) *Forecasting: Principles and Practice*. 3rd edn. Melbourne: OTexts.

UCI Machine Learning Repository (2017) *Appliances Energy Prediction*.
