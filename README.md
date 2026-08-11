# Appliance Energy Forecasting

University of Hertfordshire Time Series Case Study project.

## Workflow
1. Download and prepare the UCI Appliance Energy Prediction dataset.
2. Resample 10-minute observations to hourly observations.
3. Perform EDA, decomposition, ACF/PACF and stationarity tests.
4. Evaluate mean, naive, daily seasonal naive, weekly seasonal naive and drift benchmarks.
5. Search SARIMAX parameters using AIC.
6. Diagnose SARIMAX residuals and produce confidence intervals.
7. Engineer time, lag and rolling features.
8. Train an XGBoost feature-based model.
9. Run a time-series foundation model.
10. Compare all models using MAE, RMSE, MASE and Bias.

## Project structure
See the directory structure in the repository. Run scripts from the project root with the virtual environment activated.
