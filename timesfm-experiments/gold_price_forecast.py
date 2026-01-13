# Reading the Data
import pandas as pd
df = pd.read_csv("datas/1979-2021.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').resample('MS').mean()
df = df.reset_index() # Reset index to have 'Date' as a column again
print(df.head())

# Visualise the Dataset
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
sns.lineplot(x="Date", y='India(INR)', data=df, color='green')
plt.title('Monthly Gold Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Gold Price in INR')
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
df.set_index("Date", inplace=True)
result = seasonal_decompose(df['India(INR)'])
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
result.observed.plot(ax=ax1, color='green')
ax1.set_ylabel('Observed')
result.trend.plot(ax=ax2, color='green')
ax2.set_ylabel('Trend')
result.seasonal.plot(ax=ax3, color='green')
ax3.set_ylabel('Seasonal')
result.resid.plot(ax=ax4, color='green')
ax4.set_ylabel('Residual')
plt.tight_layout()
plt.show()
df.reset_index(inplace=True)

# Arranging the Data in Format as Required by the Models
df = pd.DataFrame({'unique_id':[1]*len(df),'ds': df["Date"], "y":df['India(INR)']})
train_df = df[df['ds'] <= '31-07-2019']
test_df = df[df['ds'] > '31-07-2019']


# 1. Statistical Forecasting

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS

# Define the AutoARIMA model
autoarima = AutoARIMA(season_length=12)
# Define the AutoETS model
autoets = AutoETS(season_length=12)

# Create StatsForecast object with AutoARIMA
statforecast = StatsForecast(
    models=[autoarima, autoets],
    freq='MS',
    n_jobs=-1)
statforecast.fit(train_df)

# Generate forecasts for 24 periods ahead
sf_forecast = statforecast.forecast(df=train_df, h=24, fitted=True)
sf_forecast = sf_forecast.reset_index()
print("StatsForecast:", sf_forecast)


# 2. ML Forecasting

from mlforecast import MLForecast
from mlforecast.target_transforms import AutoDifferences
from numba import njit
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from mlforecast import MLForecast
from mlforecast.lag_transforms import (
 RollingMean, RollingStd, RollingMin, RollingMax, RollingQuantile,
 SeasonalRollingMean, SeasonalRollingStd, SeasonalRollingMin,
 SeasonalRollingMax, SeasonalRollingQuantile,
 ExpandingMean
)

models = [lgb.LGBMRegressor(verbosity=-1),
 xgb.XGBRegressor(),
 RandomForestRegressor(random_state=0),
]
fcst = MLForecast(
    models=models, # List of models to be used for forecasting
    freq='MS', # Monthly frequency, starting at the beginning of each month
    lags=[1,3,5,7,12], # Lag features: values from 1, 3, 5, 7, and 12 time steps ago
    lag_transforms={
        1: [ # Transformations applied to lag 1
            RollingMean(window_size=3), 
            RollingStd(window_size=3), 
            RollingMin(window_size=3),
            RollingMax(window_size=3), 
            RollingQuantile(p=0.5, window_size=3),
            ExpandingMean()
        ],
        6:[ # Transformations applied to lag 6
            RollingMean(window_size=6), 
            RollingStd(window_size=6),
            RollingMin(window_size=6), 
            RollingMax(window_size=6), 
            RollingQuantile(p=0.5, window_size=6), 
        ],
        12: [ # Transformations applied to lag 12 (likely for yearly seasonality)
            SeasonalRollingMean(season_length=12, window_size=3), 
            SeasonalRollingStd(season_length=12, window_size=3), 
            SeasonalRollingMin(season_length=12, window_size=3), 
            SeasonalRollingMax(season_length=12, window_size=3), 
            SeasonalRollingQuantile(p=0.5, season_length=12, window_size=3) 
        ]
    },
    date_features=['year', 'month', 'quarter'], # Extract year, month, and quarter from the date as features
    target_transforms=[AutoDifferences(max_diffs=3)]
)
 
fcst.fit(train_df)
ml_forecast = fcst.predict(len(test_df))
print("MLForecast:", ml_forecast)


# 3. TimeGPT Zero-shot Forecasting

from nixtla import NixtlaClient
nixtla_client = NixtlaClient(api_key = 'nixak-PJPGa3MxJ3VdxZhKvylOcu2XHBtZ8ssIykc7wzoLKB0sVcDMnHoD53kGpvuJGk9e5lj83KojwKaljmcK')
timegpt_forecast = nixtla_client.forecast(df=train_df, h=24, freq='MS')
print("TimeGPT: ", timegpt_forecast)


# 4. TimesFM Forecasting

import torch
import numpy as np
import pandas as pd
import timesfm

torch.set_float32_matmul_precision("high")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,                # Maximum context length
        max_horizon=256,                 # Maximum forecast horizon
        normalize_inputs=True,           # Normalize time series before forecasting
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
H = 24  # Forecast horizon

series_list = []
for _, g in train_df.groupby("unique_id"):
    series_list.append(g["y"].values.astype(np.float32))

point_forecast, quantile_forecast = model.forecast(
    horizon=H,
    inputs=series_list,
)

forecasts = []
for (uid, group), preds in zip(train_df.groupby("unique_id"), point_forecast):
    # Get last date and extend future timestamps
    last_date = group["ds"].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=H + 1, freq="MS")[1:]
    df_pred = pd.DataFrame({
        "unique_id": uid,
        "ds": future_dates,
        "timesfm": preds
    })
    forecasts.append(df_pred)

timesfm_forecast = pd.concat(forecasts, ignore_index=True)
print("TimesEM: ", timesfm_forecast)


# Assuming the DataFrames have a common column 'ds' for the dates
# Convert 'ds' to datetime in all DataFrames if necessary
sf_forecast['ds'] = pd.to_datetime(sf_forecast['ds'])
ml_forecast['ds'] = pd.to_datetime(ml_forecast['ds'])
timegpt_forecast['ds'] = pd.to_datetime(timegpt_forecast['ds'])
timesfm_forecast['ds'] = pd.to_datetime(timesfm_forecast['ds'])
test_df['ds'] = pd.to_datetime(test_df['ds'])

# Print shapes to debug
print("sf_forecast shape:", sf_forecast.shape)
print("ml_forecast shape:", ml_forecast.shape)
print("timegpt_forecast shape:", timegpt_forecast.shape)
print("timesfm_forecast shape:", timesfm_forecast.shape)
print("test_df shape:", test_df.shape)

# Check the first few dates
print("\nFirst dates:")
print("sf_forecast:", sf_forecast['ds'].head(3).tolist())
print("ml_forecast:", ml_forecast['ds'].head(3).tolist())
print("test_df:", test_df['ds'].head(3).tolist())

# Perform the merges - start with test_df to ensure we keep all test dates
merged_fcst = test_df[['ds', 'y', 'unique_id']].copy()
merged_fcst = pd.merge(merged_fcst, sf_forecast[['ds', 'AutoARIMA', 'AutoETS']], on='ds', how='left')
merged_fcst = pd.merge(merged_fcst, ml_forecast[['ds', 'LGBMRegressor', 'XGBRegressor', 'RandomForestRegressor']], on='ds', how='left')
merged_fcst = pd.merge(merged_fcst, timegpt_forecast[['ds', 'TimeGPT']], on='ds', how='left')
merged_fcst = pd.merge(merged_fcst, timesfm_forecast[['ds', 'timesfm']], on='ds', how='left')

print("\nMerged forecast shape:", merged_fcst.shape)
print("\nMerged forecast columns:", merged_fcst.columns.tolist())
print("\nFirst few rows of merged_fcst:")
print(merged_fcst.head())
print("\nNull counts:")
print(merged_fcst.isnull().sum())
merged_fcst.to_csv("TimesFM_Forecast_Comparison.csv")


# Evaluation of the Models

import numpy as np

def calculate_error_metrics(actual_values, predicted_values):
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    
    # Remove any NaN values
    mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
    actual_values = actual_values[mask]
    predicted_values = predicted_values[mask]
    
    if len(actual_values) == 0:
        print(f"Warning: No valid data points after removing NaNs")
        return pd.DataFrame({'Metric': ['MAE', 'RMSE', 'MAPE'], 'Value': [np.nan, np.nan, np.nan]})
    
    metrics_dict = {
        'MAE': np.mean(np.abs(actual_values - predicted_values)),
        'RMSE': np.sqrt(np.mean((actual_values - predicted_values)**2)),
        'MAPE': np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
    }
    
    result_df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value'])
    return result_df

# Use actual gold prices from merged dataframe
actuals = merged_fcst['y'].values
error_metrics_dict = {}

# Model columns to evaluate
model_columns = ['AutoARIMA', 'AutoETS', 'LGBMRegressor', 'XGBRegressor', 'RandomForestRegressor', 'TimeGPT', 'timesfm']

for col in model_columns:
    if col in merged_fcst.columns:
        print(f"\nEvaluating {col}...")
        predicted_values = merged_fcst[col].values
        print(f"  Actuals shape: {actuals.shape}, Predictions shape: {predicted_values.shape}")
        print(f"  Non-null predictions: {(~np.isnan(predicted_values)).sum()}")
        error_metrics_dict[col] = calculate_error_metrics(actuals, predicted_values)['Value'].values
    else:
        print(f"\nWarning: {col} not found in merged_fcst")

error_metrics_df = pd.DataFrame(error_metrics_dict)
error_metrics_df.insert(0, 'Metric', ['MAE', 'RMSE', 'MAPE'])

print("\n" + "="*80)
print("FINAL ERROR METRICS:")
print("="*80)
print(error_metrics_df)