import torch
import numpy as np
from datetime import datetime, timedelta
import timesfm
import matplotlib.pyplot as plt

# Initialize TimesFM model
print("Loading TimesFM model...")
torch.set_float32_matmul_precision("high")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

def generate_forecast(historical_data, horizon=90, start_date=None, target_column="Bike Rentals"):

    if len(historical_data) < 30:
        raise ValueError("Historical data must contain at least 30 data points")
    
    if horizon > 256 or horizon < 1:
        raise ValueError("Horizon must be between 1 and 256 days")
    
    historical_array = np.array(historical_data, dtype=np.float32)
    
    print(f"Generating {horizon}-day forecast...")
    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon,
        inputs=[historical_array],
    )
    
    # Extract forecasts
    forecast_values = point_forecast[0]
    quantiles = quantile_forecast[0]  # Shape: [horizon, 11]
    
    # Generate dates
    if start_date:
        if isinstance(start_date, str):
            start = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start = start_date
    else:
        start = datetime.now() - timedelta(days=len(historical_data))
    
    historical_dates = [start + timedelta(days=i) for i in range(len(historical_data))]
    forecast_start = start + timedelta(days=len(historical_data))
    forecast_dates = [forecast_start + timedelta(days=i) for i in range(horizon)]
    
    # Calculate summary statistics
    historical_mean = np.mean(historical_data)
    historical_std = np.std(historical_data)
    forecast_mean = np.mean(forecast_values)
    forecast_std = np.std(forecast_values)
    
    # Print summary
    print("\n" + "="*70)
    print(f"FORECAST SUMMARY: {target_column}")
    print("="*70)
    
    print("\nHistorical Data Statistics:")
    print(f"  Period: {historical_dates[0].strftime('%Y-%m-%d')} to {historical_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Count: {len(historical_data)} days")
    print(f"  Mean: {historical_mean:.2f}")
    print(f"  Std Dev: {historical_std:.2f}")
    print(f"  Min: {np.min(historical_data):.2f}")
    print(f"  Max: {np.max(historical_data):.2f}")
    
    print("\nForecast Statistics:")
    print(f"  Period: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")
    print(f"  Count: {horizon} days")
    print(f"  Mean: {forecast_mean:.2f}")
    print(f"  Std Dev: {forecast_std:.2f}")
    print(f"  Min: {np.min(forecast_values):.2f}")
    print(f"  Max: {np.max(forecast_values):.2f}")
    
    print("\nTrend Analysis:")
    change_percent = ((forecast_mean - historical_mean) / historical_mean) * 100
    trend_direction = "increasing" if change_percent > 0 else "decreasing"
    print(f"  Direction: {trend_direction.upper()}")
    print(f"  Change: {change_percent:+.2f}%")
    
    print("\nConfidence Intervals (Average):")
    print(f"  80% Interval: [{np.mean(quantiles[:, 1]):.2f}, {np.mean(quantiles[:, 9]):.2f}]")
    print(f"  50% Interval: [{np.mean(quantiles[:, 3]):.2f}, {np.mean(quantiles[:, 7]):.2f}]")
    
    print("\nFirst 10 Forecast Values:")
    for i in range(min(10, len(forecast_values))):
        date_str = forecast_dates[i].strftime('%Y-%m-%d')
        val = forecast_values[i]
        q10 = quantiles[i, 1]
        q90 = quantiles[i, 9]
        print(f"  {date_str}: {val:.2f} (80% CI: [{q10:.2f}, {q90:.2f}])")
    
    if len(forecast_values) > 10:
        print(f"  ... and {len(forecast_values) - 10} more days")
    
    print("="*70 + "\n")
    
    # Generate plot for historical and forecasted data
    plt.figure(figsize=(15, 7))
    plt.plot(historical_dates, historical_data, 
             label="Historical Data", color="black", linewidth=2, marker='o', 
             markersize=3, markevery=max(1, len(historical_data)//50))
    plt.plot(forecast_dates, forecast_values, 
             label="Forecast", color="red", linewidth=2, marker='s',
             markersize=3, markevery=max(1, horizon//50))
    
    # Add confidence intervals
    plt.fill_between(
        forecast_dates,
        quantiles[:, 1],  # 10th percentile
        quantiles[:, 9],  # 90th percentile
        alpha=0.2,
        color='red',
        label='80% Prediction Interval'
    )
    
    plt.fill_between(
        forecast_dates,
        quantiles[:, 3],  # 25th percentile
        quantiles[:, 7],  # 75th percentile
        alpha=0.3,
        color='red',
        label='50% Prediction Interval'
    )
    
    plt.legend(loc='best', fontsize=11)
    plt.title(f"TimesFM Forecast: {target_column}", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=13)
    plt.ylabel(target_column, fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Return detailed results
    return {
        "forecast_values": forecast_values.tolist(),
        "forecast_dates": [d.strftime('%Y-%m-%d') for d in forecast_dates],
        "quantiles": {
            "q10": quantiles[:, 1].tolist(),
            "q25": quantiles[:, 3].tolist(),
            "q50": quantiles[:, 5].tolist(),
            "q75": quantiles[:, 7].tolist(),
            "q90": quantiles[:, 9].tolist()
        },
        "historical_mean": float(historical_mean),
        "forecast_mean": float(forecast_mean),
        "trend": trend_direction,
        "change_percent": float(change_percent)
    }

if __name__ == "__main__":
    # Create synthetic data with trend and seasonality
    np.random.seed(42)
    days = 365
    trend = np.linspace(150, 300, days)
    seasonality = 50 * np.sin(np.linspace(0, 4*np.pi, days))
    noise = np.random.normal(0, 20, days)
    historical_data = trend + seasonality + noise
    historical_data = np.maximum(historical_data, 0)  # Ensure non-negative
    
    # Generate forecast
    results = generate_forecast(
        historical_data=historical_data.tolist(),
        horizon=90,
        start_date="2023-01-01",
        target_column="Bike Rentals"
    )
    print(results)