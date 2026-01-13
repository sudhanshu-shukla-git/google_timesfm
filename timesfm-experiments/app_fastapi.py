from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import timesfm
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

# Initialize FastAPI app
app = FastAPI(title="TimesFM Financial Forecasting API")

# Initialize TimesFM model (do this once at startup)
torch.set_float32_matmul_precision("high")

# Option 1: Load from local path
LOCAL_MODEL_PATH = "./timesfm_model"  # Change this to your local model path
try:
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(LOCAL_MODEL_PATH)
    print(f"✓ Model loaded from local path: {LOCAL_MODEL_PATH}")
except Exception as e:
    print(f"Failed to load from local path: {e}")
    print("Attempting to load from HuggingFace...")
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

# Request model
class ForecastRequest(BaseModel):
    csv_path: Optional[str] = Field(
        None,
        description="Path to CSV file to load data from"
    )
    target_column: str = Field(
        ...,
        description="Column name to forecast (e.g., 'Actual_Spend')",
        example="Actual_Spend"
    )
    date_column: str = Field(
        default="Date",
        description="Column name containing date information"
    )
    horizon: int = Field(
        default=6,
        ge=1,
        le=256,
        description="Number of periods to forecast into the future"
    )
    aggregation: Optional[str] = Field(
        default="sum",
        description="Aggregation method: 'sum', 'mean', 'median', 'count'"
    )
    time_period: Optional[str] = Field(
        default="month",
        description="Time period for aggregation: 'day', 'week', 'month', 'year'"
    )

# Response model
class ForecastResponse(BaseModel):
    success: bool
    data: List[Dict[str, Any]]
    forecast_data: Dict[str, Any]
    line_data: Dict[str, Any]
    image_base64: str
    summary: Dict[str, Any]
    target_column: str
    timestamp: str

def load_and_prepare_data(csv_path: str, target_column: str, date_column: str, 
                         aggregation: str, time_period: str):
    """Load CSV and prepare time series data"""
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column], format='%d-%m-%Y', errors='coerce')
    
    # Remove rows with invalid dates or missing target values
    df = df.dropna(subset=[date_column, target_column])
    
    # Sort by date
    df = df.sort_values(date_column)
    
    # Create period column based on time_period
    if time_period == "day":
        df['period'] = df[date_column].dt.strftime('%d-%b-%Y')
        period_format = "day"
    elif time_period == "week":
        df['period'] = df[date_column].dt.strftime('Week %U, %Y')
        period_format = "week"
    elif time_period == "month":
        df['period'] = df[date_column].dt.strftime('%B %Y')
        period_format = "month"
    elif time_period == "year":
        df['period'] = df[date_column].dt.year.astype(str)
        period_format = "year"
    else:
        df['period'] = df[date_column].dt.strftime('%B %Y')
        period_format = "month"
    
    # Group by period and aggregate
    if aggregation == "sum":
        grouped = df.groupby('period')[target_column].sum()
    elif aggregation == "mean":
        grouped = df.groupby('period')[target_column].mean()
    elif aggregation == "median":
        grouped = df.groupby('period')[target_column].median()
    elif aggregation == "count":
        grouped = df.groupby('period')[target_column].count()
    else:
        grouped = df.groupby('period')[target_column].sum()
    
    periods = grouped.index.tolist()
    values = grouped.values.tolist()
    
    return periods, values, period_format

def generate_forecast_labels(last_period: str, horizon: int, period_format: str):
    """Generate forecast period labels based on the last historical period"""
    forecast_labels = []
    
    if period_format == "month":
        # Extract month and year from last_period (e.g., "January 2022")
        try:
            last_date = pd.to_datetime(last_period, format='%B %Y')
            for i in range(1, horizon + 1):
                next_date = last_date + pd.DateOffset(months=i)
                forecast_labels.append(next_date.strftime('%B %Y'))
        except:
            # Fallback to generic labels
            forecast_labels = [f"Forecast {i+1}" for i in range(horizon)]
    elif period_format == "day":
        try:
            last_date = pd.to_datetime(last_period, format='%d-%b-%Y')
            for i in range(1, horizon + 1):
                next_date = last_date + pd.Timedelta(days=i)
                forecast_labels.append(next_date.strftime('%d-%b-%Y'))
        except:
            forecast_labels = [f"Forecast {i+1}" for i in range(horizon)]
    elif period_format == "year":
        try:
            last_year = int(last_period)
            forecast_labels = [str(last_year + i) for i in range(1, horizon + 1)]
        except:
            forecast_labels = [f"Forecast {i+1}" for i in range(horizon)]
    else:
        forecast_labels = [f"Forecast {i+1}" for i in range(horizon)]
    
    return forecast_labels

def generate_forecast_plot(historical_data, forecast_values, quantile_forecast, 
                          historical_labels, forecast_labels, target_column):
    """Generate a matplotlib plot and return as base64 encoded image"""
    plt.figure(figsize=(15, 7))
    
    # Create continuous x-axis positions for both historical and forecast
    total_points = len(historical_data) + len(forecast_values)
    all_x = list(range(total_points))
    
    # Split x positions
    hist_x = all_x[:len(historical_data)]
    forecast_x = all_x[len(historical_data)-1:]  # Start from last historical point
    
    # Combine labels
    all_labels = historical_labels + forecast_labels
    
    # Plot historical data
    plt.plot(hist_x, historical_data, 
             label="Historical Data", color="#3498db", linewidth=2, marker='o', markersize=4)
    
    # Prepare forecast with connection point
    forecast_with_connection = [historical_data[-1]] + list(forecast_values)
    
    # Plot forecast - starting from last historical point
    plt.plot(forecast_x, forecast_with_connection, 
             label="Forecast", color="#2ecc71", linewidth=2, marker='s', markersize=4)
    
    # Add confidence intervals
    quantiles_with_connection = np.vstack([[historical_data[-1]] * quantile_forecast.shape[1], quantile_forecast])
    
    plt.fill_between(
        forecast_x,
        quantiles_with_connection[:, 1],  # 10th percentile
        quantiles_with_connection[:, 9],  # 90th percentile
        alpha=0.2,
        color='#2ecc71',
        label='80% Prediction Interval'
    )
    
    plt.fill_between(
        forecast_x,
        quantiles_with_connection[:, 3],  # 25th percentile
        quantiles_with_connection[:, 7],  # 75th percentile
        alpha=0.3,
        color='#2ecc71',
        label='50% Prediction Interval'
    )
    
    # Set x-axis labels
    step = max(1, len(all_labels) // 15)
    tick_positions = list(range(0, len(all_labels), step))
    tick_labels = [all_labels[i] for i in tick_positions]
    
    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
    
    plt.legend(loc='best', fontsize=10)
    plt.title(f"TimesFM Forecast: {target_column}", fontsize=14, fontweight='bold')
    plt.xlabel("Period", fontsize=12)
    plt.ylabel(target_column, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Convert plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return image_base64

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "TimesFM Financial Forecasting API",
        "version": "1.0",
        "model": "TimesFM 2.5 200M PyTorch",
        "endpoints": {
            "/forecast": "POST - Generate financial forecast with graph",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_financial_data(request: ForecastRequest):
    """
    Generate financial forecast from CSV data.
    
    Parameters:
    - csv_path: Path to CSV file
    - target_column: Column name to forecast
    - date_column: Column containing date information
    - horizon: Number of periods to forecast (default: 6)
    - aggregation: How to aggregate data ('sum', 'mean', 'median', 'count')
    - time_period: Time period for aggregation ('day', 'week', 'month', 'year')
    
    Returns:
    - data: Formatted data with last 6 historical + 6 forecast periods
    - forecast_data: Detailed forecast values and quantiles
    - line_data: Data formatted for frontend chart libraries
    - image_base64: Base64 encoded PNG image of the forecast plot
    - summary: Statistical summary and metrics
    """
    try:
        # if not request.csv_path:
        #     raise HTTPException(
        #         status_code=400,
        #         detail="csv_path is required"
        #     )
        
        # Load and prepare data
        periods, values, period_format = load_and_prepare_data(
            request.csv_path or "./datas/data.csv",
            request.target_column,
            request.date_column,
            request.aggregation,
            request.time_period
        )
        
        if len(values) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data points after aggregation. Got {len(values)}, need at least 3"
            )
        
        # Convert to numpy array
        historical_array = np.array(values, dtype=np.float32)
        
        # Generate forecast
        point_forecast, quantile_forecast = model.forecast(
            horizon=request.horizon,
            inputs=[historical_array],
        )
        
        # Extract forecasts
        forecast_values = point_forecast[0]
        quantiles = quantile_forecast[0]  # Shape: [horizon, 11]
        
        # Generate forecast labels
        forecast_labels = generate_forecast_labels(periods[-1], request.horizon, period_format)
        
        # Convert numpy arrays to Python lists with native types
        def convert_to_native(arr):
            """Convert numpy array to list of native Python floats"""
            return [float(x) for x in arr]
        
        # Prepare the data array with last 6 historical + 6 forecast
        data_array = []
        
        # Get last 6 historical periods (or less if not available)
        hist_start_idx = max(0, len(periods) - 6)
        for i in range(hist_start_idx, len(periods)):
            data_array.append({
                "name": periods[i],
                "history": round(float(values[i]), 2),
                "color": "#3498db"
            })
        
        # Add forecast periods (up to 6 or horizon, whichever is smaller)
        forecast_count = min(6, len(forecast_values))
        for i in range(forecast_count):
            data_array.append({
                "name": forecast_labels[i],
                "forecast": round(float(forecast_values[i]), 2),
                "color": "#2ecc71"
            })
        
        # Prepare forecast_data
        forecast_data = {
            "periods": forecast_labels,
            "point_forecast": convert_to_native(forecast_values),
            "quantiles": {
                "q10": convert_to_native(quantiles[:, 1]),
                "q25": convert_to_native(quantiles[:, 3]),
                "q50": convert_to_native(quantiles[:, 5]),
                "q75": convert_to_native(quantiles[:, 7]),
                "q90": convert_to_native(quantiles[:, 9])
            },
            "horizon": int(len(forecast_values)),
            "aggregation": request.aggregation,
            "time_period": request.time_period
        }
        
        # Prepare line_data for chart libraries
        historical_data_with_connection = convert_to_native(historical_array) + [float(historical_array[-1])] + [None] * (len(forecast_values) - 1)
        forecast_data_with_connection = [None] * len(values) + [float(historical_array[-1])] + convert_to_native(forecast_values)
        
        line_data = {
            "labels": periods + forecast_labels,
            "datasets": [
                {
                    "label": f"Historical {request.target_column}",
                    "data": historical_data_with_connection,
                    "borderColor": "#3498db",
                    "backgroundColor": "rgba(52, 152, 219, 0.1)",
                    "borderWidth": 2,
                    "pointRadius": 3,
                    "spanGaps": False
                },
                {
                    "label": "Forecast",
                    "data": forecast_data_with_connection,
                    "borderColor": "#2ecc71",
                    "backgroundColor": "rgba(46, 204, 113, 0.1)",
                    "borderWidth": 2,
                    "pointRadius": 3,
                    "spanGaps": False
                },
                {
                    "label": "80% Prediction Interval (Upper)",
                    "data": [None] * len(values) + [float(historical_array[-1])] + convert_to_native(quantiles[:, 9]),
                    "borderColor": "rgba(46, 204, 113, 0.3)",
                    "backgroundColor": "rgba(46, 204, 113, 0.1)",
                    "borderWidth": 1,
                    "borderDash": [5, 5],
                    "pointRadius": 0,
                    "fill": "+1",
                    "spanGaps": False
                },
                {
                    "label": "80% Prediction Interval (Lower)",
                    "data": [None] * len(values) + [float(historical_array[-1])] + convert_to_native(quantiles[:, 1]),
                    "borderColor": "rgba(46, 204, 113, 0.3)",
                    "backgroundColor": "rgba(46, 204, 113, 0.2)",
                    "borderWidth": 1,
                    "borderDash": [5, 5],
                    "pointRadius": 0,
                    "spanGaps": False
                }
            ],
            "x_axis": {
                "label": "Period",
                "type": "category"
            },
            "y_axis": {
                "label": request.target_column,
                "type": "linear"
            },
            "legend": {
                "display": True,
                "position": "top"
            }
        }
        
        # Calculate summary statistics
        historical_mean = float(np.mean(values))
        historical_std = float(np.std(values))
        forecast_mean = float(np.mean(forecast_values))
        forecast_std = float(np.std(forecast_values))
        
        summary = {
            "historical_stats": {
                "count": len(values),
                "mean": round(historical_mean, 2),
                "std": round(historical_std, 2),
                "min": round(float(np.min(values)), 2),
                "max": round(float(np.max(values)), 2),
                "total": round(float(np.sum(values)), 2),
                "periods": periods[:5] if len(periods) > 5 else periods
            },
            "forecast_stats": {
                "count": len(forecast_values),
                "mean": round(forecast_mean, 2),
                "std": round(forecast_std, 2),
                "min": round(float(np.min(forecast_values)), 2),
                "max": round(float(np.max(forecast_values)), 2),
                "total": round(float(np.sum(forecast_values)), 2)
            },
            "trend": {
                "direction": "increasing" if forecast_mean > historical_mean else "decreasing",
                "change_percent": round(((forecast_mean - historical_mean) / historical_mean) * 100, 2) if historical_mean != 0 else 0
            },
            "confidence_intervals": {
                "80_percent": {
                    "lower": round(float(np.mean(quantiles[:, 1])), 2),
                    "upper": round(float(np.mean(quantiles[:, 9])), 2)
                },
                "50_percent": {
                    "lower": round(float(np.mean(quantiles[:, 3])), 2),
                    "upper": round(float(np.mean(quantiles[:, 7])), 2)
                }
            },
            "data_info": {
                "total_periods": len(values),
                "aggregation_method": request.aggregation,
                "time_period": request.time_period,
                "period_format": period_format
            }
        }
        
        # Generate plot image
        image_base64 = generate_forecast_plot(
            values,
            forecast_values,
            quantiles,
            periods,
            forecast_labels,
            request.target_column
        )
        
        return ForecastResponse(
            success=True,
            data=data_array,
            forecast_data=forecast_data,
            line_data=line_data,
            image_base64=image_base64,
            summary=summary,
            target_column=request.target_column,
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating forecast: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)