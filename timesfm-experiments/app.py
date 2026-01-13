import streamlit as st
import pandas as pd
import numpy as np
import torch
import timesfm
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="TimesFM Forecasting Tool",
    page_icon="📈",
    layout="wide"
)

# Initialize TimesFM model
@st.cache_resource
def load_model():
    torch.set_float32_matmul_precision("high")
    LOCAL_MODEL_PATH = "./timesfm_model"
    
    try:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(LOCAL_MODEL_PATH)
        st.success(f"✓ Model loaded from local path: {LOCAL_MODEL_PATH}")
    except Exception as e:
        st.warning(f"Failed to load from local path, trying HuggingFace...")
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        st.success("✓ Model loaded from HuggingFace")
    
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
    return model

def load_and_prepare_data(df, target_column, date_column, aggregation, time_period):
    """Prepare time series data from dataframe"""
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Convert date column to datetime with multiple format attempts
    date_formats = [
        '%d-%m-%Y',  # 01-01-1985
        '%m-%d-%Y',  # 01-01-1985 (alternative interpretation)
        '%Y-%m-%d',  # 1985-01-01
        '%m/%d/%Y %H:%M',  # 2/24/2003 0:00
        '%d/%m/%Y %H:%M',  # 24/2/2003 0:00
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%d-%m-%Y %H:%M:%S',
        '%d-%m-%Y %H:%M',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y/%m/%d',
        '%d.%m.%Y',  # 01.01.1985
        '%Y.%m.%d',  # 1985.01.01
    ]
    
    parsed_successfully = False
    successful_format = None
    
    for fmt in date_formats:
        try:
            test_parse = pd.to_datetime(df[date_column], format=fmt, errors='coerce')
            # Check if at least some dates were parsed
            if test_parse.notna().sum() > 0:
                df[date_column] = test_parse
                parsed_successfully = True
                successful_format = fmt
                break
        except:
            continue
    
    # If no format worked, try automatic parsing
    if not parsed_successfully:
        try:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce', infer_datetime_format=True)
            if df[date_column].notna().sum() > 0:
                parsed_successfully = True
                successful_format = "auto-detected"
        except:
            pass
    
    if not parsed_successfully:
        # Show first few non-null values to help debug
        sample_values = df[date_column].dropna().head(5).tolist()
        raise ValueError(f"Could not parse dates in column '{date_column}'. Sample values: {sample_values}")
    
    # Remove rows with invalid dates or missing target values
    initial_count = len(df)
    df = df.dropna(subset=[date_column, target_column])
    removed_count = initial_count - len(df)
    
    if len(df) == 0:
        raise ValueError(f"No valid data after removing {removed_count} rows with invalid dates or missing target values.")
    
    # Sort by date
    df = df.sort_values(date_column)
    
    # Create period column based on time_period
    if time_period == "Day":
        df['period'] = df[date_column].dt.strftime('%d-%b-%Y')
        period_format = "day"
    elif time_period == "Week":
        df['period'] = df[date_column].dt.strftime('Week %U, %Y')
        period_format = "week"
    elif time_period == "Month":
        df['period'] = df[date_column].dt.strftime('%B %Y')
        period_format = "month"
    elif time_period == "Year":
        df['period'] = df[date_column].dt.year.astype(str)
        period_format = "year"
    else:
        df['period'] = df[date_column].dt.strftime('%B %Y')
        period_format = "month"
    
    # Group by period and aggregate
    if aggregation == "Sum":
        grouped = df.groupby('period')[target_column].sum()
    elif aggregation == "Mean":
        grouped = df.groupby('period')[target_column].mean()
    elif aggregation == "Median":
        grouped = df.groupby('period')[target_column].median()
    elif aggregation == "Count":
        grouped = df.groupby('period')[target_column].count()
    else:
        grouped = df.groupby('period')[target_column].sum()
    
    periods = grouped.index.tolist()
    values = grouped.values.tolist()
    
    return periods, values, period_format

def generate_forecast_labels(last_period, horizon, period_format):
    """Generate forecast period labels"""
    forecast_labels = []
    
    if period_format == "month":
        try:
            last_date = pd.to_datetime(last_period, format='%B %Y')
            for i in range(1, horizon + 1):
                next_date = last_date + pd.DateOffset(months=i)
                forecast_labels.append(next_date.strftime('%B %Y'))
        except:
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

def generate_chat_response(query, forecast_data, summary):
    """Generate responses to user queries about the forecast"""
    query_lower = query.lower()
    
    if "trend" in query_lower or "direction" in query_lower:
        direction = summary['trend']['direction']
        change = summary['trend']['change_percent']
        return f"The forecast shows a **{direction}** trend with a {abs(change):.2f}% change compared to historical data."
    
    elif "highest" in query_lower or "maximum" in query_lower or "peak" in query_lower:
        max_val = summary['forecast_stats']['max']
        max_idx = forecast_data['point_forecast'].index(max_val)
        max_period = forecast_data['periods'][max_idx]
        return f"The highest forecasted value is **{max_val:,.2f}** in **{max_period}**."
    
    elif "lowest" in query_lower or "minimum" in query_lower:
        min_val = summary['forecast_stats']['min']
        min_idx = forecast_data['point_forecast'].index(min_val)
        min_period = forecast_data['periods'][min_idx]
        return f"The lowest forecasted value is **{min_val:,.2f}** in **{min_period}**."
    
    elif "average" in query_lower or "mean" in query_lower:
        avg = summary['forecast_stats']['mean']
        return f"The average forecasted value is **{avg:,.2f}**."
    
    elif "total" in query_lower or "sum" in query_lower:
        total = summary['forecast_stats']['total']
        return f"The total forecasted value across all periods is **{total:,.2f}**."
    
    elif "confidence" in query_lower or "interval" in query_lower:
        ci_80 = summary['confidence_intervals']['80_percent']
        return f"The 80% confidence interval ranges from **{ci_80['lower']:,.2f}** to **{ci_80['upper']:,.2f}**."
    
    elif "compare" in query_lower or "historical" in query_lower:
        hist_mean = summary['historical_stats']['mean']
        fore_mean = summary['forecast_stats']['mean']
        diff = fore_mean - hist_mean
        pct = ((fore_mean - hist_mean) / hist_mean * 100) if hist_mean != 0 else 0
        return f"Historical average: **{hist_mean:,.2f}**\nForecast average: **{fore_mean:,.2f}**\nDifference: **{diff:+,.2f}** ({pct:+.2f}%)"
    
    else:
        return "I can help you with questions about:\n- Trend and direction\n- Highest/lowest values\n- Average and totals\n- Confidence intervals\n- Comparing historical vs forecast data\n\nPlease ask a specific question!"

# Main UI
st.title("📈 TimesFM Forecasting Tool")
st.markdown("Upload your data and generate time series forecasts using Google's TimesFM model")

# Load model
with st.spinner("Loading TimesFM model..."):
    model = load_model()

# Sidebar for controls
st.sidebar.header("⚙️ Forecast Configuration")

# Data input method
input_method = st.sidebar.radio("Data Input Method", ["Upload CSV", "Paste Text Data"])

df = None

if input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])
    if uploaded_file:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        df = None
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip')
                st.sidebar.success(f"✓ Loaded {len(df)} rows (encoding: {encoding})")
                break
            except (UnicodeDecodeError, Exception) as e:
                continue
        
        if df is None:
            st.sidebar.error("❌ Could not read file. Please check the file encoding.")
else:
    text_data = st.sidebar.text_area("Paste CSV Data (with headers)", height=200)
    if text_data:
        from io import StringIO
        df = pd.read_csv(StringIO(text_data))
        st.sidebar.success(f"✓ Loaded {len(df)} rows")

# Main content area
if df is not None:
    # Show data preview
    with st.expander("📊 Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
        st.info(f"Total rows: {len(df)} | Columns: {len(df.columns)}")
    
    # Configuration
    col1, col2 = st.sidebar.columns(2)
    
    # Get numeric and date columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    with col1:
        date_column = st.selectbox("Date Column", all_cols)
    
    with col2:
        target_column = st.selectbox("Target Column", numeric_cols)
    
    col3, col4 = st.sidebar.columns(2)
    
    with col3:
        time_period = st.selectbox("Time Period", ["Day", "Week", "Month", "Year"])
    
    with col4:
        aggregation = st.selectbox("Aggregation", ["Sum", "Mean", "Median", "Count"])
    
    horizon = st.sidebar.slider("Forecast Horizon", min_value=1, max_value=24, value=6)
    
    # Forecast button
    if st.sidebar.button("🚀 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating forecast..."):
            try:
                # Prepare data
                periods, values, period_format = load_and_prepare_data(
                    df.copy(), target_column, date_column, aggregation, time_period
                )
                
                if len(values) < 3:
                    st.error(f"Not enough data points after aggregation. Got {len(values)}, need at least 3")
                else:
                    # Generate forecast
                    historical_array = np.array(values, dtype=np.float32)
                    point_forecast, quantile_forecast = model.forecast(
                        horizon=horizon,
                        inputs=[historical_array],
                    )
                    
                    forecast_values = point_forecast[0]
                    quantiles = quantile_forecast[0]
                    
                    # Generate labels
                    forecast_labels = generate_forecast_labels(periods[-1], horizon, period_format)
                    
                    # Convert to native types
                    forecast_values_list = [float(x) for x in forecast_values]
                    
                    # Store in session state
                    st.session_state['forecast_data'] = {
                        'periods': forecast_labels,
                        'point_forecast': forecast_values_list,
                        'quantiles': {
                            'q10': [float(x) for x in quantiles[:, 1]],
                            'q25': [float(x) for x in quantiles[:, 3]],
                            'q50': [float(x) for x in quantiles[:, 5]],
                            'q75': [float(x) for x in quantiles[:, 7]],
                            'q90': [float(x) for x in quantiles[:, 9]]
                        },
                        'horizon': horizon
                    }
                    
                    st.session_state['summary'] = {
                        'historical_stats': {
                            'count': len(values),
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'total': float(np.sum(values))
                        },
                        'forecast_stats': {
                            'count': len(forecast_values),
                            'mean': float(np.mean(forecast_values)),
                            'std': float(np.std(forecast_values)),
                            'min': float(np.min(forecast_values)),
                            'max': float(np.max(forecast_values)),
                            'total': float(np.sum(forecast_values))
                        },
                        'trend': {
                            'direction': 'increasing' if np.mean(forecast_values) > np.mean(values) else 'decreasing',
                            'change_percent': ((np.mean(forecast_values) - np.mean(values)) / np.mean(values) * 100) if np.mean(values) != 0 else 0
                        },
                        'confidence_intervals': {
                            '80_percent': {
                                'lower': float(np.mean(quantiles[:, 1])),
                                'upper': float(np.mean(quantiles[:, 9]))
                            },
                            '50_percent': {
                                'lower': float(np.mean(quantiles[:, 3])),
                                'upper': float(np.mean(quantiles[:, 7]))
                            }
                        }
                    }
                    
                    st.session_state['periods'] = periods
                    st.session_state['values'] = values
                    st.session_state['forecast_labels'] = forecast_labels
                    st.session_state['quantiles'] = quantiles
                    st.session_state['target_column'] = target_column
                    
                    st.success("✓ Forecast generated successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
    
    # Display results if forecast exists
    if 'forecast_data' in st.session_state:
        st.markdown("---")
        st.header("📊 Forecast Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Historical Mean",
                f"{st.session_state['summary']['historical_stats']['mean']:,.2f}"
            )
        
        with col2:
            st.metric(
                "Forecast Mean",
                f"{st.session_state['summary']['forecast_stats']['mean']:,.2f}",
                delta=f"{st.session_state['summary']['trend']['change_percent']:.2f}%"
            )
        
        with col3:
            st.metric(
                "Trend",
                st.session_state['summary']['trend']['direction'].title()
            )
        
        with col4:
            st.metric(
                "Forecast Periods",
                st.session_state['forecast_data']['horizon']
            )
        
        # Visualization
        st.subheader("📈 Forecast Visualization")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        periods = st.session_state['periods']
        values = st.session_state['values']
        forecast_labels = st.session_state['forecast_labels']
        forecast_values = st.session_state['forecast_data']['point_forecast']
        quantiles = st.session_state['quantiles']
        
        # Create x-axis
        total_points = len(values) + len(forecast_values)
        all_x = list(range(total_points))
        hist_x = all_x[:len(values)]
        forecast_x = all_x[len(values)-1:]
        
        # Plot
        ax.plot(hist_x, values, label="Historical Data", color="#3498db", 
                linewidth=2, marker='o', markersize=4)
        
        forecast_with_connection = [values[-1]] + forecast_values
        ax.plot(forecast_x, forecast_with_connection, label="Forecast", 
                color="#2ecc71", linewidth=2, marker='s', markersize=4)
        
        # Confidence intervals
        quantiles_with_connection = np.vstack([[values[-1]] * quantiles.shape[1], quantiles])
        
        ax.fill_between(forecast_x, quantiles_with_connection[:, 1], 
                        quantiles_with_connection[:, 9], alpha=0.2, 
                        color='#2ecc71', label='80% Prediction Interval')
        
        ax.fill_between(forecast_x, quantiles_with_connection[:, 3], 
                        quantiles_with_connection[:, 7], alpha=0.3, 
                        color='#2ecc71', label='50% Prediction Interval')
        
        # Labels
        all_labels = periods + forecast_labels
        step = max(1, len(all_labels) // 12)
        tick_positions = list(range(0, len(all_labels), step))
        tick_labels = [all_labels[i] for i in tick_positions]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.set_title(f"TimesFM Forecast: {st.session_state['target_column']}", 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Period")
        ax.set_ylabel(st.session_state['target_column'])
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Forecast table
        st.subheader("📋 Forecast Values")
        
        forecast_df = pd.DataFrame({
            'Period': st.session_state['forecast_data']['periods'],
            'Forecast': st.session_state['forecast_data']['point_forecast'],
            'Lower (10%)': st.session_state['forecast_data']['quantiles']['q10'],
            'Lower (25%)': st.session_state['forecast_data']['quantiles']['q25'],
            'Median': st.session_state['forecast_data']['quantiles']['q50'],
            'Upper (75%)': st.session_state['forecast_data']['quantiles']['q75'],
            'Upper (90%)': st.session_state['forecast_data']['quantiles']['q90']
        })
        
        st.dataframe(forecast_df.style.format({
            'Forecast': '{:,.2f}',
            'Lower (10%)': '{:,.2f}',
            'Lower (25%)': '{:,.2f}',
            'Median': '{:,.2f}',
            'Upper (75%)': '{:,.2f}',
            'Upper (90%)': '{:,.2f}'
        }), use_container_width=True)
        
        # Chat interface
        st.markdown("---")
        st.subheader("💬 Ask Questions About Your Forecast")
        
        user_query = st.text_input(
            "Ask me anything about the forecast:",
            placeholder="e.g., What is the trend? What is the highest forecasted value?"
        )
        
        if user_query:
            response = generate_chat_response(
                user_query, 
                st.session_state['forecast_data'],
                st.session_state['summary']
            )
            st.info(response)
        
        # Example questions
        with st.expander("💡 Example Questions"):
            st.markdown("""
            - What is the trend?
            - What is the highest forecasted value?
            - What is the average forecast?
            - Show me the confidence interval
            - Compare historical and forecast data
            - What is the total forecasted amount?
            """)

else:
    # Welcome screen
    st.info("👈 Please upload a CSV file or paste your data to get started")
    
    st.markdown("""
    ### How to use this tool:
    
    1. **Upload Data**: Choose to upload a CSV file or paste text data
    2. **Configure**: Select your date column, target column, and forecast settings
    3. **Generate**: Click the "Generate Forecast" button
    4. **Analyze**: View the forecast visualization and ask questions
    
    ### Supported Features:
    
    - ✅ Multiple time periods (Day, Week, Month, Year)
    - ✅ Various aggregation methods (Sum, Mean, Median, Count)
    - ✅ Customizable forecast horizon (1-24 periods)
    - ✅ Confidence intervals (50% and 80%)
    - ✅ Interactive Q&A about forecasts
    """)