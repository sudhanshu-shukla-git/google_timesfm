# TimesFM Forecasting Demo

A collection of time series forecasting examples and applications using Google's TimesFM model along with other forecasting approaches.

## Overview

This repository demonstrates various time series forecasting methods with a focus on Google's TimesFM model. It includes examples for:
- Bike rental prediction
- Gold price forecasting
- Interactive web application using Streamlit

## Features

- Zero-shot time series forecasting using TimesFM
- Multiple forecasting methods comparison:
  - Statistical (AutoARIMA, AutoETS)
  - Machine Learning (LightGBM, XGBoost, Random Forest)
  - Deep Learning (TimesFM)
  - GPT-based (TimeGPT)
- Interactive web interface for forecasting
- Confidence interval visualization
- Extensive error metrics (MAE, RMSE, MAPE)

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -e .
```

## Usage

### Web Application

Run the Streamlit app:

```bash
streamlit run app.py
```

### Command Line Examples

Run bike rental forecast:
```bash
python bike_rental_forecast.py
```

Run gold price forecast:
```bash
python gold_price_forecast.py
```

## Data

The repository includes sample datasets:
- `datas/1979-2021.csv`: Gold price historical data
- `datas/GoldPrices.csv`: Additional gold price data
- `datas/sales_data_sample.csv`: Sample sales data

## Requirements

- Python >= 3.13
- Key dependencies:
  - timesfm
  - streamlit
  - langchain
  - lightgbm
  - xgboost
  - statsforecast
  - mlforecast

## License

See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.