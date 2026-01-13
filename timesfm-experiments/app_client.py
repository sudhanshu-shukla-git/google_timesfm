import requests
import base64
from PIL import Image
import io

# Make forecast request
try:
    response = requests.post(
        "http://localhost:8000/forecast",
        json={
            "csv_path": "./datas/GoldPrices.csv",  # Path to your CSV file
            "target_column": "India(INR)",
            "date_column": "Date",
            "horizon": 12,  # Forecast next 6 periods
            "aggregation": "sum",  # Options: 'sum', 'mean', 'median', 'count'
            "time_period": "month"  # Options: 'day', 'week', 'month', 'year'
        },
        timeout=120
    )
    
    response.raise_for_status()
    result = response.json()
    
    print("✓ Forecast Success!")
    
    print("\n=== Formatted Data (Last 6 Historical + 6 Forecast) ===")
    for item in result['data']:
        if 'history' in item:
            print(f"{item['name']}: {item['history']:,.2f} (Historical) - Color: {item['color']}")
        else:
            print(f"{item['name']}: {item['forecast']:,.2f} (Forecast) - Color: {item['color']}")
    
    print(f"\n=== Summary ===")
    print(f"Target Column: {result['target_column']}")
    print(f"Time Period: {result['summary']['data_info']['time_period']}")
    print(f"Total Periods: {result['summary']['data_info']['total_periods']}")
    
    print(f"\n=== Historical Statistics ===")
    hist_stats = result['summary']['historical_stats']
    print(f"Mean: {hist_stats['mean']:,.2f}")
    print(f"Total: {hist_stats['total']:,.2f}")
    
    print(f"\n=== Forecast Statistics ===")
    fcst_stats = result['summary']['forecast_stats']
    print(f"Mean: {fcst_stats['mean']:,.2f}")
    print(f"Total: {fcst_stats['total']:,.2f}")
    
    print(f"\n=== Trend Analysis ===")
    trend = result['summary']['trend']
    print(f"Direction: {trend['direction']}")
    print(f"Change: {trend['change_percent']}%")
    
    # Save and display the forecast image
    image_data = base64.b64decode(result['image_base64'])
    image = Image.open(io.BytesIO(image_data))
    image.save("forecast_plot.png")
    print("\n✓ Forecast plot saved as 'forecast_plot.png'")
    image.show()
    
    # Print the data array in the format you requested
    print("\n=== Data Array (JSON Format) ===")
    import json
    print(json.dumps(result['data'], indent=2))
    
except requests.exceptions.ConnectionError:
    print("❌ Error: Cannot connect to API. Is the server running on http://localhost:8000?")
except requests.exceptions.HTTPError as e:
    print(f"❌ HTTP Error: {e}")
    if hasattr(e.response, 'text'):
        print(f"Response: {e.response.text}")
except Exception as e:
    print(f"❌ Error: {e}")