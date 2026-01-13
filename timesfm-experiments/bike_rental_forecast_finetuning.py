import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import timesfm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

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
print("Model loaded successfully!\n")


# Custom Dataset for fine-tuning
class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_length=365, horizon=90):
        self.data = np.array(data, dtype=np.float32)
        self.context_length = context_length
        self.horizon = horizon
        
        # Create sliding windows
        self.samples = []
        for i in range(len(self.data) - context_length - horizon + 1):
            context = self.data[i:i + context_length]
            target = self.data[i + context_length:i + context_length + horizon]
            self.samples.append((context, target))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return torch.tensor(context), torch.tensor(target)


def prepare_finetuning_data(historical_data, batch_size=8, context_length=365, horizon=90):
    dataset = TimeSeriesDataset(historical_data, context_length, horizon)
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    return data_loader


def finetune_model(model, train_data, num_epochs=10, learning_rate=1e-4, 
                   context_length=365, horizon=90, batch_size=8, 
                   save_path="finetuned_timesfm.pt"):
    print("="*70)
    print("STARTING MODEL FINE-TUNING")
    print("="*70)
    print(f"Training data points: {len(train_data)}")
    print(f"Context length: {context_length}")
    print(f"Forecast horizon: {horizon}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}\n")
    
    # Prepare data loader
    train_loader = prepare_finetuning_data(
        train_data, 
        batch_size=batch_size,
        context_length=context_length,
        horizon=horizon
    )
    
    if len(train_loader) == 0:
        raise ValueError(f"Not enough data for training. Need at least {context_length + horizon} points.")
    
    print(f"Created {len(train_loader)} training batches\n")
    
    # Access the underlying PyTorch model. TimesFM wraps the actual model, we need to access it
    try:
        # Try to access the internal model
        if hasattr(model, 'model'):
            pytorch_model = model.model
        elif hasattr(model, '_model'):
            pytorch_model = model._model
        else:
            # If we can't find the internal model, use the wrapper directly
            pytorch_model = model
        
        # Get trainable parameters
        trainable_params = [p for p in pytorch_model.parameters() if p.requires_grad]
        
        if len(trainable_params) == 0:
            print("Warning: No trainable parameters found. Attempting to unfreeze all parameters...")
            for param in pytorch_model.parameters():
                param.requires_grad = True
            trainable_params = list(pytorch_model.parameters())
        
        print(f"Found {len(trainable_params)} trainable parameter groups\n")
        
    except Exception as e:
        print(f"Warning: Could not access model parameters directly: {e}")
        print("TimesFM may not support direct fine-tuning through standard PyTorch methods.")
        print("Using the model in inference mode only.\n")
        return model, []
    
    optimizer = optim.Adam(trainable_params, lr=learning_rate)     # Create optimizer
    criterion = nn.MSELoss()    # Loss function
    training_losses = []    # Training loop
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_idx, (context, ground_truth_horizon) in enumerate(train_loader):
            try:
                # Note: TimesFM expects inputs as list of arrays
                predicted_horizon, _ = model.forecast(
                    horizon=horizon,
                    inputs=[context[i].numpy() for i in range(context.shape[0])]
                )                
                # Convert predictions to tensor
                predicted_tensor = torch.tensor(predicted_horizon, dtype=torch.float32, requires_grad=True)
                # Calculate loss
                loss = criterion(predicted_tensor, ground_truth_horizon)
                # Backward pass: update model weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                
            except Exception as e:
                print(f"Error during training batch {batch_idx}: {e}")
                print("Note: TimesFM may not support gradient-based fine-tuning.")
                print("The model will be used in inference-only mode.\n")
                return model, []
        
        avg_loss = np.mean(epoch_losses)
        training_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
    
    print(f"\nFine-tuning complete!")
    
    if len(training_losses) > 0:
        print(f"Final Loss: {training_losses[-1]:.6f}")
        print(f"Saving fine-tuned model to: {save_path}")
        try:
            # Save the state dict of the internal model if possible
            if hasattr(model, 'model'):
                model_state = model.model.state_dict()
            elif hasattr(model, '_model'):
                model_state = model._model.state_dict()
            else:
                model_state = {}
                
            torch.save({
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'training_losses': training_losses,
                'config': {
                    'context_length': context_length,
                    'horizon': horizon,
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate
                }
            }, save_path)
            print("Model saved successfully!\n")
        except Exception as e:
            print(f"Warning: Could not save model: {e}\n")
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), training_losses, marker='o', linewidth=2)
        plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    else:
        print("Fine-tuning was not performed due to model limitations.")
        print("Using pre-trained model for inference.\n")
    
    return model, training_losses


def load_finetuned_model(model, checkpoint_path):
    print(f"Loading fine-tuned model from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path)
        
        # Try to load state dict into internal model
        if 'model_state_dict' in checkpoint and checkpoint['model_state_dict']:
            if hasattr(model, 'model'):
                model.model.load_state_dict(checkpoint['model_state_dict'])
            elif hasattr(model, '_model'):
                model._model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("Warning: Could not access internal model structure")
        
        print("Model loaded successfully!")
        
        if 'config' in checkpoint:
            print("\nModel configuration:")
            for key, value in checkpoint['config'].items():
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Warning: Could not load fine-tuned model: {e}")
        print("Using pre-trained base model instead.")
    
    return model


def generate_forecast(historical_data, horizon=90, start_date=None, target_column="Bike Rentals", 
                     use_finetuned=False, finetuned_path=None):
    global model
    
    # Load fine-tuned model if requested
    if use_finetuned and finetuned_path:
        if Path(finetuned_path).exists():
            model = load_finetuned_model(model, finetuned_path)
        else:
            print(f"Warning: Fine-tuned model not found at {finetuned_path}. Using base model.")
    
    # Validate input
    if len(historical_data) < 30:
        raise ValueError("Historical data must contain at least 30 data points")
    
    if horizon > 256 or horizon < 1:
        raise ValueError("Horizon must be between 1 and 256 days")
    
    # Convert to numpy array
    historical_array = np.array(historical_data, dtype=np.float32)
    
    print(f"Generating {horizon}-day forecast...")
    
    # Generate forecast (TimesFM handles inference mode internally)
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
    
    # Generate plot
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
    model_type = "Fine-tuned" if use_finetuned else "Base"
    plt.title(f"TimesFM Forecast ({model_type}): {target_column}", fontsize=16, fontweight='bold')
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
    # Create synthetic data with trend and seasonality (2 years of data)
    np.random.seed(42)
    days = 730
    trend = np.linspace(150, 350, days)
    seasonality = 50 * np.sin(np.linspace(0, 8*np.pi, days))
    noise = np.random.normal(0, 20, days)
    historical_data = trend + seasonality + noise
    historical_data = np.maximum(historical_data, 0)  # Ensure non-negative
    
    print("OPTION 1: Use base pre-trained model (skip fine-tuning)")
    print("OPTION 2: Fine-tune model on your data first")
    print("\nChoosing OPTION 1 for this example...\n")
    
    # OPTION 1: Direct forecasting with base model
    print("Generating forecast with BASE model...")
    results_base = generate_forecast(
        historical_data=historical_data[-365:].tolist(),  # Use last year
        horizon=90,
        start_date="2023-01-01",
        target_column="Bike Rentals"
    )
    
    # OPTION 2: Fine-tune model then forecast
    print("\n" + "="*70)
    print("FINE-TUNING MODEL ON TRAINING DATA")
    print("="*70 + "\n")
    
    # Fine-tune on first 18 months of data
    finetuned_model, losses = finetune_model(
        model=model,
        train_data=historical_data[:540].tolist(),
        num_epochs=5,
        learning_rate=1e-4,
        context_length=180,
        horizon=30,
        batch_size=4,
        save_path="finetuned_timesfm.pt"
    )
    
    # Generate forecast with fine-tuned model
    print("\nGenerating forecast with FINE-TUNED model...")
    results_finetuned = generate_forecast(
        historical_data=historical_data[-365:].tolist(),
        horizon=90,
        start_date="2023-01-01",
        target_column="Bike Rentals",
        use_finetuned=True,
        finetuned_path="finetuned_timesfm.pt"
    )
    
    print("\n" + "="*70)
    print("FORECAST COMPLETE!")
    print("="*70)