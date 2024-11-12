import logging , os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_forecast(model, steps_ahead, model_type="arima", time_stamp=60, scaler=None):
    """
    Generate forecast for the specified number of steps ahead.
    Args:
    - model: Trained model (ARIMA/SARIMA/LSTM).
    - steps_ahead: Number of steps (e.g., months) to forecast.
    - model_type: Type of model used ("arima", "sarima", or "lstm").
    - time_stamp: Time window for LSTM.
    - scaler: Scaler object for inverse transformation (for LSTM).
    
    Returns:
    - forecast: Forecasted values with confidence intervals where applicable.
    """
    if model_type in ["arima", "sarima"]:
        # Extract forecast and confidence intervals
        forecast_result = model.get_forecast(steps=steps_ahead).summary_frame(alpha=0.05)
        forecast = forecast_result["mean"].values
        conf_int = (forecast_result["mean_ci_lower"].values, forecast_result["mean_ci_upper"].values)
        return forecast, conf_int
    elif model_type == "lstm" and scaler is not None:
        lstm_forecast = []
        last_time_window = scaler.transform(test_data[-time_stamp:].values.reshape(-1, 1))
        for _ in range(steps_ahead):
            next_pred = model.predict(last_time_window.reshape(1, time_stamp, 1))
            lstm_forecast.append(next_pred[0][0])
            last_time_window = np.append(last_time_window[1:], next_pred, axis=0)
        lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()
        return lstm_forecast, None
    else:
        raise ValueError("Invalid model type or missing scaler for LSTM")

def plot_forecast(train_data, test_data, forecast, conf_int=None, model_name="Model Forecast"):
    """
    Visualize the forecast along with historical data.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(train_data.index, train_data.values, label="Training Data")
    plt.plot(test_data.index, test_data.values, label="Test Data")
    future_index = pd.date_range(start=test_data.index[-1], periods=len(forecast) + 1, freq='M')[1:]
    plt.plot(future_index, forecast, label=f"{model_name} Forecast", color='orange')
    
    if conf_int is not None:
        plt.fill_between(future_index, conf_int[0], conf_int[1], color='orange', alpha=0.2, label="Confidence Interval")
    
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"Tesla Stock Price Forecast using {model_name}")
    plt.legend()
    plt.show()