import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os

def train_arima(df, order=(5, 1, 0), steps=30, show_plot=True):
    """
    Train ARIMA model and forecast next 'steps' days.

    Parameters:
        df: DataFrame with 'Close' column
        order: ARIMA(p,d,q) parameters
        steps: Forecast horizon
        show_plot: If True, displays matplotlib plot (False for Streamlit)

    Returns:
        forecast: pd.Series of forecasted values with datetime index
    """
    close_prices = df['Close']

    # Fit the ARIMA model
    model = ARIMA(close_prices, order=order)
    model_fit = model.fit()

    # Forecast the next 'steps' days
    forecast_values = model_fit.forecast(steps=steps)
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    forecast = pd.Series(forecast_values.values, index=future_dates)

    if show_plot:
        os.makedirs("outputs", exist_ok=True)
        plt.style.use("seaborn-whitegrid")
        plt.figure(figsize=(12, 6))
        plt.plot(close_prices, label='Historical Close', linewidth=2)
        plt.plot(forecast.index, forecast, label='Forecast', color='red', linewidth=2, marker='o')
        plt.axvline(x=forecast.index[0], color='gray', linestyle='--', label='Forecast Start')
        plt.title('ARIMA Forecast')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig("outputs/arima_forecast.png")  # Save PNG
        plt.show()

    return forecast