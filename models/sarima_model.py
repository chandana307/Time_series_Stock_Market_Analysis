from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os
import pandas as pd

def train_sarima(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), steps=30, show_plot=True):
    """
    Trains SARIMA model and returns forecast.
    Also plots and saves the forecast if show_plot is True.
    """

    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    model = SARIMAX(df['Close'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast_values = model_fit.forecast(steps=steps)

    # Create forecast index
    last_date = df.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)

    forecast = pd.Series(forecast_values.values, index=forecast_index)

    if show_plot:
        os.makedirs("outputs", exist_ok=True)

        plt.style.use("seaborn-whitegrid")
        plt.figure(figsize=(12, 6))

        # Plot historical data
        plt.plot(df['Close'], label="Historical", linewidth=2)
        plt.plot(forecast.index, forecast, label="SARIMA Forecast", color='green', linewidth=2, marker='o')
        plt.axvline(x=forecast.index[0], color='gray', linestyle='--', label='Forecast Start')

        # Zooming logic
        start_zoom = df.index[-200] if len(df) >= 200 else df.index[0]
        plt.xlim(start_zoom, forecast.index[-1] + pd.Timedelta(days=5))

        # Labels
        plt.title("SARIMA Forecast")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig("outputs/sarima_forecast.png")
        plt.show()

    return forecast