import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

def train_prophet(df, periods=30, show_plot=True):
    """
    Train Prophet model and forecast next 'periods' days.

    Parameters:
        df: DataFrame with 'Close' and datetime index
        periods: Forecast horizon
        show_plot: If True, shows plot (False for Streamlit)

    Returns:
        forecast: DataFrame with ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    """

    # Prepare data in Prophet format
    prophet_df = df.reset_index()[['Close']]
    prophet_df['ds'] = pd.to_datetime(df.index).tz_localize(None)  # ← FIXED
    prophet_df.rename(columns={'Close': 'y'}, inplace=True)

    # Initialize and train model
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)

    # Create future dates
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    if show_plot:
        os.makedirs("outputs", exist_ok=True)

        plt.style.use("seaborn-whitegrid")
        fig = model.plot(forecast)
        plt.title("Prophet Forecast")
        plt.xlabel("Date")
        plt.ylabel("Close Price")

        # Forecast start line
        forecast_start = forecast['ds'].iloc[-periods]
        plt.axvline(x=forecast_start, color='gray', linestyle='--', label='Forecast Start')

        plt.legend()
        plt.tight_layout()
        plt.savefig("outputs/prophet_forecast.png")
        plt.show()

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)