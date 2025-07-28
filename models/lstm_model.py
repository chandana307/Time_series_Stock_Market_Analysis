import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import os

def train_lstm(df, steps=30, sequence_len=60, show_plot=True):
    """
    Train an LSTM model and forecast future closing prices.

    Parameters:
        df: pd.DataFrame with datetime index and a 'Close' column
        steps: Number of future days to forecast
        sequence_len: Number of previous days to use for training
        show_plot: If True, saves and displays the plot

    Returns:
        forecast_series: pd.Series of forecasted prices with datetime index
    """

    # ✅ Scale the 'Close' price data
    close_data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data)

    # ✅ Create sequences for LSTM input
    X, y = [], []
    for i in range(sequence_len, len(scaled_data) - steps):
        X.append(scaled_data[i - sequence_len:i, 0])
        y.append(scaled_data[i:i + steps, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # ✅ Build the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(steps)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # ✅ Use the last sequence to forecast future values
    last_sequence = scaled_data[-sequence_len:]
    last_sequence = last_sequence.reshape(1, sequence_len, 1)
    forecast_scaled = model.predict(last_sequence, verbose=0)
    forecast = scaler.inverse_transform(forecast_scaled).flatten()

    # ✅ Generate datetime index for forecast
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    forecast_series = pd.Series(forecast, index=future_dates, name='LSTM Forecast')

    # ✅ Plot results if enabled
    if show_plot:
        os.makedirs("outputs", exist_ok=True)
        plt.style.use("seaborn-whitegrid")
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, close_data, label='Historical Close', linewidth=2)
        plt.plot(forecast_series.index, forecast_series.values, color='orange', marker='o', linewidth=2, label='LSTM Forecast')
        plt.axvline(x=forecast_series.index[0], color='gray', linestyle='--', label='Forecast Start')
        plt.title("LSTM Forecast")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig("outputs/lstm_forecast.png")
        plt.show()

    return forecast_series