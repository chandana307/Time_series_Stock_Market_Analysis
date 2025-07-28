import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import metrics

from utils.preprocessing import load_and_clean_data
from models.arima_model import train_arima
from models.prophet_model import train_prophet
from models.lstm_model import train_lstm
from models.sarima_model import train_sarima

# Streamlit config
st.set_page_config(page_title="Time Series Forecasting", layout="centered")

st.title("Stock Price Forecasting (ARIMA, Prophet, LSTM, SARIMA)")

# Sidebar controls
model_choice = st.sidebar.selectbox("Select Forecasting Model", ["ARIMA", "Prophet", "LSTM", "SARIMA"])
forecast_range = st.sidebar.selectbox(
    "Select Forecast Horizon", ["7 days", "30 days", "90 days"]
)
steps = int(forecast_range.split()[0])

# Load and clean data
df = load_and_clean_data("data/raw_data.csv")

# Show raw line chart
st.subheader("Historical Close Prices")
st.line_chart(df['Close'])

# Forecast Output Section
st.subheader("Forecast Output")

if model_choice == "ARIMA":
    st.info(f"Forecasting using ARIMA model for {steps} days")
    forecast = train_arima(df, steps=steps, show_plot=False)
    fig, ax = plt.subplots()
    ax.plot(df['Close'], label='Historical', linewidth=2)
    ax.plot(forecast.index, forecast, label='Forecast', color='red', linewidth=2)
    ax.axvline(x=forecast.index[0], color='gray', linestyle='--', label='Forecast Start')
    ax.set_title('ARIMA Forecast')
    ax.legend()
    st.pyplot(fig)
    forecast_df = pd.DataFrame({'Date': forecast.index, 'Predicted Close': forecast.values})
    st.dataframe(forecast_df)

elif model_choice == "Prophet":
    st.info(f"Forecasting using Prophet model for {steps} days")
    forecast = train_prophet(df, periods=steps, show_plot=False)
    st.line_chart(forecast.set_index("ds")["yhat"])
    forecast_renamed = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps).rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted Price',
        'yhat_lower': 'Min Expected',
        'yhat_upper': 'Max Expected'
    })
    st.dataframe(forecast_renamed)

elif model_choice == "LSTM":
    st.info(f"Forecasting using LSTM model for {steps} days")

    # Get forecast (index = dates)
    forecast_series = train_lstm(df, steps=steps, show_plot=False)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(df['Close'], label='Historical (Raw)', linewidth=2)
    ax.plot(forecast_series.index, forecast_series.values, label='LSTM Forecast', color='orange', linewidth=2)
    ax.axvline(x=forecast_series.index[0], color='gray', linestyle='--', label='Forecast Start')
    ax.set_title('LSTM Forecast')
    ax.legend()
    st.pyplot(fig)

    # Table output
    lstm_df = pd.DataFrame({
        'Date': forecast_series.index,
        'Predicted Close': forecast_series.values
    })
    st.dataframe(lstm_df)

elif model_choice == "SARIMA":
    st.info(f"Forecasting using SARIMA model for {steps} days")
    forecast = train_sarima(df, steps=steps, show_plot=False)
    fig, ax = plt.subplots()
    ax.plot(df['Close'], label='Historical', linewidth=2)
    ax.plot(forecast.index, forecast, label='SARIMA Forecast', color='green', linewidth=2)
    ax.axvline(x=forecast.index[0], color='gray', linestyle='--', label='Forecast Start')
    ax.set_title('SARIMA Forecast')
    ax.legend()
    st.pyplot(fig)
    forecast_df = pd.DataFrame({'Date': forecast.index, 'Predicted Close': forecast.values})
    st.dataframe(forecast_df)

# Model Comparison Section
st.subheader("Model Performance Comparison")
comparison_steps = steps
actual = df['Close'][-comparison_steps:]

forecast_arima = train_arima(df, steps=comparison_steps, show_plot=False)
forecast_sarima = train_sarima(df, steps=comparison_steps, show_plot=False)
forecast_prophet_df = train_prophet(df, periods=comparison_steps, show_plot=False)
forecast_prophet = forecast_prophet_df['yhat'].tail(comparison_steps).values
forecast_lstm = train_lstm(df, steps=steps, show_plot=False)

arima_rmse, arima_mape = metrics.calculate_metrics(actual, forecast_arima)
sarima_rmse, sarima_mape = metrics.calculate_metrics(actual, forecast_sarima)
prophet_rmse, prophet_mape = metrics.calculate_metrics(actual.values, forecast_prophet)
lstm_rmse, lstm_mape = metrics.calculate_metrics(actual.values, forecast_lstm)

metrics_df = pd.DataFrame({
    'Model': ['ARIMA', 'SARIMA', 'Prophet', 'LSTM'],
    'RMSE': [arima_rmse, sarima_rmse, prophet_rmse, lstm_rmse],
    'MAPE (%)': [arima_mape, sarima_mape, prophet_mape, lstm_mape]
}).sort_values('RMSE')

st.dataframe(metrics_df.set_index('Model'))

best_model = metrics_df.iloc[0]['Model']
st.success(f"\u2705 Best Model Based on RMSE: **{best_model}**")