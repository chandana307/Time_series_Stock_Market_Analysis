from utils.preprocessing import load_and_clean_data
from models.lstm_model import train_lstm

df = load_and_clean_data("data/raw_data.csv")
forecast = train_lstm(df, steps=30, show_plot=True)

print("Forecasted Prices (LSTM):")
print(forecast)