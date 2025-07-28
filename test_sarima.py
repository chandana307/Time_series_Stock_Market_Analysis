from utils.preprocessing import load_and_clean_data
from models.sarima_model import train_sarima

df = load_and_clean_data("data/raw_data.csv")
forecast = train_sarima(df, steps=30)

print("Forecasted Prices (SARIMA):")
print(forecast)