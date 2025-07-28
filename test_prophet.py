from utils.preprocessing import load_and_clean_data
from models.prophet_model import train_prophet

df = load_and_clean_data("data/raw_data.csv")
forecast = train_prophet(df, periods=30)

print("Prophet Forecast:")
print(forecast)