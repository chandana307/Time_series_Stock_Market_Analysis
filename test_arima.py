from utils.preprocessing import load_and_clean_data
from models.arima_model import train_arima

# Load data
df = load_and_clean_data("data/raw_data.csv")
# Train and forecast
forecast = train_arima(df, order=(5,1,0), steps=30)
print("\nForecasted Prices:")
print(forecast)
print(df.dtypes)
print(df.head())
print(type(df['Close'][0]))