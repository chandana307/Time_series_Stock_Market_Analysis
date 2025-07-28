# Time series Stock Price Forecasting App

A Streamlit-powered web app that forecasts stock prices using advanced time series models: **ARIMA**, **SARIMA**, **Prophet**, and **LSTM**. Built to compare model performance using RMSE and MAPE metrics.


📊 Features

- 🧠 Supports 4 forecasting models:
  - ARIMA
  - SARIMA
  - Prophet
  - LSTM (Keras)
- 📅 Select forecast horizon (7 / 30 / 90 days)
- 📉 Visual forecast plots with prediction lines
- 📦 Performance comparison (RMSE / MAPE)
- 📁 Data loading and cleaning utilities
- 🖼️ Saves forecast plots as PNG


🛠️ Tech Stack

- Python 3.10
- [Streamlit](https://streamlit.io/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [Prophet](https://facebook.github.io/prophet/)
- [statsmodels](https://www.statsmodels.org/)
- [TensorFlow / Keras](https://www.tensorflow.org/)



 🗂️ Project Structure

   stock-forecasting-app/
├── data/
│   └── raw_data.csv
├── models/
│   ├── arima_model.py
│   ├── sarima_model.py
│   ├── prophet_model.py
│   └── lstm_model.py
├── utils/
│   └── preprocessing.py
├── outputs/
│   └── forecast images
├── streamlit_app.py
├── get_data.py
├── metrics.py
└── README.md



🚀 Getting Started

1. Clone the Repository


git clone https://github.com/abhijit1620/stock-forecasting-app.git
cd stock-forecasting-app

2. Run App

streamlit run streamlit_app.py

** Author **

Abhijeet Sharma
BTech CSE-AIML | Lucknow
abhijitsharma.ab@gmail.com

