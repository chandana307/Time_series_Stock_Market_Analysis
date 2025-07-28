from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

def calculate_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return round(rmse, 2), round(mape, 2)