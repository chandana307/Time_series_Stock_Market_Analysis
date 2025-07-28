import pandas as pd

def load_and_clean_data(filepath):
    # Load the CSV and parse the first column as index for datew
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    # Drop rows with missing values
    df.dropna(inplace=True)
    # Sort data by date (just in case)
    df.sort_index(inplace=True)
    # Keep only important columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df