import pandas as pd

def load_and_clean_data(filepath):
    # Load data and parse date as index
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # Rename multi-indexed columns if needed
    if df.columns.nlevels > 1 or isinstance(df.columns[0], tuple):
        df.columns = df.columns.get_level_values(-1)

    # Keep only required columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Convert to numeric (forcefully), handle errors
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Sort by date
    df.sort_index(inplace=True)

    return df