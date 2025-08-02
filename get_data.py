import yfinance as yf   #for fetching the data 
import pandas as pd     #for data framing
import os             # for creating the folder 

os.makedirs("data", exist_ok=True)

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

all_data = []

print("Downloading data...")

for ticker in tickers:
    print(f"→ {ticker}")
    df = yf.download(ticker, start="2010-01-01", end="2024-01-01", interval="1d")
    
    if not df.empty:
        df.reset_index(inplace=True)
        df['Company'] = ticker #Adds company name as a new column (df['Company'] = ticker)
        all_data.append(df) #Appends each company's data to the all_data list
    else:
        print(f"⚠️ No data found for {ticker}")

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)#Merges all company data into one DataFrame
    print(f" Combined dataset rows: {len(combined_df)}")

    sample_df = combined_df.sample(n=10000, random_state=42)

    combined_df.to_csv("data/full_data.csv", index=False)
    sample_df.to_csv("data/sample_10k.csv", index=False)

    print("Saved full_data.csv and sample_10k.csv")
else:
    print(" No data downloaded. Check tickers or internet connection.")