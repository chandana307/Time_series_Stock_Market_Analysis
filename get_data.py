import yfinance as yf
import pandas as pd
import os

# ✅ Create folders if not exist
os.makedirs("data", exist_ok=True)

# ✅ List of tickers
tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

# ✅ Download and store all data
all_data = []

print("📥 Downloading data...")

for ticker in tickers:
    print(f"→ {ticker}")
    df = yf.download(ticker, start="2010-01-01", end="2024-01-01", interval="1d")
    
    if not df.empty:
        df.reset_index(inplace=True)
        df['Company'] = ticker
        all_data.append(df)
    else:
        print(f"⚠️ No data found for {ticker}")

# ✅ Combine into one big DataFrame
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Combined dataset rows: {len(combined_df)}")

    # ✅ Take random sample of 10,000 rows
    sample_df = combined_df.sample(n=10000, random_state=42)

    # ✅ Save both files
    combined_df.to_csv("data/full_data.csv", index=False)
    sample_df.to_csv("data/sample_10k.csv", index=False)

    print("✅ Saved full_data.csv and sample_10k.csv")
else:
    print("❌ No data downloaded. Check tickers or internet connection.")