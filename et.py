import yfinance as yf
import pandas as pd
import os

TICKERS = ['AAPL', 'MSFT', 'SPY']
START_DATE = '2020-01-01'
END_DATE = '2023-12-31'
OUTPUT_FILE = 'clean_finance_data.csv'

def build_dataset():
    raw_data = yf.download(TICKERS, start=START_DATE, end=END_DATE, group_by='ticker')
    
    processed_dfs = []
    
    for ticker in TICKERS:
        df = raw_data[ticker].dropna().copy()
        
        # Feature Engineering
        df['Ticker'] = ticker
        df['Daily_Return'] = df['Close'].pct_change()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Volatility_20d'] = df['Daily_Return'].rolling(window=20).std()
        
        # Target Variable for later
        df['Target_Next_Day_Return'] = df['Daily_Return'].shift(-1)
        
        # Clean and append
        df = df.dropna()
        processed_dfs.append(df)
        
    # Combine all data
    final_df = pd.concat(processed_dfs)

    final_df.to_csv(OUTPUT_FILE, index=True)

if __name__ == "__main__":
    build_dataset()