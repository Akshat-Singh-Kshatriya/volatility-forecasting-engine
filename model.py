import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = 'all_tickers_data.csv'

def run_multi_ticker_pipeline():
    try:
        df = pd.read_csv(INPUT_FILE, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}. Please export the full table from pgAdmin.")
        return

    tickers = df['Ticker'].unique()
    print(f"Found tickers: {tickers}")
    
    for ticker in tickers:
        ticker_df = df[df['Ticker'] == ticker].copy()
        
        # --- FEATURE ENGINEERING ---
        ticker_df['Target_1d_Return'] = ticker_df['Daily_Return'].shift(-1)
        ticker_df['Lag_1_Return'] = ticker_df['Daily_Return'].shift(1)
        ticker_df['Lag_2_Return'] = ticker_df['Daily_Return'].shift(2)
        
        ticker_df['SMA_20'] = ticker_df['Close'].rolling(window=20).mean()
        ticker_df['Std_Dev_20'] = ticker_df['Close'].rolling(window=20).std()
        ticker_df['Bollinger_Upper'] = ticker_df['SMA_20'] + (ticker_df['Std_Dev_20'] * 2)
        ticker_df['Bollinger_Lower'] = ticker_df['SMA_20'] - (ticker_df['Std_Dev_20'] * 2)
        ticker_df['Price_to_Band'] = (ticker_df['Close'] - ticker_df['Bollinger_Lower']) / (ticker_df['Bollinger_Upper'] - ticker_df['Bollinger_Lower'])
        ticker_df['Volatility_5d'] = ticker_df['Daily_Return'].rolling(window=5).std()
        
        ticker_df = ticker_df.dropna()

        features = [
            'Daily_Return', 'Volatility_20d', 'Volatility_5d', 
            'Lag_1_Return', 'Lag_2_Return', 'Price_to_Band'
        ]
        
        X = ticker_df[features]
        y = ticker_df['Target_1d_Return']
        
        # --- TIME-SERIES CROSS VALIDATION ---
        tscv = TimeSeriesSplit(n_splits=5)
        model = GradientBoostingRegressor(
            n_estimators=150,      
            learning_rate=0.03,    
            max_depth=3,           
            subsample=0.7,         
            random_state=42
        )
        
        all_actuals = []
        all_preds = []
        all_dates = []
        
        for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            all_actuals.extend(y_test.values)
            all_preds.extend(predictions)
            all_dates.extend(y_test.index)
            
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            print(f"Fold {fold} RMSE: {rmse:.6f}")
            
        # --- THE BACKTEST LOGIC ---
        results_df = pd.DataFrame({
            'Actual_Return': all_actuals,
            'Predicted_Return': all_preds
        }, index=all_dates)
        
        results_df['Position'] = np.where(results_df['Predicted_Return'] > 0, 1, -1)
        results_df['Strategy_Return'] = results_df['Position'] * results_df['Actual_Return']
        
        results_df['Cumulative_Market'] = (1 + results_df['Actual_Return']).cumprod()
        results_df['Cumulative_Strategy'] = (1 + results_df['Strategy_Return']).cumprod()

        # --- PROFESSIONAL PLOTTING (9 SEPARATE FILES) ---
        sns.set_theme(style="darkgrid")
        
        # Plot 1: Zoomed-in Line Chart
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        subset = results_df.iloc[-100:]
        ax1.plot(subset.index, subset['Actual_Return'], label=f'{ticker} Actual Daily Return', color='#ff4d4d', alpha=0.8)
        ax1.plot(subset.index, subset['Predicted_Return'], label='Predicted Daily Return', color='#00cc44', linewidth=2)
        ax1.set_title(f"{ticker} 1-Day Forecasting: Actual vs Predicted (Last 100 Days)")
        ax1.set_ylabel("Daily Return")
        ax1.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{ticker}_1_forecast_line.png")
        plt.close(fig1) # Close to save memory
        
        # Plot 2: Scatter Plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.regplot(x='Predicted_Return', y='Actual_Return', data=results_df, ax=ax2, 
                    scatter_kws={'alpha':0.3, 'color':'#34495e'}, line_kws={'color':'red'})
        ax2.set_title(f"{ticker} Prediction Correlation: Does the model have an edge?")
        ax2.set_xlabel("Predicted Daily Return")
        ax2.set_ylabel("Actual Daily Return")
        plt.tight_layout()
        plt.savefig(f"{ticker}_2_scatter_correlation.png")
        plt.close(fig2)
        
        # Plot 3: Cumulative Backtest
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        ax3.plot(results_df.index, results_df['Cumulative_Market'], label=f'Buy & Hold {ticker}', color='#3498db', linewidth=2)
        ax3.plot(results_df.index, results_df['Cumulative_Strategy'], label='Algorithmic Strategy', color='#f39c12', linewidth=2)
        ax3.set_title(f"{ticker} Strategy Backtest: Algorithmic Trading vs Buy & Hold")
        ax3.set_ylabel("Cumulative Return Multiplier")
        ax3.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{ticker}_3_cumulative_backtest.png")
        plt.close(fig3)

if __name__ == "__main__":
    run_multi_ticker_pipeline()