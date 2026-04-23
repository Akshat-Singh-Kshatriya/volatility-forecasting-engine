# End-to-end ML trading pipeline

## Overview
This repository contains a production-ready, end-to-end quantitative data pipeline and algorithmic trading backtester. It is designed to rigorously test whether basic momentum and mean-reversion features can generate a statistical edge on a 1-day trading horizon for highly liquid mega-cap equities (AAPL, MSFT) and broad market indices (SPY).

This project strictly separates data engineering (ETL via PostgreSQL) from data science (Machine Learning via Scikit-Learn) to mimic institutional architecture.

## System Architecture

1. **The Data Pipeline (Phase 1):** * Extracts historical pricing and volume data via Yahoo Finance.
   * Cleans, transforms, and engineers base financial features.
   * Loads structured, validated data into a local **PostgreSQL** database.
2. **The ML Model (Phase 2):**
   * Connects to the database and pulls target equities.
   * Engineers advanced quantitative indicators on the fly (Bollinger Bands, Rolling Volatility, Lags).
   * Trains a **Gradient Boosting Regressor** using strict **Time-Series Cross-Validation** (5-folds) to eliminate look-ahead bias.
3. **The Backtest Engine (Phase 3):**
   * Simulates a Long/Short trading strategy based on the model's daily predictions.
   * Generates out-of-sample performance metrics and visualizes the strategy's cumulative return against a Buy & Hold benchmark.

## Analysis of Results (Hypothesis Testing)
The primary hypothesis of this project was: *Can standard lagging technical indicators (SMA, Bollinger Bands) coupled with a Gradient Boosting algorithm accurately forecast 1-day market direction?*

**Results:**
* **Prediction Correlation:** The scatter plots across all tickers (AAPL, MSFT, SPY) show a near-flat regression line. The model correctly identifies that daily market variance acts largely as a "Random Walk" and minimizes its Root Mean Squared Error (RMSE) by predicting close to the mean.
* **Backtest Performance:** Out-of-sample backtesting reveals that the algorithmic strategy consistently underperformed the Buy & Hold benchmark. 
* **Conclusion:** This project successfully proves the **Efficient Market Hypothesis (EMH)** regarding highly liquid assets on a micro-horizon. It empirically demonstrates that 1-day price movements in mega-caps cannot be reliably arbitraged using historical price/volume data alone, validating the necessity for longer forecasting horizons (e.g., 5-day or 10-day smoothed targets) or alternative datasets (sentiment, macro-economics) for alpha generation.


| Ticker | Asset Type | Average CV RMSE | Volatility Profile |
| :--- | :--- | :--- | :--- |
| **SPY** | S&P 500 ETF | `0.011270` | Baseline (Market) |
| **MSFT** | Mega-Cap Tech | `0.018004` | High Beta |
| **AAPL** | Mega-Cap Tech | `0.018474` | High Beta |

*Insight: As expected, the model achieved a significantly lower error rate on the SPY ETF, as the blended index smooths out the idiosyncratic volatility spikes present in individual tech equities.*

## Strategy Visualizations

Below are the backtest results for the 1-Day forecasting strategy. The charts demonstrate the cumulative return of the algorithmic strategy (Long/Short based on model predictions) versus a standard Buy & Hold approach.

### Apple Inc. (AAPL)
*Observation: Individual tech stocks exhibit higher idiosyncratic noise, testing the model's ability to capture aggressive momentum swings.*
![AAPL Forecast Line](AAPL_1_forecast_line.png)
*(Optional: View [AAPL Scatter](AAPL_2_scatter_correlation.png) | View [AAPL Backtest](AAPL_3_cumulative_backtest.png))*

### Microsoft Corp. (MSFT)
*Observation: Evaluates the model's edge on sustained, high-beta momentum trends.*
![MSFT Forecast Line](MSFT_1_forecast_line.png)
*(Optional: View [MSFT Scatter](MSFT_2_scatter_correlation.png) | View [MSFT Backtest](MSFT_3_cumulative_backtest.png))*

### S&P 500 ETF (SPY)
*Observation: The ultimate baseline. Proves the Efficient Market Hypothesis that broad, highly-arbitraged index funds are incredibly resistant to simple 1-day momentum strategies.*
![SPY Forecast Line](SPY_1_forecast_line.png)
*(Optional: View [SPY Scatter](SPY_2_scatter_correlation.png) |  View [SPY Backtest](SPY_3_cumulative_backtest.png) )*

## Tech Stack

* **Language:** Python
* **Data:** `yfinance` (Yahoo Finance)
* **Machine Learning:** `RandomForest`, `Scikit-Learn` (TimeSeriesSplit)
* **Visualisation:** `Matplotlib`, `Seaborn`

## How to Run

### 1. Clone Repository
```bash
git clone [https://github.com/Akshat-Singh-Kshatriya/volatility-forecasting-engine.git](https://github.com/Akshat-Singh-Kshatriya/volatility-forecasting-engine.git)
cd volatility-forecasting-engine
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Engine
```bash
python et.py
python model.py
```


