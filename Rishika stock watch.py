import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Sector Tickers
# -------------------------------
tickers = {
    'Nuclear': 'AAPL',
    'AI Chips': 'SMH',
    'Quantum': 'MSMT',
    'SMR': 'SMR',
    'Benchmark': 'SPY'
}

start_date = "2018-01-01"
end_date = "2025-06-01"

# -------------------------------
# Risk-Free Rate
# -------------------------------
irx_data = yf.download("^IRX", start=start_date, end=end_date)['Close']
risk_free_rate = irx_data.mean() / 100 if not irx_data.empty else 0  

# -------------------------------
# Metric Functions
# -------------------------------
def calculate_cagr(start_value, end_value, num_years):
    return ((end_value / start_value) ** (1 / num_years)) - 1

def calculate_annualized_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(daily_returns, rf=risk_free_rate):
    excess_return = daily_returns.mean() * 252 - rf
    volatility = calculate_annualized_volatility(daily_returns)
    return excess_return / volatility if volatility != 0 else np.nan

def calculate_max_drawdown(prices):
    running_max = prices.cummax()
    drawdown = prices / running_max - 1
    max_dd = drawdown.min()
    end_dd_date = drawdown.idxmin()
    start_dd_date = prices.loc[:end_dd_date].idxmax()
    return abs(max_dd), start_dd_date, end_dd_date

# -------------------------------
# Main Logic
# -------------------------------
results = []
normalized_returns = pd.DataFrame()

for label, symbol in tickers.items():
    print(f"Fetching {label} ({symbol})...") 
    df = yf.download(symbol, start=start_date, end=end_date)['Adj Close']
    if df.empty:
        print(f" No data for {label}")
        continue

    daily_returns = df.pct_change().dropna()
    normalized_returns[label] = (1 + daily_returns).cumprod()

    start_price = df.iloc[0]
    end_price = df.iloc[-1]
    num_years = (df.index[-1] - df.index[0]).days / 365.25

    cagr = calculate_cagr(start_price, end_price, num_years) * 100
    std_dev = calculate_annualized_volatility(daily_returns) * 100
    sharpe = calculate_sharpe_ratio(daily_returns)
    max_dd, peak_date, trough_date = calculate_max_drawdown(df)

    results.append({
        'Sector': label,
        'CAGR (%)': round(cagr, 2),
        'Annualized Std Dev (%)': round(std_dev, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Max Drawdown': f"{max_dd * 100:.2f}% ({peak_date.strftime('%m-%Y')} to {trough_date.strftime('%m-%Y')})"
    })

# -------------------------------
# Display Performance Table
# -------------------------------
performance_df = pd.DataFrame(results).set_index('Sector')
print("\n Rishika Market Watch - Sector Performance:")
print(performance_df)

# -------------------------------
# Plot Cumulative Returns
# -------------------------------
normalized_returns.plot(figsize=(12, 6), title='Cumulative Returns by Sector (Growth of $1)')
plt.ylabel("Value of $1")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()
