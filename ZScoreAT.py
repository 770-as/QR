import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
 

# --- SCENARIO SELECTOR (Modify these for your 4 tests) ---
SYMBOL = 'XLU'            # Stable Asset: Utilities ETF
START_DATE = '2023-01-01' # Period Start
END_DATE = '2024-01-01'   # Period End
WINDOW = 20               # Z-score Lookback
ENTRY_Z = 2.0             # Entry Threshold
TP_PCT = 0.02             # Take Profit: 2%
SL_PCT = 0.01             # Stop Loss: 1%

# 1. Load Data
df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)

# 2. ADF Test (Stationarity)
def run_adf(series):
    res = adfuller(series.dropna())
    print(f"ADF P-Value for {SYMBOL}: {res[1]:.4f}")
    return res[1] < 0.05

is_stationary = run_adf(df['Close'])


# Force the columns to be flat (fixes the ValueError)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Use only the Close column as a Series
close_series = df['Close'].squeeze()

# 2. ADF Test
res = adfuller(close_series.dropna())
print(f"ADF P-Value for {SYMBOL}: {res[1]:.4f}")

# 3. Calculations (Using the Series to ensure single column output)
df['MA'] = close_series.rolling(window=WINDOW).mean()
df['STD'] = close_series.rolling(window=WINDOW).std()
df['Z'] = (close_series - df['MA']) / df['STD']


# 4. Logic with TP and SL
df['Signal'] = 0
df['Entry_Price'] = np.nan

for i in range(WINDOW, len(df)):
    # Entry Logic (Long Only for this example)
    if df['Z'].iloc[i] < -ENTRY_Z and df['Signal'].iloc[i-1] == 0:
        df.at[df.index[i], 'Signal'] = 1
        df.at[df.index[i], 'Entry_Price'] = df['Close'].iloc[i]
    
    # Exit Logic (TP/SL)
    elif df['Signal'].iloc[i-1] == 1:
        entry = df['Entry_Price'].ffill().iloc[i]
        curr_price = df['Close'].iloc[i]
        
        # Check Stop Loss or Take Profit
        if curr_price >= entry * (1 + TP_PCT) or curr_price <= entry * (1 - SL_PCT):
            df.at[df.index[i], 'Signal'] = 0
        else:
            df.at[df.index[i], 'Signal'] = 1 # Keep Position

# 5. Performance
df['Returns'] = df['Close'].pct_change()
df['Strategy'] = df['Signal'].shift(1) * df['Returns']
print(f"Cumulative Return: {df['Strategy'].cumsum().iloc[-1]:.2%}")



   
 

# 4. Quick Result Preview
print("\n--- Last 5 rows of Z-Score calculation ---")
print(df[['Close', 'Z']].tail())







