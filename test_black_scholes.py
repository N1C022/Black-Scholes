
#Tests:
# - Black-Scholes price vs. market option price
# - Implied volatility vs. Yahoo IV
# - Error statistics and paired t-test


import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy.stats import ttest_rel

from black_scholes import bs_price, implied_vol_newton

# Parameters
TICKER = "AAPL"
RISK_FREE_RATE = 0.045     # approximate
MIN_VOLUME = 50            # avoid illiquid options
MAX_BIDASK_SPREAD = 5.0    # skip broken options
MAX_IV = 3.0               # skip insane implied vols
PRINT_LIMIT = 10           # print first N detailed rows

# Fetch stock + options
ticker = yf.Ticker(TICKER)
S = ticker.history(period="1d")["Close"].iloc[-1]

exp = ticker.options[0]  # nearest expiration
chain = ticker.option_chain(exp)
calls = chain.calls


# Compute time to expiration
T = (datetime.strptime(exp, "%Y-%m-%d") - datetime.utcnow()).days / 365

# Clean option chain (filter bad data)
df = calls.copy()
df = df[df["volume"] > MIN_VOLUME]
df = df[df["impliedVolatility"] < MAX_IV]
df = df[(df["ask"] - df["bid"]) < MAX_BIDASK_SPREAD]

df = df.reset_index(drop=True)

# Calculate BS price + recovered IV
results = []

for i, row in df.iterrows():
    K = float(row["strike"])
    market_price = float(row["lastPrice"])
    market_iv = float(row["impliedVolatility"])
    if market_price <= 0:
        continue

    # Black-Scholes theoretical price
    bs_p = bs_price(S, K, T, RISK_FREE_RATE, market_iv, "call")

    # Percentage difference
    pct_diff = (bs_p - market_price) / market_price * 100

    # Recovered implied volatility
    recovered_iv = implied_vol_newton(
        market_price, S, K, T, RISK_FREE_RATE, "call", initial_sigma=market_iv
    )

    results.append({
        "Strike": K,
        "Market Price": market_price,
        "BS Price": bs_p,
        "Pct Error (%)": pct_diff,
        "Market IV": market_iv,
        "Recovered IV": recovered_iv,
        "IV Diff": recovered_iv - market_iv
    })

results_df = pd.DataFrame(results)

# Summary Statistics

mean_err = results_df["Pct Error (%)"].mean()
median_err = results_df["Pct Error (%)"].median()
corr = results_df[["Market Price", "BS Price"]].corr().iloc[0,1]

# Paired t-test for price differences
t_stat, p_value = ttest_rel(
    results_df["BS Price"],
    results_df["Market Price"]
)

# Print results
print(f"\n===== Black-Scholes Test for {TICKER} ({exp}) =====")
print(f"Current stock price: {S:.2f}")
print(f"Options tested: {len(results_df)}")
print("\n--- Summary Errors ---")
print(f"Mean % error:   {mean_err:.4f}%")
print(f"Median % error: {median_err:.4f}%")
print(f"Correlation (BS vs Market): {corr:.4f}")

print("\n--- Paired t-test ---")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4f}")
if p_value < 0.05:
    print("→ Significant difference between BS and market prices")
else:
    print("→ No statistically significant difference")

print("\n--- First few rows ---")
print(results_df.head(PRINT_LIMIT))

# save to CSV
results_df.to_csv("bs_test_results.csv", index=False)
print("\nSaved results to bs_test_results.csv")
