# Black-Scholes Pricing and Risk Engine

A Python implementation of the Black-Scholes model for European options, with analytical Greeks, implied volatility solving, and Monte Carlo cross-validation.

## Features

- **Analytical pricing** for European call and put options
- **Closed-form Greeks**: Delta, Gamma, Vega, Theta, Rho
- **Hybrid Newton-Raphson / Bisection** implied volatility solver — uses Newton when it converges, falls back to bisection in regions where Newton is unstable (flat vega, deep ITM/OTM, near-expiry)
- **Monte Carlo cross-validation** with **antithetic-variate** variance reduction for sanity-checking the analytical price
- **Edge-case verification** at T → 0 and σ → 0 boundaries where analytical formulas degenerate

## Files

| File | Purpose |
|---|---|
| `black_scholes.py` | Core analytical pricing engine and Greeks |
| `mc_black_scholes.py` | Monte Carlo simulator with antithetic variates |
| `test_black_scholes.py` | Test suite (analytical vs Monte Carlo, edge cases, put-call parity) |
| `UserInterfaceManager.py` | CLI for interactive pricing |
| `bs_test_results.csv` | Recorded test outputs |
| `convergence_plot.png` | MC convergence to analytical price |
| `variance_reduction_plot.png` | Variance reduction from antithetic variates |

## Why a hybrid IV solver?

Pure Newton-Raphson converges quickly when it works but fails when:
- The option is deep in- or out-of-the-money (vega → 0, Newton step diverges)
- The initial guess is far from the true root
- Time-to-expiry is very small (function becomes nearly flat)

Pure bisection always converges but is slow (linear vs Newton's quadratic). The hybrid approach uses Newton when the step size is reasonable and the function value is decreasing, and falls back to bisection otherwise. This gives Newton's speed in the common case and bisection's reliability at the edges.

## Why antithetic variates?

For each random sample Z in the Monte Carlo path, also simulate −Z. The two paths are negatively correlated, so averaging them reduces variance without increasing the number of samples. Cuts MC variance roughly in half for the same compute. The `variance_reduction_plot.png` shows the empirical improvement.

## Running

```bash
python black_scholes.py        # Direct pricing
python mc_black_scholes.py     # Monte Carlo verification
python test_black_scholes.py   # Full test suite
```

## Notes

- Assumes European exercise, no dividends, constant volatility, log-normal returns
- Stable across the parameter ranges S ∈ [1, 1000], K ∈ [1, 1000], T ∈ [0.01, 5], σ ∈ [0.01, 2], r ∈ [0, 0.2]
