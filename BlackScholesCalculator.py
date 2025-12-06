"""
black_scholes.py

Provides:
- bs_price: Black-Scholes price for European call/put
- bs_greeks: delta, gamma, vega, theta, rho
- implied_vol_newton: implied volatility via Newton-Raphson (with fallback to bisection)
- example usage in __main__
"""

import numpy as np
import scipy.stats as st
from math import log, sqrt, exp
from typing import Tuple

EPS = 1e-12

def _norm_pdf(x):
    return st.norm.pdf(x)

def _norm_cdf(x):
    return st.norm.cdf(x)

def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Black-Scholes price for European option.
    S: spot price
    K: strike
    T: time to maturity in years (if 0, return intrinsic)
    r: risk-free continuous rate
    sigma: volatility (annualized)
    option_type: "call" or "put"
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)
    else:
        price = K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    return float(price)

def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> dict:
    """
    Returns Greeks: delta, gamma, vega, theta (per year), rho (per 1% rate)
    Vega here is per 1 vol (so divide by 100 if vol in pct).
    """
    if T <= 0:
        # At expiry: delta is step function; gamma, vega, theta -> 0
        if option_type == "call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = 0.0 if S > K else -1.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = _norm_pdf(d1)
    delta = _norm_cdf(d1) if option_type == "call" else _norm_cdf(d1) - 1
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T)  # per 1 vol (i.e., sigma in decimal)
    # theta: approximate per year (some use per day / 365)
    if option_type == "call":
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * _norm_cdf(d2))
    else:
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * _norm_cdf(-d2))
    rho = K * T * np.exp(-r * T) * (_norm_cdf(d2) if option_type == "call" else -_norm_cdf(-d2))

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega),
            "theta": float(theta), "rho": float(rho)}

def implied_vol_newton(price: float, S: float, K: float, T: float, r: float,
                       option_type: str = "call", initial_sigma: float = 0.2,
                       tol: float = 1e-8, max_iter: int = 100) -> float:
    """
    Implied volatility via Newton-Raphson with vega as derivative.
    Fallback to bisection if divergence or non-convergence.
    """

    if price <= 0:
        return 0.0

    sigma = initial_sigma
    for i in range(max_iter):
        bs_p = bs_price(S, K, T, r, sigma, option_type)
        v = bs_greeks(S, K, T, r, sigma, option_type)["vega"]
        diff = bs_p - price

        if abs(diff) < tol:
            return max(sigma, 0.0)

        # Avoid division by zero
        if v < 1e-12:
            break

        sigma = sigma - diff / v
        if sigma <= 0 or sigma > 5:  # unrealistic, fall back
            break

    # Bisection fallback in [1e-6, 5]
    low, high = 1e-6, 5.0
    p_low = bs_price(S, K, T, r, low, option_type) - price
    p_high = bs_price(S, K, T, r, high, option_type) - price
    if p_low * p_high > 0:
        # can't bracket: return current sigma clipped
        return max(min(sigma, high), low)

    for i in range(200):
        mid = 0.5 * (low + high)
        p_mid = bs_price(S, K, T, r, mid, option_type) - price
        if abs(p_mid) < tol:
            return mid
        if p_low * p_mid <= 0:
            high = mid
            p_high = p_mid
        else:
            low = mid
            p_low = p_mid
    return mid

if __name__ == "__main__":
    # Example usage
    S = 100.0
    K = 100.0
    T = 30 / 365  # 30 days
    r = 0.01
    sigma = 0.25

    call_price = bs_price(S, K, T, r, sigma, "call")
    put_price = bs_price(S, K, T, r, sigma, "put")
    print("Call price:", call_price)
    print("Put price:", put_price)
    print("Call greeks:", bs_greeks(S, K, T, r, sigma, "call"))

    # implied vol example: start with call price, recover sigma
    implied = implied_vol_newton(call_price, S, K, T, r, "call", initial_sigma=0.2)
    print("Recovered implied vol:", implied)
