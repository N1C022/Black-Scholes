import numpy as np
from scipy.stats import norm
#   Black–Scholes closed-form & Greeks

def bs_call_price(S, K, T, r, sigma):
    #Black–Scholes price for a European call option
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_call_delta(S, K, T, r, sigma):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)

def bs_call_vega(S, K, T, r, sigma):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

def bs_call_gamma(S, K, T, r, sigma):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

#   Implied Volatility Solver

def implied_vol_call(target_price, S, K, T, r, tol=1e-8, max_iter=100):

    #Newton-Raphson implied volatility solver with bisection fallback.

    sigma = 0.2  # initial guess
    for _ in range(max_iter):
        price = bs_call_price(S, K, T, r, sigma)
        vega = bs_call_vega(S, K, T, r, sigma)

        if abs(price - target_price) < tol:
            return sigma

        # avoid division by zero
        if vega < 1e-8:
            break

        sigma -= (price - target_price) / vega
        if sigma <= 0:
            break

    # fallback: bisection
    low, high = 1e-8, 5.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        price = bs_call_price(S, K, T, r, mid)
        if price > target_price:
            high = mid
        else:
            low = mid
        if abs(high - low) < tol:
            return mid

    return mid

#   Monte Carlo Engines

def simulate_terminal_stock(S0, r, sigma, T, n_paths, antithetic=False, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        z_all = np.concatenate([z, -z])
        if n_paths % 2 == 1:
            z_all = np.concatenate([z_all, rng.standard_normal(1)])
    else:
        z_all = rng.standard_normal(n_paths)

    return S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z_all)

def mc_naive_call(S0, K, r, sigma, T, n_paths, antithetic=False, rng=None):
    ST = simulate_terminal_stock(S0, r, sigma, T, n_paths, antithetic, rng)
    discounted = np.exp(-r*T) * np.maximum(ST - K, 0)
    return discounted.mean(), discounted.var(ddof=1)

def mc_control_variate_call(S0, K, r, sigma, T, n_paths, antithetic=False, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    ST = simulate_terminal_stock(S0, r, sigma, T, n_paths, antithetic, rng)
    Y = np.exp(-r*T) * np.maximum(ST - K, 0)
    X = np.exp(-r*T) * ST  # control variate, E[X] = S0

    cov = np.cov(Y, X, ddof=1)
    b_opt = cov[0,1] / cov[1,1]
    Y_cv = Y - b_opt * (X - S0)

    return Y_cv.mean(), Y_cv.var(ddof=1), b_opt
