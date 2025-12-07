import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ============================================================
#   Black–Scholes Closed-Form for European Call Options
# ============================================================

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes price for a European call."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ============================================================
#   Monte Carlo Simulation Primitives
# ============================================================

def simulate_terminal_stock(S0, r, sigma, T, n_paths, antithetic=False, rng=None):
    """
    Simulate terminal stock price ST under risk-neutral GBM.
    Supports antithetic variates for variance reduction.
    """
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

    ST = S0 * np.exp((r - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * z_all)
    return ST


def mc_naive_call(S0, K, r, sigma, T, n_paths, antithetic=False, rng=None):
    """Naive Monte Carlo estimator for a call option."""
    ST = simulate_terminal_stock(S0, r, sigma, T, n_paths, antithetic, rng)
    payoffs = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoffs
    return discounted.mean(), discounted.var(ddof=1), discounted


# ============================================================
#   Control Variate Monte Carlo Estimator
# ============================================================

def mc_control_variate_call(S0, K, r, sigma, T, n_paths, antithetic=False, rng=None):
    """
    Control variate estimator using discounted ST (expected = S0).
    Returns (cv_mean, cv_variance, naive_mean, naive_variance).
    """
    if rng is None:
        rng = np.random.default_rng()

    ST = simulate_terminal_stock(S0, r, sigma, T, n_paths, antithetic, rng)
    Y = np.exp(-r * T) * np.maximum(ST - K, 0.0)   # payoffs
    X = np.exp(-r * T) * ST                        # control variate (E[X] = S0)

    # Optimal control variate coefficient
    cov = np.cov(Y, X, ddof=1)
    cov_yx = cov[0, 1]
    var_x = cov[1, 1]
    b_opt = cov_yx / var_x if var_x > 0 else 0.0

    # Adjust samples
    Y_cv = Y - b_opt * (X - S0)

    return Y_cv.mean(), Y_cv.var(ddof=1), Y.mean(), Y.var(ddof=1), b_opt


# ============================================================
#   Example Test & Plots
# ============================================================

if __name__ == "__main__":
    # Option parameters
    S0 = 100
    K = 100
    r = 0.01
    sigma = 0.25
    T = 30/365

    true_price = bs_call_price(S0, K, T, r, sigma)
    print("Black-Scholes analytic price:", true_price)

    n_list = [500, 1000, 5000, 10000, 20000, 50000]
    rng = np.random.default_rng(123)

    naive_means = []
    cv_means = []
    naive_vars = []
    cv_vars = []

    for n in n_list:
        ## Naive MC
        naive_mean, naive_var, _ = mc_naive_call(S0, K, r, sigma, T, n, rng=rng)

        ## Control Variate MC
        cv_mean, cv_var, _, _, b_opt = mc_control_variate_call(S0, K, r, sigma, T, n, rng=rng)

        naive_means.append(naive_mean)
        cv_means.append(cv_mean)
        naive_vars.append(naive_var)
        cv_vars.append(cv_var)

        print(f"\nPaths = {n:,}")
        print("  Naive MC        → mean =", naive_mean, "var =", naive_var)
        print("  Control Variate → mean =", cv_mean, "var =", cv_var)
        print("  Variance Reduction =", naive_var / cv_var, "x")
        print("  Optimal b =", b_opt)

    # -------- Plot variances --------
    plt.figure(figsize=(8,4))
    plt.loglog(n_list, naive_vars, marker='o', label="Naive Variance")
    plt.loglog(n_list, cv_vars, marker='o', label="Control Variate Variance")
    plt.title("Monte Carlo Variance Comparison")
    plt.xlabel("Number of paths (log)")
    plt.ylabel("Variance (log)")
    plt.grid(True)
    plt.legend()
    plt.show()
