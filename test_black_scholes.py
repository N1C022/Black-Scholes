from black_scholes import (
    bs_call_price, implied_vol_call,
    mc_naive_call, mc_control_variate_call
)

import numpy as np
import matplotlib.pyplot as plt


#   Core Validation Tests

def test_monte_carlo_engines():
    S0, K, r, sigma = 100, 100, 0.01, 0.25
    T = 30/365
    TRUE = bs_call_price(S0, K, T, r, sigma)
    rng = np.random.default_rng(42)

    n_paths = 50000
    naive_mean, naive_var = mc_naive_call(S0, K, r, sigma, T, n_paths, rng=rng)
    cv_mean, cv_var, b_opt = mc_control_variate_call(S0, K, r, sigma, T, n_paths, rng=rng)

    print("\n==== Monte Carlo Tests ====")
    print("Analytic BS price:", TRUE)
    print("Naive MC:", naive_mean, "var =", naive_var)
    print("Control Variate MC:", cv_mean, "var =", cv_var)
    print("Variance Reduction =", naive_var / cv_var, "x")
    print("Optimal b =", b_opt)


def test_implied_vol():
    S0, K, r, sigma = 100, 100, 0.01, 0.25
    T = 0.5
    true_price = bs_call_price(S0, K, T, r, sigma)
    iv = implied_vol_call(true_price, S0, K, T, r)

    print("\n==== Implied Volatility Test ====")
    print("True vol:", sigma)
    print("Recovered vol:", iv)


#   Plot 1 — Generic Monte Carlo Convergence (1/N and 1/sqrt(N))

def plot_mc_convergence():
    n_list = np.array([500, 1000, 2000, 5000, 10000, 20000, 50000])
    variances = 1 / n_list
    errors = 1 / np.sqrt(n_list)

    plt.figure(figsize=(8, 5))
    plt.loglog(n_list, variances, marker='o', label="Variance ∼ 1/N")
    plt.loglog(n_list, errors, marker='s', label="Error ∼ 1/√N")
    plt.title("Monte Carlo Convergence Illustration")
    plt.xlabel("Number of Paths (log scale)")
    plt.ylabel("Magnitude (log scale)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_plot.png", dpi=200)
    print("Saved convergence_plot.png")


#   Plot 2 — Naive MC vs Control Variate Variance Reduction

def plot_variance_reduction():
    S0, K, r, sigma = 100, 100, 0.01, 0.25
    T = 30 / 365
    rng = np.random.default_rng(123)

    n_list = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    naive_vars = []
    cv_vars = []

    for n in n_list:
        _, naive_var = mc_naive_call(S0, K, r, sigma, T, n, rng=rng)
        _, cv_var, _ = mc_control_variate_call(S0, K, r, sigma, T, n, rng=rng)

        naive_vars.append(naive_var)
        cv_vars.append(cv_var)

    plt.figure(figsize=(8, 5))
    plt.loglog(n_list, naive_vars, marker='o', label="Naive MC Variance")
    plt.loglog(n_list, cv_vars, marker='s', label="Control Variate Variance")
    plt.title("Variance Reduction: Naive vs Control Variate MC")
    plt.xlabel("Number of Paths (log scale)")
    plt.ylabel("Variance (log scale)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("variance_reduction_plot.png", dpi=200)
    print("Saved variance_reduction_plot.png")


#   Run All Tests + Generate Plots

def run_all_tests():
    test_monte_carlo_engines()
    test_implied_vol()
    plot_mc_convergence()
    plot_variance_reduction()


if __name__ == "__main__":
    run_all_tests()
