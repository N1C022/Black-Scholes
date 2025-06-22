import numpy as np
from scipy.stats import norm

class BlackScholesCalculator:
    def __init__(self):
        # No need for initial parameters here as we'll pass them to each method
        pass

    def _d1_d2(self, S, K, T, r, sigma):
        # Helper method to calculate d1 and d2
        # Handles T=0 case: If T is 0, option value is intrinsic.
        # For d1/d2, if T is very small, this can cause issues.
        # It's usually better to check T > 0 before calling this.
        if T <= 0: # Avoid division by zero
            # These values aren't strictly meaningful if T=0 for d1/d2 calculation,
            # but setting them to handle potential edge case in the formula itself.
            # Realistically, if T<=0, your main pricing functions should return intrinsic value.
            # If S >= K and T=0, d1/d2 conceptually go to infinity. If S < K, to -infinity.
            # We'll rely on the main pricing functions to handle T<=0.
            return np.nan, np.nan # Not a number for invalid d1/d2 if T is non-positive

        # Ensure sigma is not zero to prevent division by zero
        if sigma <= 0:
            sigma = 1e-6 # Use a very small positive number to avoid div by zero

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def call_option_price(self, S, K, T, r, sigma):
        if T <= 0:
            return max(0, S - K) # Intrinsic value at expiration
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return price

    def put_option_price(self, S, K, T, r, sigma):
        if T <= 0:
            return max(0, K - S) # Intrinsic value at expiration
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price