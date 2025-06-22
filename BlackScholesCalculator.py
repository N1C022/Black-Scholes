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
    
    # --- Greeks ---
    # Delta: Sensitivity of option price to a $1 change in underlying price
    def delta(self, S, K, T, r, sigma, option_type='call'):
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0 # At expiry, ITM call delta is 1, OTM is 0
            else: # put
                return -1.0 if S < K else 0.0 # At expiry, ITM put delta is -1, OTM is 0
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        if option_type == 'call':
            return norm.cdf(d1)
        else: # put
            return norm.cdf(d1) - 1

    # Gamma: Rate of change of Delta with respect to a change in underlying price
    def gamma(self, S, K, T, r, sigma):
        if T <= 0: return 0.0 # Gamma approaches zero at expiry
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        # norm.pdf is the standard normal probability density function (N'(d1))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega: Sensitivity of option price to a 1% change in implied volatility
    def vega(self, S, K, T, r, sigma):
        if T <= 0: return 0.0 # Vega approaches zero at expiry
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        # Vega is typically given per 1% change in volatility, so divide by 100 for the final output
        return (S * np.sqrt(T) * norm.pdf(d1)) / 100 

    # Theta: Sensitivity of option price to the passage of time (time decay)
    # Often expressed per day, so divide by 365
    def theta(self, S, K, T, r, sigma, option_type='call'):
        if T <= 0: return 0.0 # Theta approaches zero at expiry
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        term1 = (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        term2_call = r * K * np.exp(-r * T) * norm.cdf(d2)
        term2_put = r * K * np.exp(-r * T) * norm.cdf(-d2)

        if option_type == 'call':
            theta_val = -term1 - term2_call
        else: # put
            theta_val = -term1 + term2_put
        
        return theta_val / 365 # Per day

    # Rho: Sensitivity of option price to a 1% change in the risk-free rate
    def rho(self, S, K, T, r, sigma, option_type='call'):
        if T <= 0: return 0.0 # Rho approaches zero at expiry
        
        _, d2 = self._d1_d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            return (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100 # Per 1% rate change
        else: # put
            return (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100 # Per 1% rate change
        
def implied_volatility_newton_raphson(market_price, S, K, T, r, option_type='call', initial_sigma=0.5, tolerance=1e-6, max_iterations=100):

    bs_calc = BlackScholesCalculator() # Create an instance of the calculator

    sigma = initial_sigma
    for i in range(max_iterations):
        if option_type == 'call':
            bs_price = bs_calc.call_option_price(S, K, T, r, sigma)
        else: # put
            bs_price = bs_calc.put_option_price(S, K, T, r, sigma)

        price_difference = bs_price - market_price
        
        if abs(price_difference) < tolerance:
            return sigma
        
        # Get Vega for the current sigma. Vega is the derivative of option price w.r.t. volatility.
        # Our vega method returns per 1%, so multiply by 100 to get the direct derivative.
        v = bs_calc.vega(S, K, T, r, sigma) * 100 # Adjust vega to be the direct derivative

        # Handle cases where vega is too small (might cause division by zero or large steps)
        if abs(v) < 1e-10: 
            # If vega is tiny, either we are very close to expiry or OTM, or at an extreme point.
            # Attempt a small step or return current sigma
            sigma += 0.01 if price_difference > 0 else -0.01 # Small step
            if sigma <= 0: sigma = 0.001 # Ensure positive
            continue

        # Newton-Raphson step: sigma_new = sigma - f(sigma) / f'(sigma)
        # where f(sigma) = bs_price - market_price and f'(sigma) = vega
        sigma_new = sigma - price_difference / v
        
        # Ensure sigma remains positive and within reasonable bounds
        if sigma_new <= 0: 
            sigma_new = 0.001 # Set to a small positive value
        elif sigma_new > 5.0: # Cap at 500% to prevent unrealistic values
            sigma_new = 5.0 
            
        sigma = sigma_new
    
    # If max_iterations reached without convergence, return last estimate
    # In a real-world app, you might want to return np.nan or raise an error.
    return sigma