import numpy as np
from scipy.stats import norm
#import streamlit as st
import yfinance


class Black_Scholes:
    def __init__(self, S, K, T, r, vol = None):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.vol = vol
        self. d1 = (np.log(S/K)+ (r + 0.5 * vol**2)*T)/(vol * (T**0.5))
        self.d2 = self.d1 - vol * (T**0.5)
    def call_option(self):
        if self.vol:
            c = self.S * norm.cdf(self.d1) - self.K * np.exp( -self.r * self.T) * norm.cdf(self.d2)
            return c
        return "Input a Volatility to use this function"

    def put_option(self):
        if self.vol:
            p = self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - S * norm.cdf(-self.d1)
            return p
        return "Input a Volatility to use this function"        
