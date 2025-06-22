import streamlit as st
import numpy as np
from BlackScholesCalculator import *

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Professional Options Calculator",
    page_icon="📈",
    layout="wide", # Use wide layout for more space
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    /* Overall App Container Padding */
    .st-emotion-cache-vk325g { /* This class might change with Streamlit updates, but generally targets the main app area */
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Main Title Styling */
    h1 {
        color: #007BFF; /* A nice blue */
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Subheaders */
    h2, h3, h4 {
        color: #333333;
        margin-top: 1.5em;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sidebar Header */
    .st-emotion-cache-6qob1r { /* Target sidebar header by internal class */
        color: #007BFF;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border-radius: 0.5rem; /* Slightly rounded corners */
        border: none;
        padding: 0.6rem 1.2rem;
        font-size: 1.1rem;
        transition: all 0.2s ease-in-out; /* Smooth hover effect */
        box-shadow: 0 2px 4px rgba(0,0,0,0.2); /* Subtle shadow */
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Darker blue on hover */
        transform: translateY(-2px); /* Slight lift on hover */
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Radio Button Layout (horizontal) */
    .stRadio > label > div {
        padding-right: 15px;
    }
    .stRadio div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap; /* Allow wrapping on smaller screens */
        gap: 15px; /* Space out radio options */
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
        color: #555;
    }
    .stTabs [data-baseweb="tab-list"] button.st-emotion-cache-1ft9m9p { /* Selected tab button - specific class might vary */
        color: #007BFF !important;
        border-bottom: 3px solid #007BFF !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #28a745; /* Success green for progress bar */
    }

    /* Info/Success/Warning/Error boxes */
    .stAlert {
        border-radius: 0.5rem;
    }

</style>
""", unsafe_allow_html=True)


# --- Header ---
st.markdown("<h1>📈 Professional Options Toolkit</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Calculate Black-Scholes prices, Greeks, and Implied Volatility.</p>", unsafe_allow_html=True)


# --- Main Content Area ---
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["📊 Option Pricing & Greeks", "🔍 Implied Volatility"])

bs_calc_instance = BlackScholesCalculator() # Creating an Instance of the Calculator Object

with tab1:
    st.header("Calculate Option Price & Greeks")
    st.markdown("Enter the parameters below to calculate the theoretical option price and its sensitivities.")

    # Layout in columns
    col_params_1, col_params_2 = st.columns(2)

    with col_params_1:
        S_price = st.number_input("Underlying Price (S)",
                                 value=100.0,
                                 min_value=0.01, format="%.2f",
                                 help="Current price of the underlying asset.")
        K_strike = st.number_input("Strike Price (K)",
                                   min_value=0.01, value=100.0, format="%.2f",
                                   help="The price at which the option can be exercised.")
        volatility_input = st.number_input("Volatility (σ, annual, e.g., 0.20 for 20%)",
                                           min_value=0.001, max_value=5.0, value=0.20, format="%.4f",
                                           help="The expected annual volatility of the underlying asset's returns. Must be positive.")

    with col_params_2:
        r_rate = st.number_input("Risk-Free Rate (r, annual, e.0, 0.05 for 5%)",
                                 min_value=0.0, max_value=0.20, value=0.05, format="%.4f",
                                 help="The annual risk-free interest rate, as a decimal. Often based on Treasury yields.")

        # Time to expiration input
        T_years_price = st.number_input("Time to Expiration (Years, e.g., 0.5)",
                                        min_value=0.0, value=0.5, format="%.4f",
                                        help="Time remaining until expiration, in years. Enter 0 for an option at expiration.")
        
        option_type_price = st.radio("Option Type", ('Call', 'Put'), horizontal=True,
                                    help="Specify whether you are pricing a Call or a Put option.")

    if st.button("Calculate Price & Greeks", key="calc_price_button"):
        if S_price <= 0 or K_strike <= 0 or r_rate < 0 or volatility_input <= 0:
            st.error("❗ Please ensure **Underlying Price (S)**, **Strike Price (K)**, **Risk-Free Rate (r)**, and **Volatility (σ)** are positive.")
        elif T_years_price < 0:
            st.error("❗ **Time to Expiration (T)** cannot be negative. Please enter a valid positive value or 0 for an option at expiration.")
        else:
            # Handle T=0 for intrinsic value, preventing division by zero in Greeks
            if T_years_price == 0:
                st.info("🕒 Option is at expiration (T = 0). Calculating intrinsic value only.")
                if option_type_price == 'Call':
                    calculated_price = max(0, S_price - K_strike)
                else:
                    calculated_price = max(0, K_strike - S_price)
                st.success(f"### Calculated Option Price: **${calculated_price:.4f}**")
                st.info("Greeks are typically not meaningful for options at their expiration.")
            else:
                # Perform actual Black-Scholes calculation
                if option_type_price == 'Call':
                    calculated_price = bs_calc_instance.call_option_price(S_price, K_strike, T_years_price, r_rate, volatility_input)
                else:
                    calculated_price = bs_calc_instance.put_option_price(S_price, K_strike, T_years_price, r_rate, volatility_input)
                
                st.success(f"### Calculated Option Price: **${calculated_price:.4f}**")
                st.markdown(f"*(Based on volatility: {volatility_input*100:.2f}%)*")
                
                st.markdown("---")
                st.subheader("Option Greeks")
                # Using st.columns for Greeks display asw
                greeks_col1, greeks_col2 = st.columns(2)
                with greeks_col1:
                    st.write(f"**Delta:** `{bs_calc_instance.delta(S_price, K_strike, T_years_price, r_rate, volatility_input, option_type_price.lower()):.4f}`")
                    st.write(f"**Gamma:** `{bs_calc_instance.gamma(S_price, K_strike, T_years_price, r_rate, volatility_input):.4f}`")
                    st.write(f"**Vega:** `{bs_calc_instance.vega(S_price, K_strike, T_years_price, r_rate, volatility_input):.4f}` (per 1% vol change)")
                with greeks_col2:
                    st.write(f"**Theta:** `{bs_calc_instance.theta(S_price, K_strike, T_years_price, r_rate, volatility_input, option_type_price.lower()):.4f}` (per day)")
                    st.write(f"**Rho:** `{bs_calc_instance.rho(S_price, K_strike, T_years_price, r_rate, volatility_input, option_type_price.lower()):.4f}` (per 1% rate change)")

with tab2:
    st.header("Calculate Implied Volatility")
    st.markdown("Enter the market price of an option along with its parameters to derive its implied volatility.")

    col_iv_params_1, col_iv_params_2 = st.columns(2)

    with col_iv_params_1:
        S_iv = st.number_input("Underlying Price (S)",
                               value=100.0,
                               min_value=0.01, format="%.2f", key="S_iv",
                               help="Current price of the underlying asset.")
        K_iv = st.number_input("Strike Price (K)",
                               min_value=0.01, value=100.0, format="%.2f", key="K_iv",
                               help="The strike price of the option.")

    with col_iv_params_2:
        r_iv = st.number_input("Risk-Free Rate (r, annual, e.g., 0.05 for 5%)",
                               min_value=0.0, max_value=0.20, value=0.05, format="%.4f", key="r_iv",
                               help="The annual risk-free interest rate, as a decimal. Often based on Treasury yields.")
        
        T_years_iv = st.number_input("Time to Expiration (Years, e.g., 0.5)",
                                        min_value=0.000001, value=0.5, format="%.4f", key="T_iv",
                                        help="Time remaining until expiration, in years. Must be strictly positive (> 0) for Implied Volatility calculation.")

        option_type_iv = st.radio("Option Type", ('Call', 'Put'), horizontal=True, key="type_iv",
                                 help="Specify whether you are calculating IV for a Call or a Put option.")

    st.markdown("---")
    st.subheader("Market Option Price Input")

    market_price_for_iv = st.number_input("Enter Market Option Price", min_value=0.01, value=5.0, format="%.2f",
                                          help="The actual price of the option observed in the market.")
    
    if st.button("Calculate Implied Volatility", key="calc_iv_button"):
        if S_iv <= 0 or K_iv <= 0 or r_iv < 0:
            st.error("❗ Please ensure **Underlying Price (S)**, **Strike Price (K)**, and **Risk-Free Rate (r)** are positive.")
            st.stop()
        if T_years_iv <= 0: #necessary check
            st.error("❗ **Time to Expiration (T)** must be strictly greater than 0 to calculate Implied Volatility.")
            st.stop()
        if market_price_for_iv <= 0:
            st.error("❗ Please input a valid **Market Option Price** for implied volatility calculation.")
            st.stop()
        
        # Prevent Impossible IVs
        intrinsic_value = max(0, S_iv - K_iv) if option_type_iv == 'Call' else max(0, K_iv - S_iv)
        if market_price_for_iv < intrinsic_value:
            st.error(f"❗ Market price **${market_price_for_iv:.2f}** is less than the option's intrinsic value (**${intrinsic_value:.2f}**). Cannot calculate IV.")
            st.stop()
        max_price_theoretical = S_iv if option_type_iv == 'Call' else K_iv
        if market_price_for_iv > max_price_theoretical:
             st.error(f"❗ Market price **${market_price_for_iv:.2f}** is greater than theoretical max price (**${max_price_theoretical:.2f}**). Cannot calculate IV.")
             st.stop()

        try:
            with st.spinner('Calculating Implied Volatility... this may take a moment.'):
                implied_vol = ImpliedVolCalc(
                    market_price_for_iv, S_iv, K_iv, T_years_iv, r_iv, option_type=option_type_iv.lower()
                )
            
            if not np.isnan(implied_vol) and implied_vol > 0.001 and implied_vol < 5.0:
                st.success(f"### Implied Volatility: **{implied_vol*100:.2f}%**")
                
                # Verify price
                if option_type_iv == 'Call':
                    verified_price = bs_calc_instance.call_option_price(S_iv, K_iv, T_years_iv, r_iv, implied_vol)
                else:
                    verified_price = bs_calc_instance.put_option_price(S_iv, K_iv, T_years_iv, r_iv, implied_vol)
                st.info(f"BS Price with Implied Vol: ${verified_price:.4f} (should be very close to market price: ${market_price_for_iv:.4f})")
                
                st.markdown("---")
                st.subheader("Option Greeks (at Implied Volatility)")

                greeks_iv_col1, greeks_iv_col2 = st.columns(2)
                with greeks_iv_col1:
                    st.write(f"**Delta:** `{bs_calc_instance.delta(S_iv, K_iv, T_years_iv, r_iv, implied_vol, option_type_iv.lower()):.4f}`")
                    st.write(f"**Gamma:** `{bs_calc_instance.gamma(S_iv, K_iv, T_years_iv, r_iv, implied_vol):.4f}`")
                    st.write(f"**Vega:** `{bs_calc_instance.vega(S_iv, K_iv, T_years_iv, r_iv, implied_vol):.4f}` (per 1% vol change)")
                with greeks_iv_col2:
                    st.write(f"**Theta:** `{bs_calc_instance.theta(S_iv, K_iv, T_years_iv, r_iv, implied_vol, option_type_iv.lower()):.4f}` (per day)")
                    st.write(f"**Rho:** `{bs_calc_instance.rho(S_iv, K_iv, T_years_iv, r_iv, implied_vol, option_type_iv.lower()):.4f}` (per 1% rate change)")

            else:
                st.error("❌ Could not calculate a valid implied volatility. The market price might be impossible for the given parameters, or the solver did not converge. Try adjusting market price or input parameters.")
        except Exception as e:
            st.error(f"An unexpected error occurred during implied volatility calculation: {e}. Please check your inputs.")
