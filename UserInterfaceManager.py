import streamlit as st
from BlackScholesCalculator import *

if 'current_asset_price' not in st.session_state:
    st.session_state.current_asset_price = True
if 'strike_price' not in st.session_state:
    st.session_state.strike_price = True
if 'time' not in st.session_state:
    st.session_state.time = True
if 'interest_rate' not in st.session_state:
    st.session_state.interest_rate = True
if 'vol' not in st.session_state:
    st.session_state.vol = True
if 'market_price' not in st.session_state:
    st.session_state.market_price = True

st.sidebar.title("Black Scholes Calculator")

def toggle_current_asset_price():
    st.session_state.current_asset_price = not st.session_state.current_asset_price

def toggle_strike_price():
    st.session_state.strike_price= not st.session_state.strike_price

def toggle_time():
    st.session_state.time = not st.session_state.time

def toggle_interest_rate():
    st.session_state.interest_rate = not st.session_state.interest_rate

def toggle_vol():
    st.session_state.vol = not st.session_state.vol

def toggle_market_price():
    st.session_state.market_price = not st.session_state.market_price

st.title("Black Scholes Calculator")

st.sidebar.button("Current Asset Price", on_click=toggle_current_asset_price)
current_asset_price = st.sidebar.number_input("Current Asset Price",label_visibility= 'collapsed',disabled=not st.session_state.current_asset_price, value = 100)

st.sidebar.button("Strike Price", on_click=toggle_strike_price)
strike_price = st.sidebar.number_input("Strike Price",label_visibility= 'collapsed',disabled=not st.session_state.strike_price, value = 100)

st.sidebar.button("Interest Rate", on_click=toggle_interest_rate)
interest_rate = st.sidebar.number_input("Interest Rate",label_visibility= 'collapsed',disabled=not st.session_state.interest_rate, value = 0.05)

st.sidebar.button("Time", on_click=toggle_time)
time = st.sidebar.number_input("Time",label_visibility= 'collapsed',disabled=not st.session_state.time, value = 1)

st.sidebar.button("Volatility", on_click=toggle_vol)
vol = st.sidebar.number_input("Volatility",label_visibility= 'collapsed',disabled=not st.session_state.vol, value = 0.2)

st.sidebar.button("Market Price", on_click=toggle_market_price)
market_price = st.sidebar.number_input("Market Price",label_visibility= 'collapsed',disabled=not st.session_state.market_price, value = 100)


Object = BlackScholesCalculator()
if st.session_state.vol:
    put_option_price = round(Object.put_option_price(current_asset_price, strike_price, time, interest_rate, vol), 2)
    call_option_price = round(Object.call_option_price(current_asset_price, strike_price, time, interest_rate, vol), 2)
    volatility = round(vol, 2)
else:
    put_option_price = "Market Price Given"
    call_option_price = "Market Price Given"
    volatility = ImpliedVolCalc(market_price, current_asset_price, strike_price, time, interest_rate)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
                f"<div style='text-align:center; font-size:48px; font-weight:bold'>{put_option_price}</div>",
                unsafe_allow_html=True,
            )
with col2:
    st.markdown(
                f"<div style='text-align:center; font-size:48px; font-weight:bold'>{call_option_price}</div>",
                unsafe_allow_html=True,
            )
    
with col3:
    st.markdown(
                f"<div style='text-align:center; font-size:48px; font-weight:bold'>{volatility}</div>",
                unsafe_allow_html=True,
            )