import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import brentq
from io import BytesIO
from mpl_toolkits.mplot3d import Axes3D
import yfinance as yf

# Custom dark styling
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #1e1e1e;
        color: white;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        background-color: #00ffcc;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# Session state for reset
if 'reset' not in st.session_state:
    st.session_state.reset = False

if st.button("🔄 Reset Inputs"):
    st.session_state.reset = True

# Beginner Guide
with st.expander("🧭 How to Use this Dashboard (Beginner Guide)", expanded=False):
    st.markdown("""
    Welcome to the **Black-Scholes Quant Dashboard**! Here's how to use it:

    - **Spot Price (S)**: The current market price of the asset.
    - **Strike Price (K)**: The agreed price for buying/selling the asset.
    - **Risk-Free Rate (r)**: Interest rate assumed for risk-free investing.
    - **Volatility (σ)**: Expected fluctuation in the asset price.
    - **Time to Maturity (T)**: How long until the option expires.
    - Use the toggle to switch between *Annualized* and *Daily Compounded* interest.
    """)

# Input
st.title("📊 Black-Scholes Quant Dashboard")

S = st.number_input("📌 Spot Price (S)", value=100.0 if not st.session_state.reset else 0.0, help="Current market price of the asset")
K = st.number_input("📌 Strike Price (K)", value=100.0 if not st.session_state.reset else 0.0, help="Agreed price to buy/sell asset")
sigma = st.number_input("📌 Volatility (σ, annualized)", value=0.2 if not st.session_state.reset else 0.0, help="Expected standard deviation in returns")

compounding_mode = st.radio("📈 Interest Rate Type", ["Annualized (simple)", "Daily Compounded"], index=0, help="Choose interest rate calculation method")
r_input = st.number_input("📌 Risk-Free Rate (%)", value=5.0 if not st.session_state.reset else 0.0, step=0.1, help="Interest rate for discounting") / 100
r = math.log(1 + r_input) if compounding_mode == "Daily Compounded" else r_input

col_tm1, col_tm2 = st.columns(2)
years = col_tm1.number_input("📅 Years to Maturity", value=0 if st.session_state.reset else 1, step=1, help="Years until option expires")
months = col_tm2.number_input("📅 Additional Months", value=0 if st.session_state.reset else 0, step=1, help="Extra months beyond full years")
T = years + months / 12.0
st.info(f"⏳ Time to Maturity: {T:.2f} years ({years} year(s), {months} month(s))")

# Model Functions
def black_scholes_call_put(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call, put, d1, d2

def option_greeks(S, K, T, r, sigma, d1, d2):
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * norm.pdf(d1) * math.sqrt(T)
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))) - r * K * math.exp(-r * T) * norm.cdf(d2)
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))) + r * K * math.exp(-r * T) * norm.cdf(-d2)
    rho_call = K * T * math.exp(-r * T) * norm.cdf(d2)
    rho_put = -K * T * math.exp(-r * T) * norm.cdf(-d2)
    return delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put

def implied_volatility(option_price, S, K, T, r, option_type='call'):
    def obj_func(sigma):
        call, put, _, _ = black_scholes_call_put(S, K, T, r, sigma)
        return (call - option_price) if option_type == 'call' else (put - option_price)
    try:
        return brentq(obj_func, 1e-6, 5.0)
    except ValueError:
        return None

def generate_excel(data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False)
    output.seek(0)
    return output


# Forward price input (dividend-adjusted) - shown before Run Analysis button
# Removed duplicate: Forward Price section
dividend_yield = st.number_input("Dividend Yield (%)", value=0.0, step=0.1, help="Annual dividend yield as a percentage") / 100
# Removed duplicate calculation
# Removed duplicate display


col_iv1, col_iv2 = st.columns(2)
market_price = col_iv1.number_input("Market Option Price", value=100.0, help="Enter market price of the option")
option_type = col_iv2.radio("Option Type", ["Call", "Put"], horizontal=True)
iv = implied_volatility(market_price, S, K, T, r, option_type.lower())
if iv:
    st.success(f"🧠 Implied Volatility: {iv:.2%}")
else:
    st.warning("⚠️ Could not compute implied volatility.")

if st.button("Run Analysis"):
    call, put, d1, d2 = black_scholes_call_put(S, K, T, r, sigma)
    delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put = option_greeks(S, K, T, r, sigma, d1, d2)

    col1, col2 = st.columns(2)
    col1.metric("📈 Call Price", f"R {call:.2f}")
    col2.metric("📉 Put Price", f"R {put:.2f}")

    st.subheader("🧠 Greeks (Sensitivity Measures)")
    st.dataframe(pd.DataFrame({
        "Greek": ["Delta (Call)", "Delta (Put)", "Gamma", "Vega", "Theta (Call)", "Theta (Put)", "Rho (Call)", "Rho (Put)"],
        "Value": [delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put]
    }))

    st.subheader("🔄 Put-Call Parity Check")
    parity_error = abs(call - (put + S - K * np.exp(-r * T)))
    if parity_error > 0.01:
        st.error(f"⚠️ Parity violation! Arbitrage gap: {parity_error:.4f}")
    else:
        st.success("✅ Put-call parity holds.")

    # Removed duplicate: Forward Price section
    # Removed duplicate input
    # Removed duplicate calculation
    # Removed duplicate display

    st.subheader("📡 Yahoo Finance Live Price")
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    try:
        spot_live = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
        st.success(f"📊 {ticker.upper()} Live Price: {spot_live:.2f}")
    except:
        st.warning("⚠️ Could not fetch ticker data.")

    st.subheader("🧭 3D Surface Plots")
    strike_vals = np.linspace(0.5 * S, 1.5 * S, 20)
    vol_vals = np.linspace(0.05, 1.0, 20)
    X, Y = np.meshgrid(strike_vals, vol_vals)
    Zc = np.array([[black_scholes_call_put(S, k, T, r, v)[0] for k in strike_vals] for v in vol_vals])
    Zp = np.array([[black_scholes_call_put(S, k, T, r, v)[1] for k in strike_vals] for v in vol_vals])

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, Zc, cmap='viridis')
    ax.set_title('Call Surface')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Volatility')
    ax.set_zlabel('Price')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Zp, cmap='plasma')
    ax2.set_title('Put Surface')
    ax2.set_xlabel('Strike')
    ax2.set_ylabel('Volatility')
    ax2.set_zlabel('Price')

    st.pyplot(fig)
    st.caption("📘 These surfaces help visualize how option prices change with volatility and strike levels.")

    st.subheader("📤 Export Results to Excel")
    df_export = pd.DataFrame({
        "Call": [call], "Put": [put],
        "Delta Call": [delta_call], "Delta Put": [delta_put],
        "Gamma": [gamma], "Vega": [vega],
        "Theta Call": [theta_call], "Theta Put": [theta_put],
        "Rho Call": [rho_call], "Rho Put": [rho_put]
    })
    st.download_button("💾 Download .xlsx", generate_excel(df_export), file_name="black_scholes_results.xlsx")
# --- Additional Advanced Features ---
    st.subheader("📉 Profit & Loss Profile with Breakeven Point")
    prices = np.linspace(0.5 * S, 1.5 * S, 100)
    call_pnl = np.maximum(prices - K, 0) - call
    put_pnl = np.maximum(K - prices, 0) - put

    fig_pnl, ax_pnl = plt.subplots()
    ax_pnl.plot(prices, call_pnl, label='Call P&L', color='green')
    ax_pnl.plot(prices, put_pnl, label='Put P&L', color='red')
    ax_pnl.axhline(0, linestyle='--', color='gray')
    ax_pnl.set_xlabel("Underlying Price at Expiry")
    ax_pnl.set_ylabel("Profit / Loss")
    ax_pnl.set_title("Option P&L at Expiry")
    ax_pnl.legend()
    st.pyplot(fig_pnl)
    st.caption("🔍 This chart shows how your Call and Put positions perform at expiry depending on the underlying asset price.")

    st.subheader("🛠️ Strategy Builder: Straddle & Bull Call Spread")
    straddle = np.maximum(prices - K, 0) + np.maximum(K - prices, 0) - (call + put)
    spread = np.maximum(prices - K, 0) - np.maximum(prices - (K + 10), 0) - (call - black_scholes_call_put(S, K+10, T, r, sigma)[0])

    fig_strategy, ax_strategy = plt.subplots()
    ax_strategy.plot(prices, straddle, label='Straddle', color='purple')
    ax_strategy.plot(prices, spread, label='Bull Call Spread', color='blue')
    ax_strategy.axhline(0, linestyle='--', color='gray')
    ax_strategy.set_xlabel("Underlying Price at Expiry")
    ax_strategy.set_ylabel("Profit / Loss")
    ax_strategy.set_title("Strategy Payoffs")
    ax_strategy.legend()
    st.pyplot(fig_strategy)
    st.caption("📘 Strategy Payoffs: The straddle profits from large moves in either direction, while the bull spread profits from moderate upside.")

    st.subheader("🎲 Monte Carlo P&L Simulation")
    def monte_carlo_pnl(S, K, T, r, sigma, option_type='call', sims=10000):
        Z = np.random.normal(0, 1, sims)
        ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * Z)
        if option_type == 'call':
            payoff = np.maximum(ST - K, 0)
        else:
            payoff = np.maximum(K - ST, 0)
        pnl = payoff - np.mean(payoff)
        return pnl

    pnl_sim = monte_carlo_pnl(S, K, T, r, sigma, option_type.lower())
    fig_mc, ax_mc = plt.subplots()
    ax_mc.hist(pnl_sim, bins=50, color='skyblue', edgecolor='black')
    ax_mc.set_title("Monte Carlo P&L Distribution")
    ax_mc.set_xlabel("Profit / Loss")
    ax_mc.set_ylabel("Frequency")
    st.pyplot(fig_mc)
    st.caption("🎯 This histogram shows simulated P&L outcomes under 10,000 price paths.")

    st.subheader("🔳 Heatmaps for Call/Put Prices vs Strike & Volatility")
    strike_range = np.linspace(0.5 * S, 1.5 * S, 20)
    vol_range = np.linspace(0.05, 1.0, 20)
    call_matrix = np.zeros((len(vol_range), len(strike_range)))
    put_matrix = np.zeros((len(vol_range), len(strike_range)))

    for i, vol in enumerate(vol_range):
        for j, strike in enumerate(strike_range):
            c, p, _, _ = black_scholes_call_put(S, strike, T, r, vol)
            call_matrix[i, j] = c
            put_matrix[i, j] = p

    fig_heat, ax_heat = plt.subplots(figsize=(10, 5))
    sns.heatmap(call_matrix, xticklabels=np.round(strike_range, 1), yticklabels=np.round(vol_range, 2), ax=ax_heat, cmap='YlGnBu')
    ax_heat.set_title("Call Option Price Heatmap")
    st.pyplot(fig_heat)
    st.caption("🟦 Heatmap of Call prices for combinations of strike price and volatility.")

    fig_heat2, ax_heat2 = plt.subplots(figsize=(10, 5))
    sns.heatmap(put_matrix, xticklabels=np.round(strike_range, 1), yticklabels=np.round(vol_range, 2), ax=ax_heat2, cmap='YlOrRd')
    ax_heat2.set_title("Put Option Price Heatmap")
    st.pyplot(fig_heat2)
    st.caption("🟥 Heatmap of Put prices under varying strike and volatility levels.")