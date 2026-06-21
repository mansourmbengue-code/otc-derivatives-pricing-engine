import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# ============================================
# Modèles de pricing
# ============================================

def black_scholes_price(S, K, T, r, sigma, option_type='call', q=0.0):
    """Prix Black-Scholes pour option européenne."""
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def black_scholes_greeks(S, K, T, r, sigma, option_type='call', q=0.0):
    """Calcul des 5 Grecs."""
    if T <= 0:
        return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    # Delta
    if option_type == 'call':
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
    # Gamma (identique call/put)
    gamma = nd1 * np.exp(-q * T) / (S * sigma * np.sqrt(T))
    # Vega (par variation de 1% de vol)
    vega = S * np.exp(-q * T) * nd1 * np.sqrt(T) / 100
    # Theta (par jour)
    if option_type == 'call':
        theta = (- (S * np.exp(-q * T) * nd1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
    else:
        theta = (- (S * np.exp(-q * T) * nd1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
    # Rho (par 1% de taux)
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

# ============================================
# Interface Streamlit
# ============================================

st.set_page_config(page_title="OTC Pricing Engine", layout="wide")
st.title("⚡ Moteur de Pricing OTC – Options Européennes")
st.markdown("Paramètres du sous-jacent et de l'option pour calculer le prix et les Grecs.")

with st.sidebar:
    st.header("Paramètres de l'option")
    S = st.number_input("Spot (S)", value=100.0, step=1.0)
    K = st.number_input("Strike (K)", value=105.0, step=1.0)
    T = st.slider("Maturité (années)", 0.1, 5.0, 1.0, 0.1)
    r = st.slider("Taux sans risque (%)", 0.0, 5.0, 2.0, 0.1) / 100
    sigma = st.slider("Volatilité (%)", 5, 80, 20, 1) / 100
    option_type = st.selectbox("Type d'option", ["call", "put"])
    q = st.slider("Dividendes (%)", 0.0, 5.0, 0.0, 0.1) / 100

# Calcul du prix et des Grecs
price = black_scholes_price(S, K, T, r, sigma, option_type, q)
greeks = black_scholes_greeks(S, K, T, r, sigma, option_type, q)
intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
time_value = price - intrinsic

# Affichage des résultats
col1, col2, col3 = st.columns(3)
col1.metric("Prix théorique", f"{price:.4f}")
col2.metric("Valeur intrinsèque", f"{intrinsic:.4f}")
col3.metric("Valeur temps", f"{time_value:.4f}")

st.subheader("Grecs")
greeks_df = pd.DataFrame.from_dict(greeks, orient='index', columns=['Valeur'])
st.dataframe(greeks_df.style.format("{:.4f}"))

# Graphique Prix vs Spot
st.subheader("Sensibilité au spot")
spots = np.linspace(0.5 * S, 1.5 * S, 100)
prices_spot = [black_scholes_price(s, K, T, r, sigma, option_type, q) for s in spots]
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=spots, y=prices_spot, mode='lines', name='Prix'))
fig1.add_vline(x=S, line_dash="dash", line_color="red", annotation_text="Spot actuel")
fig1.update_layout(xaxis_title="Spot", yaxis_title="Prix de l'option")
st.plotly_chart(fig1, use_container_width=True)

# Graphique des Grecs en fonction du spot
st.subheader("Évolution des Grecs avec le spot")
greeks_names = ['delta', 'gamma', 'vega', 'theta', 'rho']
greeks_values = []
for g in greeks_names:
    vals = []
    for s in spots:
        gk = black_scholes_greeks(s, K, T, r, sigma, option_type, q)
        vals.append(gk[g])
    greeks_values.append(vals)

fig2 = go.Figure()
for i, name in enumerate(greeks_names):
    fig2.add_trace(go.Scatter(x=spots, y=greeks_values[i], mode='lines', name=name.capitalize()))
fig2.update_layout(xaxis_title="Spot", yaxis_title="Valeur du Greek")
st.plotly_chart(fig2, use_container_width=True)