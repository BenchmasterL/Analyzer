import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import t
from arch import arch_model
from sklearn.mixture import GaussianMixture

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="QuantGod AI", layout="wide")
st.title("🚀 QuantGod: AI Market Analyzer")

# --- SIDEBAR (OVLÁDÁNÍ) ---
st.sidebar.header("Nastavení")
ticker = st.sidebar.text_input("Zadej Ticker (např. NVDA, SPY)", value="NVDA")
simulations = st.sidebar.slider("Počet simulací", 1000, 5000, 2000)
years_history = st.sidebar.slider("Historie (roky)", 2, 10, 5)

# Tlačítko pro spuštění
start_button = st.sidebar.button("SPUSTIT ANALÝZU 🔥")

# --- POMOCNÉ FUNKCE (ZKRÁCENÉ PRO APP) ---
# (Zde je v podstatě stejná logika jako v tvém skriptu, jen upravená pro Streamlit)

@st.cache_data(ttl=3600) # Cache aby to bylo rychlé a nestahovalo data pořád
def download_data(ticker, years):
    df = yf.download(ticker, period=f"{years}y", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        try: return df['Close'][ticker]
        except: return df.iloc[:, 0]
    return df['Close']

def get_regime_and_macro():
    # Zjednodušená verze pro rychlost
    try:
        spy = yf.download("SPY", period="2y", progress=False)['Close']
        if isinstance(spy, pd.DataFrame): spy = spy.iloc[:,0]
        
        returns = np.log(spy / spy.shift(1)).dropna().values.reshape(-1, 1)
        vol = spy.pct_change().rolling(21).std().dropna().values.reshape(-1, 1)
        min_len = min(len(returns), len(vol))
        X = np.column_stack([returns[-min_len:], vol[-min_len:]])
        
        gmm = GaussianMixture(n_components=2, random_state=42).fit(X)
        curr_state = gmm.predict(X[-1].reshape(1,-1))[0]
        means = gmm.means_[:, 1]
        is_panic = (curr_state == np.argmax(means))
        
        vix = yf.download("^VIX", period="5d", progress=False)['Close'].iloc[-1].item()
        return ("PANIC 🔴" if is_panic else "CALM 🟢"), vix, (0.8 if is_panic else 0.0)
    except: return "NEUTRAL", 20.0, 0.0

def get_hurst(series):
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def run_simulation(prices, n_sims, prob_panic, vix):
    # Garch logic...
    returns = 100 * np.log(1 + prices.pct_change().dropna())
    model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
    res = model.fit(disp='off')
    
    # Parametry
    omega = res.params['omega']/10000; alpha = res.params['alpha[1]']; beta = res.params['beta[1]']; nu = res.params['nu']
    last_vol = res.conditional_volatility.iloc[-1]/100
    start_price = prices.iloc[-1]
    
    # Drift
    hurst = get_hurst(prices.values[-100:])
    tech_drift = (returns.mean()/100)*252
    ai_penalty = -0.20 * prob_panic
    final_drift = (tech_drift + ai_penalty) / 252
    
    # Sim
    days = 252
    paths = np.zeros((days, n_sims)); paths[0] = start_price
    sim_vol = np.zeros((days, n_sims)); sim_vol[0] = last_vol
    
    t_shocks = t.rvs(nu, size=(days, n_sims)) / np.sqrt(nu/(nu-2))
    lambda_j = 0.05 * (vix/15)
    
    for i in range(1, days):
        prev_v = sim_vol[i-1]
        new_var = omega + alpha*( (t_shocks[i-1]*prev_v)**2 ) + beta*(prev_v**2)
        curr_v = np.sqrt(new_var); sim_vol[i] = curr_v
        
        jump = np.random.normal(-0.1, 0.05, n_sims) * (np.random.random(n_sims) < lambda_j/252)
        paths[i] = paths[i-1] * np.exp( (final_drift - 0.5*curr_v**2) + curr_v*t_shocks[i] + jump )
        
    return paths, hurst, start_price

# --- HLAVNÍ LOGIKA ---
if start_button:
    with st.spinner(f'🤖 AI Analyzuje {ticker}... Prosím čekej.'):
        # 1. Stahování
        data = download_data(ticker, years_history)
        if data.empty:
            st.error("Chyba: Ticker nenalezen.")
            st.stop()
            
        # 2. Makro data
        regime, vix, panic_prob = get_regime_and_macro()
        
        # 3. Simulace
        paths, hurst, start_price = run_simulation(data, simulations, panic_prob, vix)
        
        # 4. Výsledky
        final_prices = paths[-1]
        median_ret = ((np.median(final_prices) - start_price) / start_price) * 100
        var_95 = ((np.percentile(final_prices, 5) - start_price) / start_price) * 100
        
        # Kelly calculation
        ret_sim = (final_prices - start_price)/start_price
        prob_win = np.mean(ret_sim > 0)
        avg_win = np.mean(ret_sim[ret_sim>0]); avg_loss = abs(np.mean(ret_sim[ret_sim<0]))
        kelly = (prob_win - (1-prob_win)/(avg_win/avg_loss)) if avg_loss > 0 else 0
        safe_kelly = max(0, kelly * 0.5) * 100

        # --- ZOBRAZENÍ NA DISPLEJI ---
        
        # Horní metriky
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Aktuální Cena", f"${start_price:.2f}")
        col2.metric("Režim Trhu (AI)", regime)
        col3.metric("Hurst Exponent", f"{hurst:.2f}", help=">0.5 = Trend, <0.5 = Chaos")
        col4.metric("VIX (Strach)", f"{vix:.2f}")

        st.divider()

        # Hlavní Výsledek - VELKÁ ČÍSLA
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"### 🎯 Očekávaný Výnos\n# {median_ret:.2f} %")
        c2.markdown(f"### ⚠️ Riziko (VaR 95)\n# {var_95:.2f} %")
        
        color = "green" if safe_kelly > 0 else "red"
        c3.markdown(f"### 💎 Kelly Alokace\n# :{color}[{safe_kelly:.2f} %]")

        # Grafy
        st.subheader("🔮 GARCH Monte Carlo Simulace (1 Rok)")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(paths[:, :100], color='gray', alpha=0.1)
        ax.plot(np.median(paths, axis=1), color='blue', lw=2, label='Medián')
        ax.fill_between(range(252), np.percentile(paths, 5, axis=1), np.percentile(paths, 95, axis=1), color='blue', alpha=0.1)
        ax.set_title(f"Projekce ceny: {ticker}")
        ax.legend()
        st.pyplot(fig)
        
        # Verdikt textem
        if safe_kelly > 10:
            st.success(f"**VERDIKT:** Silný signál. Matematická výhoda je na tvé straně. Doporučeno investovat až {safe_kelly:.1f}% kapitálu.")
        elif safe_kelly > 0:
            st.warning(f"**VERDIKT:** Opatrný nákup. Trh je rizikový, ale má potenciál. Investuj max {safe_kelly:.1f}%.")
        else:
            st.error("**VERDIKT:** NEKUPOVAT. Riziko převažuje nad ziskem.")
