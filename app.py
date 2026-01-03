import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as sco
from scipy.stats import t
from arch import arch_model
from sklearn.mixture import GaussianMixture

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="QuantGod AI", layout="wide", page_icon="🚀")

# --- CSS STYLOVÁNÍ ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #303030;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- NASA MATH: KALMAN FILTER ---
class QuantKalman:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-3):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
        return self.posteri_estimate

    def smooth_series(self, series):
        estimates = []
        self.posteri_estimate = series[0]
        for x in series: estimates.append(self.update(x))
        return np.array(estimates)

def get_kalman_drift(prices):
    kf = QuantKalman(process_variance=1e-4, measurement_variance=0.1)
    smoothed = kf.smooth_series(prices.values)
    slope = (smoothed[-1] - smoothed[-5]) / smoothed[-5]
    return slope * (252 / 5)

# --- FRACTAL MATH: HURST ---
def get_hurst_exponent(time_series, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# --- DATA DOWNLOADER ---
@st.cache_data(ttl=3600)
def download_data(ticker, period="5y"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty: return pd.Series()
        if isinstance(df.columns, pd.MultiIndex):
            try: return df['Close'][ticker]
            except: return df.iloc[:, 0]
        return df['Close']
    except: return pd.Series()

# --- AI REGIME DETECTION ---
@st.cache_data(ttl=3600)
def get_macro_data():
    try:
        spy = yf.download("SPY", period="5y", progress=False)['Close']
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
        tnx = yf.download("^TNX", period="5d", progress=False)['Close'].iloc[-1].item()
        
        return ("PANIC 🔴" if is_panic else "CALM 🟢"), vix, (0.8 if is_panic else 0.0), tnx
    except: return "NEUTRAL", 20.0, 0.0, 4.0

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('targetMeanPrice', None)
    except: return None

# --- SIMULATION ENGINE ---
def run_garch_simulation(train_prices, horizon_days, n_sims, macro_vix, panic_prob, target_price=None):
    start_price = train_prices.iloc[-1]
    returns = 100 * np.log(1 + train_prices.pct_change().dropna())
    
    # GARCH Fit
    model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
    res = model.fit(disp='off')
    
    omega = res.params['omega']/10000; alpha = res.params['alpha[1]']; beta = res.params['beta[1]']; nu = res.params['nu']
    last_vol = res.conditional_volatility.iloc[-1]/100
    
    # Drift Logic (Kalman + Fractal + AI)
    hurst = get_hurst_exponent(train_prices.values[-100:])
    kalman_trend = get_kalman_drift(train_prices)
    long_term_drift = (returns.mean()/100)*252
    
    trust_in_trend = max(0, min(1, (hurst - 0.4) * 5))
    tech_drift = (trust_in_trend * kalman_trend) + ((1 - trust_in_trend) * long_term_drift)
    ai_penalty = -0.20 * panic_prob
    
    if target_price and target_price > 0:
        fund_drift = np.log(1 + (target_price - start_price)/start_price)
        base_drift = (0.4 * tech_drift) + (0.6 * fund_drift)
    else:
        base_drift = tech_drift
        
    final_drift_daily = (base_drift + ai_penalty) / 252
    
    # MC Loop
    paths = np.zeros((horizon_days, n_sims)); paths[0] = start_price
    sim_vol = np.zeros((horizon_days, n_sims)); sim_vol[0] = last_vol
    t_shocks = t.rvs(nu, size=(horizon_days, n_sims)) / np.sqrt(nu/(nu-2))
    lambda_j = 0.05 * (macro_vix/15)
    
    for i in range(1, horizon_days):
        prev_v = sim_vol[i-1]
        new_var = omega + alpha*((t_shocks[i-1]*prev_v)**2) + beta*(prev_v**2)
        curr_v = np.sqrt(new_var); sim_vol[i] = curr_v
        
        jump = np.random.normal(-0.1, 0.05, n_sims) * (np.random.random(n_sims) < lambda_j/252)
        paths[i] = paths[i-1] * np.exp( (final_drift_daily - 0.5*curr_v**2) + curr_v*t_shocks[i] + jump )
        
    return paths, hurst

# --- SIDEBAR UI ---
st.sidebar.image("https://img.icons8.com/color/96/000000/bullish.png", width=60)
st.sidebar.title("QuantGod AI")
mode = st.sidebar.selectbox("Vyber Režim", ["ANALYZER 🔮", "BACKTESTER 🧪", "OPTIMIZER ⚖️"])

ticker_input = "NVDA"
simulations = 2000
history_years = 5

if mode != "OPTIMIZER ⚖️":
    ticker_input = st.sidebar.text_input("Symbol Akcie", "NVDA").upper()
    simulations = st.sidebar.slider("Simulace", 1000, 5000, 3000)
    history_years = st.sidebar.slider("Historie (roky)", 2, 10, 5)
else:
    portfolio_input = st.sidebar.text_area("Portfolio (odděl čárkou)", "NVDA, MSFT, GOOGL, AMZN, KO, BTC-USD, GLD")

run_btn = st.sidebar.button("SPUSTIT VÝPOČET 🔥")

# --- HLAVNÍ LOGIKA ---
if run_btn:
    # 1. MAKRO DATA PRO VŠECHNY MODY
    regime, vix, panic_prob, tnx = get_macro_data()
    
    # --- MODE: ANALYZER ---
    if mode == "ANALYZER 🔮":
        st.subheader(f"🔮 Analýza Budoucnosti: {ticker_input}")
        
        with st.spinner("Stahuji data a počítám fraktály..."):
            data = download_data(ticker_input, f"{history_years}y")
            if data.empty: st.error("Chyba: Ticker nenalezen."); st.stop()
            
            target = get_fundamentals(ticker_input)
            paths, hurst = run_garch_simulation(data, 252, simulations, vix, panic_prob, target)
            
            # Výsledky
            start_p = paths[0,0]
            final_p = paths[-1]
            median_ret = ((np.median(final_p)-start_p)/start_p)*100
            var_95 = ((np.percentile(final_p, 5)-start_p)/start_p)*100
            
            # Kelly
            ret_sim = (final_p - start_p)/start_p
            prob_win = np.mean(ret_sim > 0)
            avg_win = np.mean(ret_sim[ret_sim>0]); avg_loss = abs(np.mean(ret_sim[ret_sim<0]))
            kelly = (prob_win - (1-prob_win)/(avg_win/avg_loss)) if avg_loss > 0 else 0
            safe_kelly = max(0, kelly * 0.5) * 100

            # UI Metriky
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cena", f"${start_p:.2f}")
            c2.metric("Režim Trhu", regime)
            c3.metric("Hurst Exp", f"{hurst:.2f}")
            c4.metric("VIX", f"{vix:.2f}")
            
            st.divider()
            k_col1, k_col2, k_col3 = st.columns(3)
            k_col1.metric("Očekávaný Výnos (1r)", f"{median_ret:.1f} %")
            k_col2.metric("Riziko (VaR 95)", f"{var_95:.1f} %", delta_color="inverse")
            k_col3.metric("💎 Kelly Alokace", f"{safe_kelly:.1f} %")
            
            # Grafy
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(paths[:, :100], color='gray', alpha=0.1)
            ax.plot(np.median(paths, axis=1), color='blue', lw=2, label='Medián')
            if target: ax.axhline(target, color='green', ls='--', label='Cíl Wall St.')
            ax.fill_between(range(252), np.percentile(paths, 5, axis=1), np.percentile(paths, 95, axis=1), color='blue', alpha=0.1)
            ax.set_title(f"GARCH-Fractal Projekce: {ticker_input}")
            ax.legend()
            st.pyplot(fig)

    # --- MODE: BACKTESTER ---
    elif mode == "BACKTESTER 🧪":
        st.subheader(f"🧪 Reality Check: {ticker_input}")
        
        with st.spinner("Cestuji v čase do minulosti..."):
            full_data = download_data(ticker_input, f"{history_years}y")
            if len(full_data) < 500: st.error("Málo dat."); st.stop()
            
            train = full_data[:-252]
            real = full_data[-252:]
            
            paths, _ = run_garch_simulation(train, 252, simulations, vix, panic_prob)
            
            # Porovnání
            fig, ax = plt.subplots(figsize=(10, 5))
            upper = np.percentile(paths, 95, axis=1)
            lower = np.percentile(paths, 5, axis=1)
            median = np.median(paths, axis=1)
            
            real_vals = real.values
            m_len = min(len(real_vals), len(median))
            
            ax.fill_between(range(m_len), lower[:m_len], upper[:m_len], color='blue', alpha=0.15, label='Model (90% Pásmo)')
            ax.plot(median[:m_len], color='blue', lw=2, label='Model Medián')
            ax.plot(real_vals[:m_len], color='red', lw=3, label='REALITA')
            ax.legend()
            st.pyplot(fig)
            
            # Skóre
            inside = np.sum((real_vals[:m_len] >= lower[:m_len]) & (real_vals[:m_len] <= upper[:m_len]))
            score = (inside / m_len) * 100
            
            if score > 80: st.success(f"✅ Model je ROBUSTNÍ. Skóre: {score:.1f}%")
            elif score > 50: st.warning(f"⚠️ Model je PŘIJATELNÝ. Skóre: {score:.1f}%")
            else: st.error(f"❌ Model SELHAL. Skóre: {score:.1f}%")

    # --- MODE: OPTIMIZER ---
    elif mode == "OPTIMIZER ⚖️":
        st.subheader("⚖️ Black-Litterman Portfolio Architect")
        
        tickers = [t.strip() for t in portfolio_input.split(',')]
        
        with st.spinner("Optimalizuji váhy aktiv..."):
            data = yf.download(tickers, period="2y", progress=False)['Close']
            if data.empty: st.error("Chyba stahování portfolia."); st.stop()
            
            # Market Caps
            caps = {}
            for t in tickers:
                try: caps[t] = yf.Ticker(t).info.get('marketCap', 1e9)
                except: caps[t] = 1e9
            mkt_w = np.array([caps[t]/sum(caps.values()) for t in tickers])
            
            # BL Math
            returns = np.log(data/data.shift(1)).dropna()
            cov = returns.cov() * 252
            pi = 2.5 * np.dot(cov, mkt_w)
            
            # Views (Kalman)
            my_views = []
            for t in tickers:
                col = data[t] if isinstance(data, pd.DataFrame) else data
                my_views.append(get_kalman_drift(col))
            my_views = np.array(my_views)
            
            tau = 0.01 if panic_prob > 0.5 else 0.05
            post_rets = (pi + (my_views * tau)) / (1 + tau)
            
            def neg_sharpe(w):
                p_ret = np.sum(post_rets * w)
                p_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                return -(p_ret - 0.04) / p_vol
            
            cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})
            bnds = tuple((0.05, 0.40) for _ in range(len(tickers)))
            
            res = sco.minimize(neg_sharpe, mkt_w, method='SLSQP', bounds=bnds, constraints=cons)
            
            # Výstup
            res_df = pd.DataFrame({
                "Ticker": tickers,
                "Alokace": [f"{x*100:.1f} %" for x in res.x],
                "Původní (Market)": [f"{x*100:.1f} %" for x in mkt_w]
            })
            st.table(res_df)
            
            exp_ret = np.sum(post_rets * res.x)*100
            exp_risk = np.sqrt(np.dot(res.x.T, np.dot(cov, res.x)))*100
            st.metric("Očekávaný Výnos Portfolia", f"{exp_ret:.2f} %")
            st.metric("Riziko Portfolia", f"{exp_risk:.2f} %")

else:
    st.info("👈 Zadej parametry v levém menu a klikni na SPUSTIT VÝPOČET.")
