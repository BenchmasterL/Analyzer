import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as sco
from scipy.stats import t
from arch import arch_model
from sklearn.mixture import GaussianMixture

# --- KONFIGURACE APLIKACE ---
st.set_page_config(page_title="QuantGod AI", layout="wide", page_icon="🧠")

# --- CSS STYLOVÁNÍ (Profi vzhled) ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
        margin-bottom: 10px;
    }
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    .stAlert {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 🧠 JÁDRO: MATEMATICKÉ MODELY
# ==============================================================================

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

def get_hurst_exponent(time_series, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

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
        return ("PANIC 🔴" if is_panic else "CALM 🟢"), vix, (0.8 if is_panic else 0.0)
    except: return "NEUTRAL", 20.0, 0.0

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('targetMeanPrice', None)
    except: return None

def run_garch_simulation(train_prices, horizon_days, n_sims, macro_vix, panic_prob, target_price=None):
    start_price = train_prices.iloc[-1]
    returns = 100 * np.log(1 + train_prices.pct_change().dropna())
    
    model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
    res = model.fit(disp='off')
    
    omega = res.params['omega']/10000; alpha = res.params['alpha[1]']; beta = res.params['beta[1]']; nu = res.params['nu']
    last_vol = res.conditional_volatility.iloc[-1]/100
    
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

# ==============================================================================
# 📱 GRAFICKÉ ROZHRANÍ (STREAMLIT)
# ==============================================================================

# 1. Sidebar Menu
st.sidebar.title("QuantGod AI 🧠")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Vyber Režim:", ["ANALYZER (Poradce)", "BACKTESTER (Ověření)", "OPTIMIZER (Portfolio)"])

if mode != "OPTIMIZER (Portfolio)":
    ticker_input = st.sidebar.text_input("Zadej Ticker", "NVDA").upper()
    
    st.sidebar.markdown("### ⚙️ Nastavení Predikce")
    target_day = st.sidebar.slider("📅 Cíl za X dní", 5, 252, 30)
    simulations = st.sidebar.slider("Počet simulací", 1000, 5000, 3000)
else:
    portfolio_input = st.sidebar.text_area("Portfolio (odděl čárkou)", "NVDA, MSFT, GOOGL, AMZN, KO, BTC-USD, GLD")

start_btn = st.sidebar.button("SPUSTIT ANALÝZU 🔥", type="primary")

# 2. Logika Aplikace
if start_btn:
    regime, vix, panic_prob = get_macro_data()
    
    # --- ANALYZER (PORADCE) ---
    if mode == "ANALYZER (Poradce)":
        st.header(f"🔮 QuantGod Poradce: {ticker_input}")
        
        with st.spinner(f"AI analyzuje {ticker_input}, počítá fraktály a simuluje trh..."):
            data = download_data(ticker_input)
            if data.empty: st.error("Ticker nenalezen."); st.stop()
            
            target = get_fundamentals(ticker_input)
            paths, hurst = run_garch_simulation(data, 252, simulations, vix, panic_prob, target)
            
            # Výpočty
            start_p = paths[0,0]
            day_idx = min(target_day, 251)
            
            # Data pro Den X
            prices_at_x = paths[day_idx]
            median_x = np.median(prices_at_x)
            roi_x = ((median_x - start_p)/start_p)*100
            
            # Data pro Konec roku (Risk management)
            final_p = paths[-1]
            var_95 = ((np.percentile(final_p, 5)-start_p)/start_p)*100
            
            # --- NOVINKA: Expected Shortfall (CVaR) ---
            worst_5_percent = final_p[final_p <= np.percentile(final_p, 5)]
            cvar_95 = ((np.mean(worst_5_percent) - start_p) / start_p) * 100
            
            # Kellyho Kritérium (Srdce modelu)
            ret_sim = (final_p - start_p)/start_p
            prob_win = np.mean(ret_sim > 0)
            avg_win = np.mean(ret_sim[ret_sim>0]); avg_loss = abs(np.mean(ret_sim[ret_sim<0]))
            kelly = (prob_win - (1-prob_win)/(avg_win/avg_loss)) if avg_loss > 0 else 0
            safe_kelly = max(0, kelly * 0.5) * 100

            # Fundamentální Analýza (Wall St.)
            wall_street_upside = 0
            if target:
                wall_street_upside = ((target - start_p) / start_p) * 100

            # --- VYHODNOCENÍ SIGNÁLU (SEMAFOR) ---
            signal_text = "NEKUPOVAT ⛔"
            signal_color = "red"
            explanation = "Riziko převažuje nad potenciálním ziskem. Model doporučuje vyčkat."
            
            # Logika rozhodování
            if safe_kelly > 15 and wall_street_upside > 0:
                signal_text = "SILNÝ NÁKUP (STRONG BUY) 🚀"
                signal_color = "green"
                explanation = "Matematika (Model) i Analytici (Wall St.) se shodují na růstu."
            elif safe_kelly > 5:
                signal_text = "AKUMULOVAT (BUY) ✅"
                signal_color = "#90EE90" # Light green
                explanation = "Technický model vidí růst, ale buď opatrný s velikostí pozice."
            elif safe_kelly == 0:
                signal_text = "VYHNOUT SE (AVOID) ❌"
                signal_color = "red"
                explanation = "Statistická pravděpodobnost ztráty je příliš vysoká."

            # --- DASHBOARD (ZOBRAZENÍ) ---
            
            # 1. Hlavní Ortel
            st.markdown(f"""
            <div style="background-color: #262730; padding: 20px; border-radius: 10px; border-left: 10px solid {signal_color};">
                <h2 style="margin:0; color: {signal_color};">{signal_text}</h2>
                <p style="font-size: 18px; margin-top: 5px;">{explanation}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()

            # 2. Klíčová Data
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Aktuální Cena", f"${start_p:.2f}")
            
            val_color = "normal"
            if wall_street_upside > 5: val_color = "normal" # Green default in dark mode
            elif wall_street_upside < 0: val_color = "inverse" # Red
            
            c2.metric("Potenciál (Wall St.)", f"{wall_street_upside:+.1f} %", help="O kolik je akcie levnější než cílová cena analytiků", delta_color=val_color)
            c3.metric("Kelly Alokace (Max sázka)", f"{safe_kelly:.1f} %", help="Kolik % portfolia do toho maximálně vložit")
            c4.metric("Pravděpodobnost Zisku", f"{prob_win*100:.1f} %", help="Šance, že obchod skončí v plusu (Win Rate)")

            # 3. Predikce na Den X
            st.subheader(f"📅 Predikce ceny za {target_day} dní")
            col_target1, col_target2, col_target3 = st.columns(3)
            
            roi_color = "normal" if roi_x > 0 else "inverse"
            col_target1.metric("Očekávaná Cena (Model)", f"${median_x:.2f}", f"{roi_x:+.2f} %", delta_color=roi_color)
            col_target2.metric("Riziko poklesu (VaR 95%)", f"{var_95:.1f} %", help="Maximální ztráta v 95% případů")
            col_target3.metric("Extrémní Riziko (CVaR)", f"{cvar_95:.1f} %", help="Průměrná ztráta při katastrofě (černá labuť)")

            # 4. Grafy
            tab1, tab2 = st.tabs(["📈 Vývoj Ceny (Simulace)", "📊 Rozdělení Pravděpodobnosti"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(paths[:, :100], color='gray', alpha=0.1)
                ax.plot(np.median(paths, axis=1), color='blue', lw=2, label='Medián (Očekávání)')
                ax.axvline(day_idx, color='magenta', ls='--', lw=2, label=f'Cíl ({target_day} dní)')
                if target: ax.axhline(target, color='green', ls='--', label='Cíl Wall St.')
                ax.fill_between(range(252), np.percentile(paths, 5, axis=1), np.percentile(paths, 95, axis=1), color='blue', alpha=0.1)
                ax.legend()
                ax.set_title(f"Simulace vývoje {ticker_input}")
                st.pyplot(fig)
            
            with tab2:
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.hist(prices_at_x, bins=50, density=True, color='#800080', alpha=0.7, edgecolor='black')
                ax2.axvline(start_p, color='orange', ls='--', lw=2, label='Start')
                ax2.axvline(median_x, color='cyan', ls='-', lw=2, label='Medián')
                ax2.legend()
                ax2.set_title(f"Kde bude cena v den {target_day}?")
                st.pyplot(fig2)

    # --- BACKTESTER ---
    elif mode == "BACKTESTER (Ověření)":
        st.header(f"🧪 Reality Check: {ticker_input}")
        with st.spinner("Cestuji v čase do minulosti..."):
            full_data = download_data(ticker_input, "5y")
            if len(full_data) < 500: st.error("Málo dat."); st.stop()
            
            train = full_data[:-252]
            real = full_data[-252:]
            
            paths, _ = run_garch_simulation(train, 252, simulations, vix, panic_prob)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            upper = np.percentile(paths, 95, axis=1)
            lower = np.percentile(paths, 5, axis=1)
            median = np.median(paths, axis=1)
            real_vals = real.values
            m_len = min(len(real_vals), len(median))
            
            ax.fill_between(range(m_len), lower[:m_len], upper[:m_len], color='blue', alpha=0.15, label='90% Pásmo spolehlivosti')
            ax.plot(median[:m_len], color='blue', label='Predikce Modelu')
            ax.plot(real_vals[:m_len], color='red', lw=2, label='SKUTEČNOST')
            ax.legend()
            st.pyplot(fig)
            
            inside = np.sum((real_vals[:m_len] >= lower[:m_len]) & (real_vals[:m_len] <= upper[:m_len]))
            score = (inside / m_len) * 100
            
            if score > 80: st.success(f"✅ Model funguje skvěle! (Přesnost: {score:.1f}%)")
            else: st.warning(f"⚠️ Model je pro tuto akcii méně přesný. (Skóre: {score:.1f}%)")

    # --- OPTIMIZER ---
    elif mode == "OPTIMIZER (Portfolio)":
        st.header("⚖️ Portfolio Architect")
        tickers = [t.strip() for t in portfolio_input.split(',')]
        
        with st.spinner("Hledám ideální poměr..."):
            data = yf.download(tickers, period="2y", progress=False)['Close']
            if data.empty: st.error("Chyba dat."); st.stop()
            
            # Market Caps
            caps = {}
            for t in tickers:
                try: caps[t] = yf.Ticker(t).info.get('marketCap', 1e9)
                except: caps[t] = 1e9
            mkt_w = np.array([caps[t]/sum(caps.values()) for t in tickers])
            
            # BL Model
            returns = np.log(data/data.shift(1)).dropna()
            cov = returns.cov() * 252
            pi = 2.5 * np.dot(cov, mkt_w)
            
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
            
            # Výsledná tabulka
            df = pd.DataFrame({"Ticker": tickers, "Doporučená Alokace": [f"{x*100:.1f} %" for x in res.x]})
            st.table(df)
            
            exp_ret = np.sum(post_rets * res.x)*100
            st.metric("Očekávaný Roční Výnos Portfolia", f"{exp_ret:.2f} %")

else:
    st.info("👈 Vlevo zadej Ticker (např. NVDA) a klikni na SPUSTIT ANALÝZU.")
