# ==============================================================================
# 🛠️ 0. INSTALACE A IMPORT KNIHOVEN (Auto-Install)
# ==============================================================================
import sys
import subprocess

def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except:
        pass # Ignorujeme chyby pokud balíček už existuje

# Instalace potřebných balíčků
try: import arch
except ImportError: 
    print("⏳ Instaluji 'arch'..."); install("arch"); import arch

try: import sklearn
except ImportError: 
    print("⏳ Instaluji 'scikit-learn'..."); install("scikit-learn"); import sklearn

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as sco
from scipy.stats import t
from arch import arch_model
from sklearn.mixture import GaussianMixture

# ==============================================================================
# 🎛️ 1. ŘÍDÍCÍ PANEL
# ==============================================================================
# VYBER REŽIM: "ANALYZER", "BACKTESTER", "OPTIMIZER"
MODE = "ANALYZER"

TICKER_SINGLE = "NVDA"     
SIMULACE = 3000            
PORTFOLIO_TICKERS = ['NVDA', 'MSFT', 'GOOGL', 'AMZN', 'KO', 'BTC-USD', 'GLD']
HISTORIE_ROKY = 5          
DNY_PREDIKCE = 252

# ==============================================================================
# 🌀 2. FRACTAL MATH: HURST EXPONENT
# ==============================================================================
def get_hurst_exponent(time_series, max_lag=20):
    """Vypočítá Hurstův exponent (0.5 = Random, >0.5 = Trend, <0.5 = Mean Reversion)."""
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# ==============================================================================
# 🛰️ 3. NASA MATH: KALMANŮV FILTR
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

# ==============================================================================
# 🧠 4. AI MOZEK: GMM (DETEKCE REŽIMU)
# ==============================================================================
def detect_market_regime(ticker="SPY", period="5y"):
    try:
        data = yf.download(ticker, period=period, progress=False)
        if df_is_multi(data): data = data['Close'][ticker] if ticker in data['Close'] else data.iloc[:, 0]
        else: data = data['Close']

        returns = np.log(data / data.shift(1)).dropna().values.reshape(-1, 1)
        volatility = data.pct_change().rolling(21).std().dropna().values.reshape(-1, 1)
        
        min_len = min(len(returns), len(volatility))
        X = np.column_stack([returns[-min_len:], volatility[-min_len:]])
        
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        gmm.fit(X)
        
        current_state = gmm.predict(X[-1].reshape(1, -1))[0]
        probs = gmm.predict_proba(X[-1].reshape(1, -1))[0]
        means = gmm.means_[:, 1] 
        high_vol_state = np.argmax(means)
        
        is_panic = (current_state == high_vol_state)
        return "PANIC" if is_panic else "CALM", probs[high_vol_state]
    except: return "NEUTRAL", 0.5

def get_macro_data():
    try:
        tnx = yf.download("^TNX", period="5d", progress=False)['Close'].iloc[-1].item()
        vix = yf.download("^VIX", period="5d", progress=False)['Close'].iloc[-1].item()
        regime, panic_prob = detect_market_regime("SPY")
        return tnx/100, vix, regime, panic_prob
    except: return 0.04, 20.0, "NEUTRAL", 0.5

# Pomocná funkce pro detekci formátu dat
def df_is_multi(df):
    return isinstance(df.columns, pd.MultiIndex)

def safe_download(ticker, period):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty: return pd.Series()
        if df_is_multi(df): 
            try: return df['Close'][ticker]
            except: return df.iloc[:, 0]
        return df['Close']
    except: return pd.Series()

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('targetMeanPrice', None)
    except: return None

# ==============================================================================
# ⚙️ 5. JÁDRO SIMULACE (GARCH + KALMAN + AI + FRACTAL)
# ==============================================================================
def run_garch_simulation(train_prices, horizon_days, n_sims, start_price, macro_vix, panic_prob, target_price=None):
    
    # 1. GARCH Kalibrace
    returns = 100 * np.log(1 + train_prices.pct_change().dropna())
    model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
    res = model.fit(disp='off')
    
    omega = res.params['omega'] / 10000
    alpha = res.params['alpha[1]']
    beta = res.params['beta[1]']
    nu = res.params['nu']
    last_vol = res.conditional_volatility.iloc[-1] / 100

    # 2. Fraktální Analýza
    hurst = get_hurst_exponent(train_prices.values[-100:])
    
    # 3. Drift Logic (Hybridní)
    kalman_trend = get_kalman_drift(train_prices)
    long_term_drift = (returns.mean() / 100) * 252
    
    # Váha trendu podle Hursta
    trust_in_trend = max(0, min(1, (hurst - 0.4) * 5))
    tech_drift = (trust_in_trend * kalman_trend) + ((1 - trust_in_trend) * long_term_drift)

    # AI Penalizace
    ai_penalty = -0.20 * panic_prob 
    
    if target_price and target_price > 0:
        fund_upside = (target_price - start_price) / start_price
        fund_drift_daily = np.log(1 + fund_upside)
        base_drift = (0.4 * tech_drift) + (0.6 * fund_drift_daily)
    else:
        base_drift = tech_drift

    final_drift_daily = (base_drift + ai_penalty) / 252

    # 4. Monte Carlo Loop
    paths = np.zeros((horizon_days, n_sims))
    paths[0] = start_price
    sim_vol = np.zeros((horizon_days, n_sims))
    sim_vol[0] = last_vol

    t_shocks = t.rvs(nu, size=(horizon_days, n_sims))
    t_shocks = t_shocks / np.sqrt(nu / (nu - 2))
    lambda_jump = 0.05 * (macro_vix / 15)

    for i in range(1, horizon_days):
        prev_v = sim_vol[i-1]
        prev_shock = t_shocks[i-1] * prev_v
        new_var = omega + alpha*(prev_shock**2) + beta*(prev_v**2)
        curr_v = np.sqrt(new_var)
        sim_vol[i] = curr_v
        
        is_jump = np.random.random(n_sims) < (lambda_jump / 252)
        jump_val = np.random.normal(-0.1, 0.05, n_sims) * is_jump
        
        drift_term = final_drift_daily - 0.5*(curr_v**2)
        diffusion = curr_v * t_shocks[i]
        paths[i] = paths[i-1] * np.exp(drift_term + diffusion + jump_val)
        
    return paths, hurst

# ==============================================================================
# 🔬 6. MODUL: ANALYZER
# ==============================================================================
def run_analyzer():
    print(f"\n🔬 SPUŠTĚN 'FRACTAL QUANT ANALYZER' PRO: {TICKER_SINGLE}")
    print("=" * 60)
    
    prices = safe_download(TICKER_SINGLE, f"{HISTORIE_ROKY}y")
    if prices.empty: print("❌ Chyba dat."); return
    
    start_price = prices.iloc[-1]
    _, vix, regime, panic_prob = get_macro_data()
    target = get_fundamentals(TICKER_SINGLE)
    
    print(f"💰 Cena: {start_price:.2f} USD | AI Regime: {regime}")
    
    paths, hurst = run_garch_simulation(prices, DNY_PREDIKCE, SIMULACE, start_price, vix, panic_prob, target)
    
    h_desc = "Trendující" if hurst > 0.55 else "Mean-Reverting" if hurst < 0.45 else "Náhodný"
    print(f"🌀 Hurst Exponent: {hurst:.2f} -> Trh je {h_desc}")

    # Vizualizace
    final_prices = paths[-1]
    var_95 = np.percentile(final_prices, 5)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graf A: Vějíř
    ax1.plot(paths[:, :100], color='gray', alpha=0.1)
    ax1.plot(np.median(paths, axis=1), color='blue', lw=2, label='Medián')
    if target: ax1.axhline(target, color='green', ls=':', lw=2, label='Cíl Wall St.')
    ax1.fill_between(range(DNY_PREDIKCE), np.percentile(paths, 5, axis=1), np.percentile(paths, 95, axis=1), color='blue', alpha=0.1)
    ax1.set_title(f"Fractal-GARCH Predikce: {TICKER_SINGLE}")
    ax1.legend()
    
    # Graf B: Histogram
    ax2.hist(final_prices, bins=60, density=True, color='purple', alpha=0.6, edgecolor='black')
    ax2.axvline(start_price, color='orange', ls='--', label='Start')
    ax2.axvline(var_95, color='red', ls='--', lw=2, label='VaR 95%')
    ax2.set_title("Rozdělení pravděpodobnosti")
    ax2.legend()
    plt.show()
    
    # Kelly
    ret_sim = (final_prices - start_price) / start_price
    prob_win = np.mean(ret_sim > 0)
    avg_win = np.mean(ret_sim[ret_sim > 0])
    avg_loss = abs(np.mean(ret_sim[ret_sim < 0]))
    kelly = (prob_win - (1 - prob_win) / (avg_win / avg_loss)) if avg_loss > 0 else 0
    
    print(f"\n🎯 INVESTIČNÍ VERDIKT:")
    print(f"1. Očekávaný výnos: {((np.median(final_prices)-start_price)/start_price)*100:.2f} %")
    print(f"2. VaR 95%: {abs((var_95-start_price)/start_price)*100:.2f} %")
    print(f"💎 KELLY ALOKACE: {max(0, kelly*0.5)*100:.2f} %")

# ==============================================================================
# 🧪 7. MODUL: BACKTESTER
# ==============================================================================
def run_backtester():
    print(f"\n🧪 SPUŠTĚN 'REALITY CHECK' PRO: {TICKER_SINGLE}")
    full_data = safe_download(TICKER_SINGLE, f"{HISTORIE_ROKY}y")
    if len(full_data) < 500: print("❌ Málo dat."); return
    
    train_data = full_data[:-DNY_PREDIKCE]
    real_future = full_data[-DNY_PREDIKCE:]
    start_price = train_data.iloc[-1]
    
    print(f"✂️ Backtest od: {train_data.index[-1].strftime('%Y-%m-%d')}")
    _, vix, _, panic_prob = get_macro_data()
    
    paths, _ = run_garch_simulation(train_data, DNY_PREDIKCE, SIMULACE, start_price, vix, panic_prob, target_price=None)
    
    plt.figure(figsize=(14, 7))
    upper_95 = np.percentile(paths, 95, axis=1)
    lower_05 = np.percentile(paths, 5, axis=1)
    median_path = np.median(paths, axis=1)
    real_vals = real_future.values
    min_len = min(len(real_vals), len(median_path))
    x_axis = range(min_len)

    plt.fill_between(x_axis, lower_05[:min_len], upper_95[:min_len], color='blue', alpha=0.15, label='90% Interval')
    plt.plot(x_axis, median_path[:min_len], color='blue', lw=2, label='Model')
    plt.plot(x_axis, real_vals[:min_len], color='red', lw=3, label='REALITA')
    plt.title(f"BACKTEST: {TICKER_SINGLE}")
    plt.legend(); plt.show()
    
    inside = np.sum((real_vals[:min_len] >= lower_05[:min_len]) & (real_vals[:min_len] <= upper_95[:min_len]))
    print(f"📊 Přesnost modelu: {(inside/min_len)*100:.1f} %")

# ==============================================================================
# ⚖️ 8. MODUL: OPTIMIZER
# ==============================================================================
def run_optimizer():
    print(f"\n⚖️ SPUŠTĚN 'PORTFOLIO ARCHITECT' (Black-Litterman + Fractal)")
    data = yf.download(PORTFOLIO_TICKERS, period=f"{HISTORIE_ROKY}y", progress=False)
    if data.empty: print("❌ Chyba dat."); return
    prices = data['Close'] if df_is_multi(data) else data
    
    caps = {}
    print("📡 Stahuji tržní kapitalizace...")
    for t in PORTFOLIO_TICKERS:
        try: caps[t] = yf.Ticker(t).info.get('marketCap', 1e9)
        except: caps[t] = 1e9
    mkt_weights = np.array([caps[t]/sum(caps.values()) for t in PORTFOLIO_TICKERS])
    
    returns = np.log(prices / prices.shift(1)).dropna()
    cov_matrix = returns.cov() * 252
    
    pi = 2.5 * np.dot(cov_matrix, mkt_weights)
    
    my_views = []
    for t in PORTFOLIO_TICKERS:
        my_views.append(get_kalman_drift(prices[t]))
    my_views = np.array(my_views)

    tau = 0.05
    _, _, _, panic_prob = get_macro_data()
    if panic_prob > 0.5: tau = 0.01 

    posterior_rets = (pi + (my_views * tau)) / (1 + tau)
    r_f = 0.04
    
    def neg_sharpe(w):
        p_ret = np.sum(posterior_rets * w)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(p_ret - r_f) / p_vol
    
    cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})
    bnds = tuple((0.05, 0.40) for _ in range(len(PORTFOLIO_TICKERS)))
    res = sco.minimize(neg_sharpe, mkt_weights, method='SLSQP', bounds=bnds, constraints=cons)
    
    print("\n🏆 OPTIMÁLNÍ PORTFOLIO:")
    for i, t in enumerate(PORTFOLIO_TICKERS):
        print(f"  🔹 {t:<8}: {res.x[i]*100:.2f} %")
    
    exp_ret = np.sum(posterior_rets * res.x)
    exp_vol = np.sqrt(np.dot(res.x.T, np.dot(cov_matrix, res.x)))
    print(f"⭐ Sharpe Ratio: {(exp_ret - r_f)/exp_vol:.2f}")

# ==============================================================================
# 🚀 START
# ==============================================================================
if __name__ == "__main__":
    if MODE == "ANALYZER": run_analyzer()
    elif MODE == "BACKTESTER": run_backtester()
    elif MODE == "OPTIMIZER": run_optimizer()
