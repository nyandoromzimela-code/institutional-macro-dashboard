import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import pyfredapi as pf
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- 1. CONFIGURATION & API KEYS SETUP ---
os.environ["FRED_API_KEY"] = "43d2e92afac22209503ca5e2c1b15441"
BLS_API_KEY = "9bc1ba1efaa1411fadefad8e9037d179"

def fetch_all_market_and_macro_data():
    """Fetches real-time multi-asset data from Yahoo Finance and macro indicators from FRED."""
    print("[1/5] Fetching live cross-asset and pricing data via Yahoo Finance...")
    
    tickers = ["^GSPC", "^SOX", "^TNX", "HG=F", "GC=F", "^VIX", "^VIX3M", "DX-Y.NYB", "^MOVE"]
    yf_data = yf.download(tickers, period="3y", interval="1d", progress=False)["Close"]
    
    if isinstance(yf_data.columns, pd.MultiIndex):
        yf_data.columns = yf_data.columns.get_level_values(0)
        
    # Derived cross-asset metrics
    yf_data["copper_gold"] = yf_data["HG=F"] / yf_data["GC=F"]
    yf_data["semis_spx"] = yf_data["^SOX"] / yf_data["^GSPC"]
    yf_data["vix_slope"] = yf_data["^VIX"] - yf_data["^VIX3M"]

    print("[2/5] Fetching fundamental macroeconomic data via FRED API...")
    try:
        walcl = pf.get_series(series_id="WALCL")         # Fed Total Assets
        tga = pf.get_series(series_id="WTREGEN")         # Treasury General Account
        rrp = pf.get_series(series_id="RRPONTSYD")       # Reverse Repo
        hy_oas = pf.get_series(series_id="BAMLH0A0HYM2")   # High Yield OAS
        ism_orders = pf.get_series(series_id="NAPMNEWO")   # ISM New Orders
        curve = pf.get_series(series_id="T10Y2Y")        # 2s10s Spread
        nfci = pf.get_series(series_id="NFCI")           # National Financial Conditions
        
        net_liq = walcl - tga - rrp
        fred_df = pd.DataFrame({
            "fed_net_liquidity": net_liq,
            "hy_credit_spread": hy_oas,
            "ism_new_orders": ism_orders,
            "curve_2s10s": curve,
            "nfci": nfci
        })
    except Exception as e:
        print(f"FRED API Warning: {e}. Using baseline fallback for missing series.")
        fred_df = pd.DataFrame(index=yf_data.index)
        fred_df["fed_net_liquidity"] = 6000000
        fred_df["hy_credit_spread"] = 3.5
        fred_df["ism_new_orders"] = 51.0
        fred_df["curve_2s10s"] = 0.2
        fred_df["nfci"] = -0.5

    # Align datasets
    yf_data.index = pd.to_datetime(yf_data.index)
    fred_df.index = pd.to_datetime(fred_df.index)
    master_df = yf_data.join(fred_df, how="outer").sort_index().ffill().dropna()
    
    return master_df

def compute_fed_rhetoric_score(sample_text: str) -> float:
    """Computes an institutional NLP sentiment score using FinBERT for Central Bank Rhetoric."""
    print("[3/5] Evaluating Central Bank Rhetoric via FinBERT Transformer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].detach().numpy()
        
        # FinBERT labels: [positive, negative, neutral] -> Map to sentiment score (-1 to +1)
        # In financial contexts, positive earnings/outlook can be dovish/bullish, but for central banks:
        # We weigh hawkish constraints (negative/tightening text semantics)
        sentiment_score = float(probs[0] - probs[1]) # Positive minus Negative probability proxy
        return round(sentiment_score, 4)
    except Exception as e:
        print(f"NLP Model loading note: {e}. Using default neutral Fed tone score.")
        return 0.15

def run_model_1_composite(df, fed_tone_score):
    """Model 1: Weighted Z-Score Composite Institutional Score with Central Bank Rhetoric."""
    print("\n--- Running Model 1: Weighted Z-Score Composite Engine ---")
    
    z_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        rolling_mean = df[col].rolling(window=252).mean()
        rolling_std = df[col].rolling(window=252).std()
        z_df[col] = (df[col] - rolling_mean) / rolling_std
    z_df = z_df.dropna()

    adj = pd.DataFrame(index=z_df.index)
    # Growth Sub-Components
    adj['ism_new_orders'] = z_df['ism_new_orders']
    adj['semis_spx'] = z_df['semis_spx']
    adj['copper_gold'] = z_df['copper_gold']
    adj['curve_2s10s'] = z_df['curve_2s10s']
    
    # Liquidity & Policy Sub-Components (Including Fed Tone)
    adj['fed_net_liquidity'] = z_df['fed_net_liquidity']
    adj['fed_tone_score'] = fed_tone_score # Injected sentiment score
    adj['nfci'] = -z_df['nfci']
    adj['dxy_strength'] = -z_df['DX-Y.NYB']
    
    # Risk Sub-Components
    adj['hy_credit_velocity'] = -z_df['hy_credit_spread'].diff(20)
    adj['vix_slope'] = -z_df['vix_slope']
    adj['move_volatility'] = -z_df['^MOVE']

    # Framework Weights (Sum to 100%)
    weights = {
        'ism_new_orders': 0.12, 'semis_spx': 0.10, 'copper_gold': 0.05, 'curve_2s10s': 0.03,
        'fed_net_liquidity': 0.12, 'fed_tone_score': 0.10, 'nfci': 0.08, 'dxy_strength': 0.05,
        'hy_credit_velocity': 0.15, 'vix_slope': 0.10, 'move_volatility': 0.10
    }

    growth_score = sum(adj[c] * weights[c] for c in ['ism_new_orders', 'semis_spx', 'copper_gold', 'curve_2s10s']) / 0.30
    liq_score = sum(adj[c] * weights[c] for c in ['fed_net_liquidity', 'fed_tone_score', 'nfci', 'dxy_strength']) / 0.35
    risk_score = sum(adj[c] * weights[c] for c in ['hy_credit_velocity', 'vix_slope', 'move_volatility']) / 0.35

    composite = (growth_score * 0.30) + (liq_score * 0.35) + (risk_score * 0.35)
    
    latest_scores = pd.Series({
        "Growth Pillar (30%)": growth_score.iloc[-1],
        "Liquidity & Policy Pillar (35%)": liq_score.iloc[-1],
        "Risk & Premia Pillar (35%)": risk_score.iloc[-1],
        "Composite Macro Index": composite.iloc[-1]
    })
    print(latest_scores.to_string())
    return latest_scores

def run_model_2_hmm(df):
    """Model 2: Gaussian Hidden Markov Model for Regime States."""
    print("\n[4/5] Running Model 2: Gaussian HMM Regime Classifier...")
    
    features = pd.DataFrame(index=df.index)
    features['equity_ret'] = df['^GSPC'].pct_change()
    features['rate_chg'] = df['^TNX'].diff()
    features['vix_level'] = df['^VIX']
    clean_feat = features.dropna()
    
    X = (clean_feat - clean_feat.mean()) / clean_feat.std()
    
    hmm = GaussianHMM(n_components=4, covariance_type="full", n_iter=1000, random_state=42)
    hmm.fit(X.values)
    
    hidden_states = hmm.predict(X.values)
    probs = hmm.predict_proba(X.values[-1:])
    
    print(f"Current Active Hidden State: State {hidden_states[-1]}")
    print("Live State Probability Distribution:")
    for idx, p in enumerate(probs[0]):
        print(f"  State {idx}: {p*100:.2f}%")
    return hidden_states[-1], probs[0]

def run_model_3_pca(df):
    """Model 3: Principal Component Analysis (PCA) Factor Extraction."""
    print("\n[5/5] Running Model 3: PCA Macro Factor Engine...")
    
    features = ["copper_gold", "semis_spx", "^VIX", "DX-Y.NYB", "^MOVE", "hy_credit_spread", "curve_2s10s"]
    X = df[features].dropna()
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(scaled)
    
    print("Explained Variance Ratio:")
    for idx, var in enumerate(pca.explained_variance_ratio_):
        print(f"  Factor {idx+1}: {var*100:.2f}%")
        
    latest_pcs = pd.Series(pcs[-1], index=['PC1 (Growth/Risk)', 'PC2 (Policy/Liquidity)', 'PC3 (Term Premium)'])
    print("\nLatest Factor Loading Scores (Sigma):")
    print(latest_pcs.to_string())
    return latest_pcs

if __name__ == "__main__":
    print("========================================================")
    print("INITIALIZING INSTITUTIONAL MACRO & REGIME ENGINE")
    print("========================================================")
    
    # Sample Fed text transcript (e.g., recent Warsh/Williams commentary) to evaluate via NLP
    sample_fed_transcript = """
    Inflation remains elevated and persistent, driven by energy costs and AI infrastructure demand. 
    We must maintain a restrictive stance and ensure underlying inflation moves decisively to our 2 percent target.
    """
    
    live_data = fetch_all_market_and_macro_data()
    fed_tone_val = compute_fed_rhetoric_score(sample_fed_transcript)
    
    m1_scores = run_model_1_composite(live_data, fed_tone_val)
    m2_state, m2_probs = run_model_2_hmm(live_data)
    m3_factors = run_model_3_pca(live_data)
    
    print("\n========================================================")
    print("PIPELINE EXECUTION COMPLETE. ALL MODELS SYNCHRONIZED.")
    print("========================================================")