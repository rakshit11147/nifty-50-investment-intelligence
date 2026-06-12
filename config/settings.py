"""
config/settings.py
──────────────────
Global configuration for the NIFTY-50 Investment Intelligence Platform.
All path references, model hyper-parameters, and UI constants live here.
"""

import os
from pathlib import Path

# ── Project Paths ────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT_DIR / "data"
MODELS_DIR      = ROOT_DIR / "models_saved"
REPORTS_DIR     = ROOT_DIR / "reports"
NOTEBOOKS_DIR   = ROOT_DIR / "notebooks"

# Create directories if they don't exist
for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
# The combined Kaggle file (all stocks in one CSV)
NIFTY_ALL_FILE  = DATA_DIR / "NIFTY50_all.csv"

# Date range covered by the dataset
DATA_START_DATE = "2000-01-01"
DATA_END_DATE   = "2021-04-30"

# ── NIFTY-50 Stock Symbols ────────────────────────────────────────────────────
NIFTY50_SYMBOLS = [
    "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
    "BAJFINANCE",  "BHARTIARTL", "BPCL",     "BRITANNIA",  "CIPLA",
    "COALINDIA",   "DIVISLAB",   "DRREDDY",  "EICHERMOT",  "GRASIM",
    "HCLTECH",     "HDFC",       "HDFCBANK", "HDFCLIFE",   "HEROMOTOCO",
    "HINDALCO",    "HINDUNILVR", "ICICIBANK","INDUSINDBK", "INFY",
    "IOC",         "ITC",        "JSWSTEEL", "KOTAKBANK",  "LT",
    "M&M",         "MARUTI",     "NESTLEIND","NTPC",        "ONGC",
    "POWERGRID",   "RELIANCE",   "SBILIFE",  "SBIN",       "SHREECEM",
    "SUNPHARMA",   "TATACONSUM", "TATAMOTORS","TATASTEEL", "TCS",
    "TECHM",       "TITAN",      "ULTRACEMCO","UPL",       "WIPRO",
]

# Sector mapping
SECTOR_MAP = {
    "Banking":              ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK"],
    "Information Technology":["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM"],
    "Energy":               ["RELIANCE", "ONGC", "BPCL", "IOC", "NTPC", "POWERGRID", "COALINDIA"],
    "Consumer Goods":       ["HINDUNILVR", "ITC", "BRITANNIA", "NESTLEIND", "TATACONSUM"],
    "Pharmaceuticals":      ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB"],
    "Automotive":           ["MARUTI", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "TATAMOTORS", "M&M"],
    "Financial Services":   ["HDFC", "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE"],
    "Materials":            ["HINDALCO", "JSWSTEEL", "TATASTEEL", "GRASIM", "ULTRACEMCO", "SHREECEM"],
    "Others":               ["ADANIPORTS", "ASIANPAINT", "LT", "TITAN", "UPL"],
}

# ── Feature Engineering ───────────────────────────────────────────────────────
TECHNICAL_INDICATORS = {
    "sma_windows":  [20, 50, 200],
    "ema_windows":  [12, 26],
    "rsi_period":   14,
    "macd_fast":    12,
    "macd_slow":    26,
    "macd_signal":  9,
    "bb_window":    20,
    "bb_std":       2,
    "atr_period":   14,
    "roc_period":   10,
    "vol_window":   20,
}

# ── Model Hyper-parameters ────────────────────────────────────────────────────
LSTM_CONFIG = {
    "lookback":     60,      # Days of history used per prediction
    "horizon":      30,      # Direct multi-horizon output size (days)
    "units":        [128, 64],
    "dropout":      0.2,
    "epochs":       50,
    "batch_size":   32,
    "test_split":   0.2,
}

RF_CONFIG = {
    "n_estimators": 200,
    "max_depth":    10,
    "random_state": 42,
}

XGB_CONFIG = {
    "n_estimators": 200,
    "max_depth":    6,
    "learning_rate":0.05,
    "subsample":    0.8,
    "random_state": 42,
}

# ── Portfolio Profiles ────────────────────────────────────────────────────────
INVESTOR_PROFILES = {
    "Conservative": {
        "description": "Capital preservation. Low risk tolerance.",
        "target_return":    0.08,    # 8% annual
        "max_volatility":   0.12,    # 12% annual
        "preferred_sectors":["Banking", "Consumer Goods", "Pharmaceuticals"],
        "max_stocks":       8,
        "color":            "#10B981",
    },
    "Balanced": {
        "description": "Moderate growth with managed risk.",
        "target_return":    0.15,
        "max_volatility":   0.20,
        "preferred_sectors":["Information Technology", "Banking", "Energy"],
        "max_stocks":       12,
        "color":            "#3B82F6",
    },
    "Aggressive": {
        "description": "Maximum growth. High risk tolerance.",
        "target_return":    0.25,
        "max_volatility":   0.35,
        "preferred_sectors":["Information Technology", "Automotive", "Materials"],
        "max_stocks":       15,
        "color":            "#F59E0B",
    },
}

# ── Risk Thresholds ───────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "sharpe_good":      1.0,
    "sharpe_excellent": 2.0,
    "max_drawdown_warn":0.20,    # 20% drawdown triggers warning
    "var_confidence":   0.95,    # 95% VaR
}

# ── UI Constants ──────────────────────────────────────────────────────────────
APP_TITLE       = "NIFTY-50 Investment Intelligence"
APP_ICON        = "📈"
APP_DESCRIPTION = "AI-powered investment decision-support platform"

THEME = {
    "primary":      "#1E40AF",
    "secondary":    "#0EA5E9",
    "accent":       "#F59E0B",
    "success":      "#10B981",
    "danger":       "#EF4444",
    "background":   "#0F172A",
    "surface":      "#1E293B",
    "text":         "#F1F5F9",
    "text_muted":   "#94A3B8",
}