"""
notebooks/02_Feature_Engineering.py
─────────────────────────────────────
Feature Engineering — converts raw OHLCV into ML-ready features.

Paste this file at:  notebooks/02_Feature_Engineering.py
"""

# %% [markdown]
# # 🔧 Feature Engineering — NIFTY-50
# Derives 30+ technical indicators from raw OHLCV data and validates them.

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from src.utils.data_loader import DataLoader
from src.utils.indicators import TechnicalIndicators

sns.set_theme(style="darkgrid")
loader = DataLoader()
ti     = TechnicalIndicators()

# %%
# Load a representative stock
df_raw = loader.load_stock("INFY")
print(f"Raw data shape: {df_raw.shape}")
print(df_raw.head())

# %% [markdown]
# ## 1. Compute All Features

# %%
df = ti.add_all(df_raw)
print(f"\nFeature-enriched shape: {df.shape}")
print(f"\nNew columns added ({len(df.columns) - len(df_raw.columns)}):")
new_cols = [c for c in df.columns if c not in df_raw.columns]
for c in new_cols:
    print(f"  • {c}")

# %% [markdown]
# ## 2. Feature Distributions

# %%
key_features = ["RSI", "MACD", "BB_Pct_B", "Volatility_20", "ATR_Pct", "ROC"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flatten(), key_features):
    df[col].dropna().hist(bins=60, ax=ax, color="#3B82F6", alpha=0.8, edgecolor="none")
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("")
plt.suptitle("Feature Distributions — INFY", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("../reports/02_feature_distributions.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Feature-Target Correlation

# %%
# Create binary target: 1 if next-day close > today's close
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

feature_cols = ti.feature_columns
available    = [c for c in feature_cols if c in df.columns]
corr_target  = df[available + ["Target"]].dropna().corr()["Target"].drop("Target")

fig, ax = plt.subplots(figsize=(5, 10))
corr_target.sort_values().plot(kind="barh", ax=ax, color=[
    "#10B981" if v >= 0 else "#EF4444" for v in corr_target.sort_values().values
], edgecolor="none")
ax.axvline(0, color="white", linewidth=0.8)
ax.set_title("Feature Correlation with Next-Day Direction")
ax.set_xlabel("Pearson Correlation")
plt.tight_layout()
plt.savefig("../reports/02_feature_target_correlation.png", dpi=150)
plt.show()

print("\nTop positive correlators (bullish indicators):")
print(corr_target.nlargest(5))
print("\nTop negative correlators (bearish indicators):")
print(corr_target.nsmallest(5))

# %% [markdown]
# ## 4. Feature Stationarity Check (ADF Test)

# %%
from statsmodels.tsa.stattools import adfuller

results_adf = []
for col in available[:10]:
    series = df[col].dropna()
    if len(series) < 50:
        continue
    stat, p_val, _, _, _, _ = adfuller(series)
    results_adf.append({
        "Feature":   col,
        "ADF Stat":  round(stat, 4),
        "p-value":   round(p_val, 6),
        "Stationary": "✅ Yes" if p_val < 0.05 else "❌ No",
    })

adf_df = pd.DataFrame(results_adf)
print(adf_df.to_string(index=False))

# %% [markdown]
# ## 5. Multi-Stock Feature Comparison

# %%
stocks     = loader.available_symbols[:6]
rsi_values = {}
for sym in stocks:
    try:
        d = ti.add_all(loader.load_stock(sym))
        rsi_values[sym] = d["RSI"].dropna().values[-252:]  # last year
    except Exception:
        pass

fig, ax = plt.subplots(figsize=(14, 4))
for sym, vals in rsi_values.items():
    ax.plot(vals, label=sym, alpha=0.7, linewidth=1)
ax.axhline(70, color="#EF4444", linestyle="--", alpha=0.6)
ax.axhline(30, color="#10B981", linestyle="--", alpha=0.6)
ax.set_title("RSI (14) — Last Year Comparison")
ax.set_ylabel("RSI")
ax.legend(fontsize=8, ncol=3)
plt.tight_layout()
plt.savefig("../reports/02_rsi_comparison.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Export Feature-Engineered Dataset

# %%
# Build a combined feature dataset for all available stocks
all_features = []
for sym in loader.available_symbols:
    try:
        d = ti.add_all(loader.load_stock(sym))
        d["Symbol"] = sym
        all_features.append(d)
    except Exception:
        pass

combined_features = pd.concat(all_features)
combined_features.to_csv("../data/features_all.csv")
print(f"\n✅ Combined feature dataset saved: {combined_features.shape}")
print(f"   Symbols: {combined_features['Symbol'].nunique()}")
print(f"   Features: {combined_features.shape[1] - 1}")