"""
notebooks/01_EDA.py
────────────────────
Exploratory Data Analysis — convert to Jupyter with:
    jupytext --to notebook 01_EDA.py

Or run directly:
    python notebooks/01_EDA.py

Paste this file at:  notebooks/01_EDA.py
"""

# %% [markdown]
# # 📊 Exploratory Data Analysis — NIFTY-50 Dataset
# This notebook explores the NIFTY-50 dataset covering Jan 2000 – Apr 2021.
# We examine data quality, price distributions, correlations, and sector behaviour.

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

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.figsize"] = (14, 5)

loader = DataLoader()
ti     = TechnicalIndicators()

print("Available symbols:", loader.available_symbols[:10], "…")
print(f"Total stocks in data/: {len(loader.available_symbols)}")

# %% [markdown]
# ## 1. Dataset Summary

# %%
desc_df = loader.describe_dataset()
print(desc_df.head(20).to_string(index=False))

# %% [markdown]
# ## 2. Single Stock Deep-Dive (RELIANCE)

# %%
df = loader.load_stock("RELIANCE")
print(df.shape)
print(df.dtypes)
print(df.describe())

# %%
# Missing value check
print("Missing values per column:")
print(df.isnull().sum())

# %%
# Price history
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
df["Close"].plot(ax=axes[0], color="#3B82F6", linewidth=1)
axes[0].set_title("RELIANCE — Close Price (2000–2021)")
axes[0].set_ylabel("Price (₹)")

df["Volume"].plot(ax=axes[1], color="#F59E0B", alpha=0.7)
axes[1].set_title("RELIANCE — Volume")
axes[1].set_ylabel("Volume")
plt.tight_layout()
plt.savefig("../reports/01_reliance_price_volume.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Returns Distribution

# %%
df_feat = ti.add_all(df)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
df_feat["Daily_Return"].dropna().hist(bins=100, ax=axes[0], color="#3B82F6", alpha=0.8, edgecolor="none")
axes[0].axvline(0, color="white", linestyle="--", linewidth=1)
axes[0].set_title("Daily Return Distribution — RELIANCE")
axes[0].set_xlabel("Daily Return")

from scipy import stats
ret = df_feat["Daily_Return"].dropna()
stats.probplot(ret, plot=axes[1])
axes[1].set_title("Q-Q Plot (Normality Check)")
plt.tight_layout()
plt.savefig("../reports/01_returns_distribution.png", dpi=150)
plt.show()

print(f"\nSkewness: {ret.skew():.4f}")
print(f"Kurtosis: {ret.kurtosis():.4f}")
print(f"Shapiro-Wilk p-value: {stats.shapiro(ret.sample(500, random_state=42))[1]:.6f}")

# %% [markdown]
# ## 4. Correlation Heatmap (Top 10 Stocks)

# %%
top10 = loader.available_symbols[:10]
combined = loader.load_combined(top10)
corr     = combined.pct_change().dropna().corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Return Correlation — Top 10 NIFTY-50 Stocks")
plt.tight_layout()
plt.savefig("../reports/01_correlation_heatmap.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Sector-wise Performance

# %%
from config.settings import SECTOR_MAP

sector_returns = {}
for sector, syms in SECTOR_MAP.items():
    loaded_syms = [s for s in syms if s in loader.available_symbols]
    if not loaded_syms:
        continue
    combined_sec = loader.load_combined(loaded_syms)
    ann_ret = combined_sec.pct_change().dropna().mean().mean() * 252
    sector_returns[sector] = ann_ret

sr_df = pd.Series(sector_returns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 4))
colors = ["#10B981" if v >= 0 else "#EF4444" for v in sr_df.values]
sr_df.plot(kind="bar", ax=ax, color=colors, edgecolor="none")
ax.set_title("Average Annual Return by Sector")
ax.set_ylabel("Annual Return")
ax.set_xticklabels(sr_df.index, rotation=30, ha="right")
plt.tight_layout()
plt.savefig("../reports/01_sector_performance.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Technical Indicators Preview

# %%
df_ind = ti.add_all(loader.load_stock("TCS"))

fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

df_ind["Close"].iloc[-252:].plot(ax=axes[0], color="#3B82F6", label="Close")
df_ind["SMA_20"].iloc[-252:].plot(ax=axes[0], color="#F59E0B", label="SMA 20")
df_ind["SMA_50"].iloc[-252:].plot(ax=axes[0], color="#10B981", label="SMA 50")
df_ind["BB_Upper"].iloc[-252:].plot(ax=axes[0], color="#EF4444", linestyle="--", alpha=0.7, label="BB Upper")
df_ind["BB_Lower"].iloc[-252:].plot(ax=axes[0], color="#EF4444", linestyle="--", alpha=0.7, label="BB Lower")
axes[0].legend(fontsize=8)
axes[0].set_title("TCS — Price with Bollinger Bands & MAs (last year)")

df_ind["RSI"].iloc[-252:].plot(ax=axes[1], color="#8B5CF6")
axes[1].axhline(70, color="#EF4444", linestyle="--", alpha=0.7)
axes[1].axhline(30, color="#10B981", linestyle="--", alpha=0.7)
axes[1].set_title("RSI (14)")

df_ind["MACD"].iloc[-252:].plot(ax=axes[2], color="#3B82F6", label="MACD")
df_ind["MACD_Signal"].iloc[-252:].plot(ax=axes[2], color="#F59E0B", linestyle="--", label="Signal")
axes[2].legend(fontsize=8)
axes[2].set_title("MACD")

df_ind["Volatility_20"].iloc[-252:].plot(ax=axes[3], color="#EF4444")
axes[3].set_title("20-Day Rolling Volatility (Annualised)")

plt.tight_layout()
plt.savefig("../reports/01_technical_indicators.png", dpi=150)
plt.show()

print("\n✅ EDA complete — charts saved to reports/")