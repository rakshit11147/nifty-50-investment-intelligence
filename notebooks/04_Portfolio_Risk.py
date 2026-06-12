"""
notebooks/04_Portfolio_Risk.py
───────────────────────────────
Portfolio construction, optimisation, and risk deep-dive.

Paste this file at:  notebooks/04_Portfolio_Risk.py
"""

# %% [markdown]
# # 💼 Portfolio Optimisation & Risk Assessment
# Builds optimised portfolios for three investor profiles and computes
# full risk metrics (Sharpe, Sortino, VaR, CVaR, Drawdown, Calmar).

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from src.utils.data_loader import DataLoader
from src.utils.indicators import TechnicalIndicators
from src.models.portfolio import PortfolioBuilder
from src.models.risk import RiskAssessor

loader  = DataLoader()
ti      = TechnicalIndicators()
builder = PortfolioBuilder()
ra      = RiskAssessor()

# Load combined prices (last 5 years)
symbols = loader.available_symbols
print(f"Building portfolios from {len(symbols)} stocks…")
prices  = loader.load_combined(symbols)
cutoff  = prices.index.max() - pd.DateOffset(years=5)
prices  = prices[prices.index >= cutoff].dropna(axis=1, thresh=int(len(prices)*0.8))
print(f"Price matrix: {prices.shape}  ({prices.index[0].date()} → {prices.index[-1].date()})")

# %% [markdown]
# ## 1. Build All Three Portfolios

# %%
portfolios = {}
for profile in ["Conservative", "Balanced", "Aggressive"]:
    result = builder.build(profile, prices)
    portfolios[profile] = result
    m = result["metrics"]
    print(f"\n── {profile} ────────────────────────────────────")
    print(f"  Return:     {m['annual_return']*100:.1f}%")
    print(f"  Volatility: {m['annual_volatility']*100:.1f}%")
    print(f"  Sharpe:     {m['sharpe_ratio']:.3f}")
    print(f"  Max DD:     {m['max_drawdown']*100:.1f}%")
    print(f"  Stocks:     {m['n_stocks']}")
    print(f"\n  Top 5 Holdings:")
    print(result["allocation_df"][["Stock","Weight_Pct"]].head(5).to_string(index=False))

# %% [markdown]
# ## 2. Portfolio Allocation Charts

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
profile_colors = {
    "Conservative": "#10B981",
    "Balanced":     "#3B82F6",
    "Aggressive":   "#F59E0B",
}
for ax, (profile, result) in zip(axes, portfolios.items()):
    w     = result["weights"]
    top_n = dict(sorted(w.items(), key=lambda x: x[1], reverse=True)[:8])
    other = sum(v for k, v in w.items() if k not in top_n)
    if other > 0.005:
        top_n["Others"] = other
    ax.pie(
        list(top_n.values()),
        labels=list(top_n.keys()),
        autopct="%1.1f%%", startangle=90,
        colors=plt.cm.Set3(np.linspace(0, 1, len(top_n))),
        pctdistance=0.75,
    )
    ax.set_title(f"{profile} Portfolio",
                 color=profile_colors[profile], fontsize=12, fontweight="bold")
plt.suptitle("Portfolio Allocations — Three Investor Profiles", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig("../reports/04_portfolio_allocations.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Cumulative Return Comparison

# %%
fig, ax = plt.subplots(figsize=(14, 6))
line_styles = ["solid", "dashed", "dotted"]
colors_list = ["#10B981", "#3B82F6", "#F59E0B"]

for (profile, result), ls, col in zip(portfolios.items(), line_styles, colors_list):
    port_ret = result["port_returns"]
    cum_ret  = (1 + port_ret).cumprod()
    ax.plot(cum_ret.index, (cum_ret - 1) * 100,
            label=profile, linestyle=ls, color=col, linewidth=2)

ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_title("Cumulative Return — All Three Portfolios (5-Year)")
ax.set_ylabel("Cumulative Return (%)")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../reports/04_cumulative_returns.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Efficient Frontier

# %%
print("Simulating 1000 random portfolios for efficient frontier…")
ef_df = builder.efficient_frontier(prices, n_points=1000)

fig, ax = plt.subplots(figsize=(12, 7))
sc = ax.scatter(
    ef_df["volatility"] * 100, ef_df["return"] * 100,
    c=ef_df["sharpe"], cmap="RdYlGn", alpha=0.5, s=8,
)
plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

for profile, (col, marker) in {
    "Conservative": ("#10B981", "^"),
    "Balanced":     ("#3B82F6", "o"),
    "Aggressive":   ("#F59E0B", "s"),
}.items():
    m = portfolios[profile]["metrics"]
    ax.scatter(m["annual_volatility"]*100, m["annual_return"]*100,
               c=col, s=180, marker=marker, zorder=5,
               edgecolors="white", linewidth=1.5, label=profile)

ax.set_xlabel("Annual Volatility (%)")
ax.set_ylabel("Annual Return (%)")
ax.set_title("Efficient Frontier — NIFTY-50 Universe")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../reports/04_efficient_frontier.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Risk Metrics — Individual Stocks

# %%
sample_stocks = loader.available_symbols[:10]
risk_results  = {}

for sym in sample_stocks:
    try:
        df_r = loader.load_stock(sym)
        df_r = ti.add_all(df_r)
        cutoff_r = df_r.index.max() - pd.DateOffset(years=5)
        df_r = df_r[df_r.index >= cutoff_r]
        if len(df_r) >= 100:
            risk_results[sym] = ra.assess_stock(df_r)
    except Exception as e:
        print(f"  Skip {sym}: {e}")

comparison_df = ra.compare_stocks(risk_results)
print("\n── Risk Comparison Table ──────────────────────────────")
print(comparison_df.to_string(index=False))

# %%
# Drawdown comparison chart
fig, ax = plt.subplots(figsize=(14, 5))
for sym, m in risk_results.items():
    dd = m.get("_drawdown_series")
    if dd is not None:
        ax.plot(dd.index, dd * 100, alpha=0.6, linewidth=1, label=sym)
ax.set_title("Drawdown Comparison — Top 10 Stocks (5-Year)")
ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=7, ncol=5)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../reports/04_drawdown_comparison.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Portfolio Risk Deep-Dive

# %%
for profile, result in portfolios.items():
    port_ret    = result["port_returns"]
    port_prices = (1 + port_ret).cumprod()
    port_risk   = ra.assess_portfolio(port_ret, port_prices)

    print(f"\n── {profile} Portfolio — Risk Metrics ──")
    for k, v in port_risk.items():
        if not k.startswith("_"):
            print(f"  {k:30s}: {v}")

# %%
# VaR comparison bar chart
var_data = {
    prof: abs(ra.assess_portfolio(res["port_returns"])["var_95_daily"]) * 100
    for prof, res in portfolios.items()
}
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(list(var_data.keys()), list(var_data.values()),
              color=[profile_colors[p] for p in var_data], edgecolor="none")
ax.bar_label(bars, fmt="%.3f%%")
ax.set_title("Daily VaR (95%) by Portfolio Profile")
ax.set_ylabel("Daily VaR (%)")
plt.tight_layout()
plt.savefig("../reports/04_var_comparison.png", dpi=150)
plt.show()

print("\n✅ Notebook 04 complete — all charts saved to reports/")