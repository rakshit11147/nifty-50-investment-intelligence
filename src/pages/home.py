"""
src/pages/home.py
─────────────────
Landing / overview page for the NIFTY-50 Investment Intelligence Platform.

Paste this file at:  src/pages/home.py
Called from:         app.py  (when user selects "Home" in sidebar nav)
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from src.utils.data_loader import DataLoader
    from src.components.charts import Charts
    from src.components.metrics_display import MetricsDisplay
    from config.settings import APP_TITLE, SECTOR_MAP
except ImportError:
    from utils.data_loader import DataLoader
    from components.charts import Charts
    from components.metrics_display import MetricsDisplay
    SECTOR_MAP = {}
    APP_TITLE  = "NIFTY-50 Intelligence"


def render(loader: DataLoader):
    """Render the home / overview page."""

    # ── Hero banner ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background:
            radial-gradient(600px 220px at 90% 0%, rgba(14,165,233,0.25), transparent 60%),
            linear-gradient(120deg, #312E81 0%, #1E1B4B 45%, #0C4A6E 100%);
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: 20px;
        padding: 44px 48px;
        margin-bottom: 24px;
        box-shadow: 0 12px 44px rgba(49,46,129,0.45);
    ">
        <div style="font-size:0.75rem; color:#3B82F6; text-transform:uppercase;
                    letter-spacing:0.15em; font-weight:600; margin-bottom:8px;">
            AI-POWERED DECISION SUPPORT
        </div>
        <h1 style="color:#F1F5F9; font-size:2.6rem; font-weight:800;
                   line-height:1.15; margin:0 0 12px;">
            NIFTY-50 Investment<br>Intelligence Platform
        </h1>
        <p style="color:#94A3B8; font-size:1.05rem; max-width:640px; margin:0;">
            Transform 20+ years of NIFTY-50 market data into actionable investment
            insights — with ML-driven predictions, optimised portfolios, and
            transparent risk analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Platform capability cards ─────────────────────────────────────────────
    st.markdown("### 🧩 Platform Capabilities")
    cols = st.columns(3)
    capabilities = [
        {
            "title": "📊 Stock Analysis",
            "desc": "Deep-dive into any NIFTY-50 stock: price history, technical indicators, "
                    "returns distribution, and sector comparisons.",
            "color": "#3B82F6",
        },
        {
            "title": "🤖 AI Predictor",
            "desc": "XGBoost/Random Forest direction classification + Monte Carlo "
                    "price simulation with evaluation metrics and feature importance.",
            "color": "#F59E0B",
        },
        {
            "title": "💼 Portfolio Builder",
            "desc": "Mean-variance optimised portfolios for Conservative, Balanced, and "
                    "Aggressive investors with quantitative justification.",
            "color": "#10B981",
        },
        {
            "title": "🛡️ Risk Assessment",
            "desc": "Sharpe, Sortino, VaR, CVaR, Maximum Drawdown, and a composite risk "
                    "score with plain-language interpretation.",
            "color": "#EF4444",
        },
        {
            "title": "📈 Efficient Frontier",
            "desc": "Visualise the risk-return tradeoff across thousands of random portfolios "
                    "and locate optimised points on the curve.",
            "color": "#8B5CF6",
        },
        {
            "title": "🔍 Explainability",
            "desc": "Understand model decisions via SHAP feature importances and "
                    "plain-language reasoning behind every recommendation.",
            "color": "#0EA5E9",
        },
    ]
    for i, cap in enumerate(capabilities):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="
                background:#1E293B; border-radius:10px; padding:20px;
                border-left:4px solid {cap['color']}; margin-bottom:12px;
                min-height:130px;
            ">
                <div style="font-size:1.0rem; font-weight:700; color:#F1F5F9;
                            margin-bottom:8px;">{cap['title']}</div>
                <div style="color:#94A3B8; font-size:0.83rem; line-height:1.5;">
                    {cap['desc']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Dataset Overview ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📁 Dataset Overview")

    symbols = loader.available_symbols
    if symbols:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            MetricsDisplay.metric_card("Stocks Loaded", str(len(symbols)),
                                       "Available for analysis", "#3B82F6", "📂")
        with col2:
            MetricsDisplay.metric_card("Sectors Covered", str(len(SECTOR_MAP)),
                                       "Banking, IT, Energy…", "#10B981", "🏭")
        with col3:
            MetricsDisplay.metric_card("Data Range", "Jan 2000 – Apr 2021",
                                       "21 years of market data", "#F59E0B", "📅")
        with col4:
            MetricsDisplay.metric_card("Indicators", "30+",
                                       "RSI, MACD, Bollinger…", "#EF4444", "📐")

        # Sample performance summary
        st.markdown("#### Quick Performance Snapshot")
        st.caption("Top 10 stocks by average daily return (using loaded data)")

        try:
            sample = symbols[:15]
            summary_rows = []
            for sym in sample:
                try:
                    df = loader.load_stock(sym)
                    ret = df["Daily_Return"].dropna()
                    summary_rows.append({
                        "Stock": sym,
                        "Avg Daily Return": f"{ret.mean()*100:.3f}%",
                        "Ann. Volatility":  f"{ret.std()*np.sqrt(252)*100:.1f}%",
                        "Total Records":    len(df),
                        "From": str(df.index.min().date()),
                        "To":   str(df.index.max().date()),
                    })
                except Exception:
                    pass

            if summary_rows:
                df_summary = pd.DataFrame(summary_rows)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
            else:
                st.info("No stock data loaded yet. Place CSV files in the `data/` folder.")
        except Exception as e:
            st.warning(f"Could not load stock data: {e}")
    else:
        MetricsDisplay.info_box(
            "No data files found. Download the NIFTY-50 dataset from Kaggle "
            "and place the CSV files in the `data/` directory, then restart the app.",
            "warning"
        )
        st.markdown("""
        **Quick Setup:**
        1. Download from [Kaggle](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data)
        2. Extract the ZIP file
        3. Copy all `.csv` files to the `data/` folder
        4. Refresh this page
        """)

    # ── Sector breakdown ──────────────────────────────────────────────────────
    if symbols:
        st.markdown("---")
        st.markdown("### 🏭 Sector Breakdown")
        sector_counts = {s: len(stocks) for s, stocks in SECTOR_MAP.items()}
        fig = Charts.sector_performance({s: c / 50 for s, c in sector_counts.items()})
        fig.update_layout(title="Stocks per Sector (as % of NIFTY-50)",
                          yaxis_title="Share of Index (%)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Navigation hint ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="
        background:#1E293B; border-radius:10px; padding:20px 24px;
        border:1px solid #334155; text-align:center;
    ">
        <div style="color:#94A3B8; font-size:0.9rem;">
            Use the <strong style="color:#3B82F6;">sidebar</strong> to navigate between
            Stock Analysis, AI Predictor, Portfolio Builder, and Risk Assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)
    