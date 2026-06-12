"""
src/pages/portfolio.py
───────────────────────
Portfolio Construction Module page — Mandatory Task B

Paste this file at:  src/pages/portfolio.py
Called from:         app.py  (sidebar nav → "Portfolio Builder")
"""

import numpy as np
import pandas as pd
import streamlit as st

try:
    from src.utils.data_loader import DataLoader
    from src.models.portfolio import PortfolioBuilder
    from src.components.charts import Charts
    from src.components.metrics_display import MetricsDisplay
    from config.settings import INVESTOR_PROFILES
except ImportError:
    from utils.data_loader import DataLoader
    from models.portfolio import PortfolioBuilder
    from components.charts import Charts
    from components.metrics_display import MetricsDisplay
    INVESTOR_PROFILES = {
        "Conservative": {"color": "#10B981", "description": "Capital preservation."},
        "Balanced":     {"color": "#3B82F6", "description": "Balanced growth."},
        "Aggressive":   {"color": "#F59E0B", "description": "Maximum growth."},
    }


def render(loader: DataLoader):
    """Render the Portfolio Builder page."""

    st.markdown("## 💼 Portfolio Construction Module")
    st.caption(
        "Build mean-variance optimised portfolios for different investor profiles. "
        "All allocations are justified with quantitative metrics."
    )

    symbols = loader.available_symbols
    if not symbols:
        MetricsDisplay.info_box("No data files found in `data/`. See Home page.", "warning")
        return

    # ── Profile selector ──────────────────────────────────────────────────────
    st.markdown("### 1️⃣  Select Your Investor Profile")
    profile_cols = st.columns(3)
    profile_descriptions = {
        "Conservative": {
            "emoji": "🛡️",
            "tagline": "Capital preservation first.",
            "traits": ["Low volatility tolerance", "Stable dividend stocks", "Quarterly rebalancing"],
        },
        "Balanced": {
            "emoji": "⚖️",
            "tagline": "Growth with managed risk.",
            "traits": ["Moderate risk tolerance", "Diversified across sectors", "Monthly rebalancing"],
        },
        "Aggressive": {
            "emoji": "🚀",
            "tagline": "Maximum growth potential.",
            "traits": ["High risk tolerance", "Growth & momentum stocks", "Weekly rebalancing"],
        },
    }

    selected_profile = st.radio(
        "Investor Profile", ["Conservative", "Balanced", "Aggressive"],
        horizontal=True, label_visibility="collapsed",
    )

    for col, (prof, info) in zip(profile_cols, profile_descriptions.items()):
        color = INVESTOR_PROFILES[prof]["color"]
        border = "2px solid " + color if prof == selected_profile else "1px solid #334155"
        traits_html = "".join([f"<li style='color:#94A3B8;font-size:0.8rem;'>{t}</li>" for t in info["traits"]])
        col.markdown(f"""
        <div style="background:#1E293B; border:{border}; border-radius:12px; padding:18px; margin:4px 0;">
            <div style="font-size:1.6rem;">{info['emoji']}</div>
            <div style="color:{color}; font-weight:700; font-size:1.0rem; margin:6px 0 4px;">{prof}</div>
            <div style="color:#64748B; font-size:0.82rem; margin-bottom:8px;">{info['tagline']}</div>
            <ul style="margin:0; padding-left:16px;">{traits_html}</ul>
        </div>
        """, unsafe_allow_html=True)

    # ── Stock universe selector ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 2️⃣  Choose Stock Universe")

    col_sym, col_min = st.columns([4, 1])
    with col_sym:
        universe = st.multiselect(
            "Stocks to consider (minimum 5)",
            options=symbols,
            default=symbols[:min(20, len(symbols))],
        )
    with col_min:
        min_history = st.number_input("Min. years data", min_value=1, max_value=10, value=3)

    if len(universe) < 5:
        MetricsDisplay.info_box("Please select at least 5 stocks to build a diversified portfolio.", "warning")
        return

    # ── Build portfolio ───────────────────────────────────────────────────────
    build_btn = st.button("🏗️ Build Portfolio", type="primary", use_container_width=False)

    if build_btn or st.session_state.get(f"portfolio_{selected_profile}"):
        with st.spinner("Optimising portfolio…"):
            try:
                # Load combined price data
                prices = loader.load_combined(universe)

                # Filter by minimum history
                cutoff = prices.index.max() - pd.DateOffset(years=min_history)
                prices = prices[prices.index >= cutoff]
                # Keep stocks with data for at least 80% of the filtered window
                prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))

                if prices.shape[1] < 5:
                    MetricsDisplay.info_box(
                        "After filtering for data quality, fewer than 5 stocks remain. "
                        "Reduce the minimum years requirement or select more stocks.", "warning"
                    )
                    return

                builder = PortfolioBuilder()
                result  = builder.build(selected_profile, prices)
                st.session_state[f"portfolio_{selected_profile}"] = result

            except Exception as e:
                MetricsDisplay.info_box(f"Portfolio optimisation failed: {e}", "error")
                return

    result = st.session_state.get(f"portfolio_{selected_profile}")
    if result is None:
        MetricsDisplay.info_box("Click **Build Portfolio** to run the optimisation.", "info")
        return

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📊 {selected_profile} Portfolio Results")

    metrics = result["metrics"]
    color   = INVESTOR_PROFILES[selected_profile]["color"]

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: MetricsDisplay.metric_card("Annual Return",   f"{metrics['annual_return']*100:.1f}%", color=color, icon="📈")
    with c2: MetricsDisplay.metric_card("Annual Vol.",     f"{metrics['annual_volatility']*100:.1f}%", color="#8B5CF6", icon="〰️")
    with c3: MetricsDisplay.metric_card("Sharpe Ratio",    f"{metrics['sharpe_ratio']:.2f}", color="#3B82F6", icon="⚡")
    with c4: MetricsDisplay.metric_card("Max Drawdown",    f"{metrics['max_drawdown']*100:.1f}%", color="#EF4444", icon="📉")
    with c5: MetricsDisplay.metric_card("# Stocks",        str(metrics["n_stocks"]), color="#0EA5E9", icon="🏢")

    # Sharpe interpretation
    MetricsDisplay.sharpe_meter(metrics["sharpe_ratio"])

    st.markdown("---")

    # Allocation chart + table side by side
    col_chart, col_table = st.columns([2, 3])

    with col_chart:
        fig_pie = Charts.portfolio_allocation(result["weights"], selected_profile)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_table:
        st.markdown("**Portfolio Allocation Details**")
        alloc_df = result["allocation_df"].copy()
        alloc_df["Ann Return"]  = (alloc_df["Ann_Return"] * 100).round(2).astype(str) + "%"
        alloc_df["Ann Vol"]     = (alloc_df["Ann_Volatility"] * 100).round(2).astype(str) + "%"
        alloc_df["Weight"]      = (alloc_df["Weight_Pct"]).round(2).astype(str) + "%"

        display_df = alloc_df[["Stock", "Weight", "Ann Return", "Ann Vol"]].reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Reasoning ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💬 Portfolio Rationale")
    MetricsDisplay.info_box(result["reasoning"], "info")

    # Rebalancing schedule
    schedule = PortfolioBuilder().rebalancing_schedule(selected_profile)
    col_a, col_b, col_c = st.columns(3)
    with col_a: MetricsDisplay.metric_card("Rebalancing", schedule["frequency"], color=color, icon="🔄")
    with col_b: MetricsDisplay.metric_card("Trigger", schedule["trigger"], color="#64748B", icon="⚠️")
    with col_c: MetricsDisplay.metric_card("Review", schedule["review"], color="#64748B", icon="📋")

    # ── Cumulative Return ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Historical Portfolio Performance")
    port_ret = result.get("port_returns")
    if port_ret is not None and len(port_ret) > 10:
        fig_cum = Charts.cumulative_return(port_ret)
        st.plotly_chart(fig_cum, use_container_width=True)

    # ── Efficient Frontier ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Efficient Frontier")
    with st.spinner("Simulating 500 random portfolios…"):
        try:
            prices_ef = loader.load_combined(universe[-20:])  # Use subset for speed
            ef_df     = builder.efficient_frontier(prices_ef, n_points=500)

            # Mark the three profile portfolios
            highlighted = {}
            for prof in ["Conservative", "Balanced", "Aggressive"]:
                try:
                    r = builder.build(prof, prices_ef)
                    highlighted[prof] = {
                        "return":     r["metrics"]["annual_return"],
                        "volatility": r["metrics"]["annual_volatility"],
                    }
                except Exception:
                    pass

            fig_ef = Charts.efficient_frontier(ef_df, highlighted)
            st.plotly_chart(fig_ef, use_container_width=True)
        except Exception as e:
            MetricsDisplay.info_box(f"Efficient frontier unavailable: {e}", "warning")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔗 Asset Correlation Matrix")
    top_stocks = list(result["weights"].keys())
    try:
        prices_corr = loader.load_combined(top_stocks)
        corr        = prices_corr.pct_change().dropna().corr()
        st.plotly_chart(Charts.correlation_heatmap(corr), use_container_width=True)
        MetricsDisplay.info_box(
            "Stocks with low correlation to each other reduce overall portfolio risk "
            "through diversification (modern portfolio theory).", "info"
        )
    except Exception as e:
        MetricsDisplay.info_box(f"Correlation matrix unavailable: {e}", "warning")

    # ── Compare all three profiles ─────────────────────────────────────────────
    with st.expander("📊 Compare All Three Profiles"):
        try:
            compare_rows = []
            for prof in ["Conservative", "Balanced", "Aggressive"]:
                r = builder.build(prof, prices)
                m = r["metrics"]
                compare_rows.append({
                    "Profile":     prof,
                    "Return":      f"{m['annual_return']*100:.1f}%",
                    "Volatility":  f"{m['annual_volatility']*100:.1f}%",
                    "Sharpe":      f"{m['sharpe_ratio']:.2f}",
                    "Max DD":      f"{m['max_drawdown']*100:.1f}%",
                    "# Stocks":    m["n_stocks"],
                })
            st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(str(e))