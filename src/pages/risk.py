"""
src/pages/risk.py
──────────────────
Risk Assessment Module page — Mandatory Task C

Paste this file at:  src/pages/risk.py
Called from:         app.py  (sidebar nav → "Risk Assessment")
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    from src.utils.data_loader import DataLoader
    from src.utils.indicators import TechnicalIndicators
    from src.models.risk import RiskAssessor
    from src.components.charts import Charts, COLORS
    from src.components.metrics_display import MetricsDisplay
except ImportError:
    from utils.data_loader import DataLoader
    from utils.indicators import TechnicalIndicators
    from models.risk import RiskAssessor
    from components.charts import Charts, COLORS
    from components.metrics_display import MetricsDisplay


def render(loader: DataLoader):
    """Render the Risk Assessment page."""

    st.markdown("## 🛡️ Risk Assessment Module")
    st.caption(
        "Evaluate historical risk for individual stocks or compare across "
        "your selected universe. Metrics: Volatility, Sharpe, Sortino, VaR, CVaR, Max Drawdown."
    )

    symbols = loader.available_symbols
    if not symbols:
        MetricsDisplay.info_box("No data files found in `data/`. See Home page.", "warning")
        return

    # ── Mode selector ─────────────────────────────────────────────────────────
    mode = st.radio(
        "Assessment Mode",
        ["Single Stock Deep-Dive", "Multi-Stock Comparison"],
        horizontal=True,
    )

    st.markdown("---")

    if mode == "Single Stock Deep-Dive":
        _render_single(loader, symbols)
    else:
        _render_comparison(loader, symbols)


# ── Single Stock ──────────────────────────────────────────────────────────────

def _render_single(loader: DataLoader, symbols: list):
    col_sym, col_period = st.columns([2, 2])
    with col_sym:
        symbol = st.selectbox("Select Stock", symbols, key="risk_sym")
    with col_period:
        period = st.selectbox("Analysis Period",
                              ["All Time", "5 Years", "3 Years", "1 Year"],
                              index=1, key="risk_period")

    try:
        df_raw = loader.load_stock(symbol)
    except FileNotFoundError as e:
        MetricsDisplay.info_box(str(e), "error")
        return

    ti = TechnicalIndicators()
    df = ti.add_all(df_raw)

    # Period filter
    period_map = {
        "All Time": df.index.min(),
        "5 Years":  df.index.max() - pd.DateOffset(years=5),
        "3 Years":  df.index.max() - pd.DateOffset(years=3),
        "1 Year":   df.index.max() - pd.DateOffset(years=1),
    }
    df = df[df.index >= period_map[period]]

    if len(df) < 60:
        MetricsDisplay.info_box("Not enough data for this period. Choose a longer window.", "warning")
        return

    ra      = RiskAssessor()
    metrics = ra.assess_stock(df)

    if "error" in metrics:
        MetricsDisplay.info_box(metrics["error"], "error")
        return

    # ── Risk badge ────────────────────────────────────────────────────────────
    st.markdown(f"### {symbol} — Risk Profile")
    col_badge, col_space = st.columns([1, 3])
    with col_badge:
        MetricsDisplay.risk_badge(metrics["risk_label"], metrics["risk_score"])

    st.markdown("")

    # ── KPI grid ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        MetricsDisplay.metric_card(
            "Annual Return",
            f"{metrics['annual_return']*100:.1f}%",
            color="#10B981" if metrics["annual_return"] >= 0 else "#EF4444", icon="📈"
        )
    with c2:
        MetricsDisplay.metric_card(
            "Annual Volatility",
            f"{metrics['annual_volatility']*100:.1f}%",
            color="#8B5CF6", icon="〰️"
        )
    with c3:
        MetricsDisplay.metric_card(
            "Max Drawdown",
            f"{metrics['max_drawdown']*100:.1f}%",
            color="#EF4444", icon="📉",
            subtitle=ra.interpret_drawdown(metrics["max_drawdown"])
        )
    with c4:
        MetricsDisplay.metric_card(
            "VaR (95%, Daily)",
            f"{metrics['var_95_daily']*100:.2f}%",
            color="#F97316", icon="⚠️",
            subtitle="Max expected daily loss"
        )

    st.markdown("")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        MetricsDisplay.metric_card("Sharpe Ratio",  f"{metrics['sharpe_ratio']:.3f}",  color="#3B82F6", icon="⚡")
    with c6:
        MetricsDisplay.metric_card("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}", color="#0EA5E9", icon="🎯")
    with c7:
        MetricsDisplay.metric_card("Calmar Ratio",  f"{metrics['calmar_ratio']:.3f}",  color="#6366F1", icon="📐")
    with c8:
        MetricsDisplay.metric_card("CVaR (95%)",    f"{metrics['cvar_95_daily']*100:.2f}%", color="#DC2626", icon="🔴")

    # Sharpe meter
    st.markdown("")
    MetricsDisplay.sharpe_meter(metrics["sharpe_ratio"])
    MetricsDisplay.info_box(ra.interpret_sharpe(metrics["sharpe_ratio"]), "info")

    st.markdown("---")

    # ── Drawdown chart ────────────────────────────────────────────────────────
    st.markdown("#### 📉 Drawdown Over Time")
    dd_series = metrics.get("_drawdown_series")
    if dd_series is not None:
        st.plotly_chart(Charts.drawdown_chart(dd_series, symbol), use_container_width=True)

    # ── Rolling Sharpe ────────────────────────────────────────────────────────
    st.markdown("#### ⚡ Rolling Sharpe Ratio (252-day)")
    rolling_sharpe = metrics.get("_rolling_sharpe")
    if rolling_sharpe is not None and len(rolling_sharpe) > 0:
        st.plotly_chart(Charts.rolling_sharpe(rolling_sharpe, symbol), use_container_width=True)

    # ── Returns distribution ───────────────────────────────────────────────────
    st.markdown("#### 📊 Daily Returns Distribution")
    st.plotly_chart(
        Charts.returns_distribution(df["Daily_Return"].dropna(), symbol),
        use_container_width=True
    )

    # ── VaR visualisation ────────────────────────────────────────────────────
    st.markdown("#### ⚠️ Value at Risk (VaR) Visualisation")
    _render_var_chart(df["Daily_Return"].dropna(), metrics)

    # ── Statistics table ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Full Risk Statistics")

    exclude = {"_drawdown_series", "_rolling_sharpe", "risk_label", "risk_score"}
    stat_rows = [
        {"Metric": k.replace("_", " ").title(), "Value": str(v)}
        for k, v in metrics.items()
        if k not in exclude and v is not None
    ]
    stats_df = pd.DataFrame(stat_rows)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ── Interpretation summary ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💡 Risk Interpretation Summary")
    _render_risk_summary(symbol, metrics, ra)


# ── Multi-Stock Comparison ────────────────────────────────────────────────────

def _render_comparison(loader: DataLoader, symbols: list):
    col_sel, col_period = st.columns([4, 1])
    with col_sel:
        selected = st.multiselect(
            "Select stocks to compare (max 20)",
            options=symbols,
            default=symbols[:min(8, len(symbols))],
            max_selections=20,
        )
    with col_period:
        period_years = st.selectbox("Period (years)", [1, 3, 5, 10], index=2, key="comp_period")

    if len(selected) < 2:
        MetricsDisplay.info_box("Select at least 2 stocks to compare.", "warning")
        return

    run_btn = st.button("📊 Run Comparison", type="primary")
    if not run_btn and "risk_comparison" not in st.session_state:
        MetricsDisplay.info_box("Click **Run Comparison** to analyse the selected stocks.", "info")
        return

    if run_btn:
        with st.spinner(f"Computing risk metrics for {len(selected)} stocks…"):
            ra      = RiskAssessor()
            ti      = TechnicalIndicators()
            results = {}

            for sym in selected:
                try:
                    df_raw = loader.load_stock(sym)
                    df     = ti.add_all(df_raw)
                    cutoff = df.index.max() - pd.DateOffset(years=period_years)
                    df     = df[df.index >= cutoff]
                    if len(df) >= 60:
                        results[sym] = ra.assess_stock(df)
                except Exception:
                    pass

            st.session_state["risk_comparison"] = results

    results = st.session_state.get("risk_comparison", {})
    if not results:
        MetricsDisplay.info_box("No results available. Try clicking Run Comparison.", "warning")
        return

    ra = RiskAssessor()
    comparison_df = ra.compare_stocks(results)

    st.markdown("### 📊 Risk Comparison Table")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ── Bubble chart: Return vs Vol vs Sharpe ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Risk-Return Map")
    _render_risk_return_bubble(results)

    # ── Bar charts: Sharpe and Max DD ─────────────────────────────────────────
    st.markdown("---")
    col_bar1, col_bar2 = st.columns(2)

    with col_bar1:
        st.markdown("**Sharpe Ratio Comparison**")
        sharpe_vals = {sym: m["sharpe_ratio"] for sym, m in results.items() if "error" not in m}
        _bar_chart(sharpe_vals, "Sharpe Ratio", threshold=1.0, higher_better=True)

    with col_bar2:
        st.markdown("**Max Drawdown Comparison**")
        dd_vals = {sym: abs(m["max_drawdown"]) * 100 for sym, m in results.items() if "error" not in m}
        _bar_chart(dd_vals, "Max Drawdown (%)", threshold=20.0, higher_better=False)

    # ── VaR comparison ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚠️ Daily VaR (95%) Comparison")
    var_vals = {sym: abs(m["var_95_daily"]) * 100 for sym, m in results.items() if "error" not in m}
    _bar_chart(var_vals, "Daily VaR 95% (%)", threshold=2.0, higher_better=False)

    # ── Best/worst picks ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏆 Recommendations Based on Risk Metrics")
    _render_recommendations(results)


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _render_var_chart(returns: pd.Series, metrics: dict):
    var_95 = metrics["var_95_daily"]
    cvar   = metrics["cvar_95_daily"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns * 100, nbinsx=80,
        name="Daily Returns",
        marker_color=COLORS["primary"], opacity=0.6, histnorm="probability density",
    ))
    fig.add_vline(x=var_95 * 100, line=dict(color=COLORS["danger"], width=2, dash="dash"),
                  annotation_text=f"VaR 95%: {var_95*100:.2f}%",
                  annotation_position="top left")
    fig.add_vline(x=cvar * 100, line=dict(color="#7C3AED", width=2, dash="dot"),
                  annotation_text=f"CVaR: {cvar*100:.2f}%",
                  annotation_position="bottom left")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="VaR & CVaR on Returns Distribution",
        xaxis_title="Daily Return (%)", yaxis_title="Density",
        margin=dict(l=0, r=0, t=40, b=0), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_risk_return_bubble(results: dict):
    syms  = [s for s, m in results.items() if "error" not in m]
    rets  = [results[s]["annual_return"] * 100 for s in syms]
    vols  = [results[s]["annual_volatility"] * 100 for s in syms]
    shrps = [results[s]["sharpe_ratio"] for s in syms]
    risks = [results[s]["risk_score"] for s in syms]

    fig = go.Figure(go.Scatter(
        x=vols, y=rets, mode="markers+text",
        text=syms, textposition="top center",
        marker=dict(
            size=[max(10, r / 4) for r in risks],
            color=shrps,
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Sharpe"),
            line=dict(color="white", width=1),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Return: %{y:.1f}%<br>"
            "Volatility: %{x:.1f}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="Risk-Return Map (bubble size = risk score, colour = Sharpe)",
        xaxis_title="Annual Volatility (%)", yaxis_title="Annual Return (%)",
        margin=dict(l=0, r=0, t=40, b=0), height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def _bar_chart(values: dict, title: str, threshold: float = None, higher_better: bool = True):
    sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=higher_better)
    syms  = [k for k, _ in sorted_items]
    vals  = [v for _, v in sorted_items]
    colors = []
    for v in vals:
        if threshold is None:
            colors.append(COLORS["primary"])
        elif higher_better:
            colors.append(COLORS["success"] if v >= threshold else COLORS["danger"])
        else:
            colors.append(COLORS["success"] if v <= threshold else COLORS["danger"])

    fig = go.Figure(go.Bar(
        x=syms, y=vals, marker_color=colors, opacity=0.85,
        text=[f"{v:.2f}" for v in vals], textposition="outside",
    ))
    if threshold is not None:
        fig.add_hline(y=threshold, line=dict(color=COLORS["accent"], dash="dash", width=1))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title=title, yaxis_title=title,
        margin=dict(l=0, r=0, t=40, b=0), height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_recommendations(results: dict):
    valid = {s: m for s, m in results.items() if "error" not in m}
    if not valid:
        return

    best_sharpe  = max(valid, key=lambda s: valid[s]["sharpe_ratio"])
    best_sortino = max(valid, key=lambda s: valid[s]["sortino_ratio"])
    lowest_dd    = min(valid, key=lambda s: abs(valid[s]["max_drawdown"]))
    lowest_risk  = min(valid, key=lambda s: valid[s]["risk_score"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        MetricsDisplay.metric_card(
            "Best Risk-Adjusted Return", best_sharpe,
            f"Sharpe: {valid[best_sharpe]['sharpe_ratio']:.2f}",
            "#3B82F6", "⚡"
        )
    with c2:
        MetricsDisplay.metric_card(
            "Best Downside Protection", best_sortino,
            f"Sortino: {valid[best_sortino]['sortino_ratio']:.2f}",
            "#10B981", "🛡️"
        )
    with c3:
        MetricsDisplay.metric_card(
            "Lowest Max Drawdown", lowest_dd,
            f"DD: {valid[lowest_dd]['max_drawdown']*100:.1f}%",
            "#F59E0B", "📉"
        )
    with c4:
        MetricsDisplay.metric_card(
            "Lowest Overall Risk", lowest_risk,
            f"Risk Score: {valid[lowest_risk]['risk_score']:.0f}/100",
            "#8B5CF6", "🎯"
        )


def _render_risk_summary(symbol: str, metrics: dict, ra: RiskAssessor):
    sharpe_interp = ra.interpret_sharpe(metrics["sharpe_ratio"])
    dd_interp     = ra.interpret_drawdown(metrics["max_drawdown"])
    risk_label    = metrics["risk_label"]
    ann_ret       = metrics["annual_return"] * 100
    ann_vol       = metrics["annual_volatility"] * 100
    pos_days      = metrics["positive_days_pct"]

    summary = (
        f"**{symbol}** has an annualised return of **{ann_ret:.1f}%** "
        f"against an annualised volatility of **{ann_vol:.1f}%**. "
        f"The stock is profitable on **{pos_days:.0f}%** of trading days. "
        f"Sharpe assessment: {sharpe_interp}. "
        f"Drawdown assessment: {dd_interp}. "
        f"Overall risk classification: **{risk_label}** (score: {metrics['risk_score']:.0f}/100)."
    )
    MetricsDisplay.info_box(summary, "info")