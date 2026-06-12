"""
src/pages/stock_analysis.py
────────────────────────────
Individual stock explorer page.

Paste this file at:  src/pages/stock_analysis.py
Called from:         app.py  (sidebar nav → "Stock Analysis")
"""

import numpy as np
import pandas as pd
import streamlit as st

try:
    from src.utils.data_loader import DataLoader
    from src.utils.indicators import TechnicalIndicators
    from src.components.charts import Charts
    from src.components.metrics_display import MetricsDisplay, format_pct
except ImportError:
    from utils.data_loader import DataLoader
    from utils.indicators import TechnicalIndicators
    from components.charts import Charts
    from components.metrics_display import MetricsDisplay, format_pct


def render(loader: DataLoader):
    """Render the Stock Analysis page."""

    st.markdown("## 📊 Stock Analysis")
    st.caption("Explore price history, technical indicators, and return statistics for any NIFTY-50 stock.")

    symbols = loader.available_symbols
    if not symbols:
        MetricsDisplay.info_box("No data files found. See Home page for setup instructions.", "warning")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    col_sym, col_period, col_ind = st.columns([2, 2, 3])

    with col_sym:
        symbol = st.selectbox("Select Stock", symbols, index=0)

    with col_period:
        period = st.selectbox("Period",
                              ["All Time", "1 Year", "3 Years", "5 Years"],
                              index=1)

    with col_ind:
        indicators = st.multiselect(
            "Overlay Indicators",
            ["SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26"],
            default=["SMA_20", "SMA_50"],
        )

    # ── Load & filter data ────────────────────────────────────────────────────
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

    if df.empty:
        MetricsDisplay.info_box("Not enough data for the selected period.", "warning")
        return

    returns = df["Daily_Return"].dropna()

    # ── KPI cards ─────────────────────────────────────────────────────────────
    last_close  = df["Close"].iloc[-1]
    first_close = df["Close"].iloc[0]
    total_ret   = (last_close / first_close - 1) * 100
    ann_vol     = returns.std() * np.sqrt(252) * 100
    ann_ret     = returns.mean() * 252 * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        MetricsDisplay.metric_card("Last Close", f"₹{last_close:,.2f}", color="#3B82F6", icon="💹")
    with c2:
        color = "#10B981" if total_ret >= 0 else "#EF4444"
        MetricsDisplay.metric_card("Period Return", f"{total_ret:+.1f}%", color=color, icon="📈")
    with c3:
        MetricsDisplay.metric_card("Ann. Return", f"{ann_ret:+.1f}%", color="#F59E0B", icon="📅")
    with c4:
        MetricsDisplay.metric_card("Volatility", f"{ann_vol:.1f}%", color="#8B5CF6", icon="〰️")
    with c5:
        MetricsDisplay.metric_card("Trading Days", f"{len(df):,}", color="#0EA5E9", icon="🗓️")

    st.markdown("---")

    # ── Candlestick ───────────────────────────────────────────────────────────
    st.markdown("#### 🕯️ Price & Volume")
    fig_candle = Charts.candlestick(df, symbol=symbol)
    # Overlay moving averages
    for ind in indicators:
        if ind in df.columns:
            import plotly.graph_objects as go
            fig_candle.add_trace(go.Scatter(
                x=df.index, y=df[ind], name=ind,
                line=dict(width=1.2), mode="lines",
            ))
    st.plotly_chart(fig_candle, use_container_width=True)

    # ── Technical Indicators ──────────────────────────────────────────────────
    st.markdown("#### 📐 Technical Indicators")
    tab1, tab2, tab3 = st.tabs(["Bollinger Bands", "RSI", "MACD"])

    with tab1:
        if "BB_Upper" in df.columns:
            st.plotly_chart(Charts.bollinger_chart(df, symbol), use_container_width=True)
        else:
            st.info("Not enough data to compute Bollinger Bands for this period.")

    with tab2:
        if "RSI" in df.columns:
            st.plotly_chart(Charts.rsi_chart(df), use_container_width=True)
            rsi_latest = df["RSI"].iloc[-1]
            if rsi_latest > 70:
                MetricsDisplay.info_box(f"RSI = {rsi_latest:.1f} — Stock appears **overbought**. Consider caution.", "warning")
            elif rsi_latest < 30:
                MetricsDisplay.info_box(f"RSI = {rsi_latest:.1f} — Stock appears **oversold**. May present a buying opportunity.", "success")
            else:
                MetricsDisplay.info_box(f"RSI = {rsi_latest:.1f} — RSI is in neutral territory.", "info")
        else:
            st.info("Not enough data for RSI.")

    with tab3:
        if "MACD" in df.columns:
            st.plotly_chart(Charts.macd_chart(df), use_container_width=True)
            macd_val    = df["MACD"].iloc[-1]
            signal_val  = df["MACD_Signal"].iloc[-1]
            if macd_val > signal_val:
                MetricsDisplay.info_box("MACD is above its signal line — **bullish momentum**.", "success")
            else:
                MetricsDisplay.info_box("MACD is below its signal line — **bearish momentum**.", "warning")
        else:
            st.info("Not enough data for MACD.")

    # ── Returns Analysis ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📉 Returns Distribution")
    col_dist, col_stats = st.columns([3, 2])

    with col_dist:
        st.plotly_chart(Charts.returns_distribution(returns, symbol), use_container_width=True)

    with col_stats:
        st.markdown("**Descriptive Statistics**")
        stats = {
            "Mean Daily Return":    f"{returns.mean()*100:.3f}%",
            "Std Dev (Daily)":      f"{returns.std()*100:.3f}%",
            "Annual Return":        f"{ann_ret:.1f}%",
            "Annual Volatility":    f"{ann_vol:.1f}%",
            "Skewness":             f"{returns.skew():.3f}",
            "Kurtosis":             f"{returns.kurtosis():.3f}",
            "Best Day":             f"{returns.max()*100:.2f}%",
            "Worst Day":            f"{returns.min()*100:.2f}%",
            "Positive Days":        f"{(returns > 0).mean()*100:.1f}%",
        }
        for k, v in stats.items():
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        padding:4px 0; border-bottom:1px solid #1E293B;">
                <span style="color:#94A3B8; font-size:0.82rem;">{k}</span>
                <span style="color:#F1F5F9; font-weight:600; font-size:0.85rem;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📉 Drawdown Analysis")
    try:
        from src.models.risk import RiskAssessor
    except ImportError:
        from models.risk import RiskAssessor
    ra       = RiskAssessor()
    dd_series = ra.drawdown_series(df["Close"])
    st.plotly_chart(Charts.drawdown_chart(dd_series, symbol), use_container_width=True)

    mdd = dd_series.min()
    mdd_date = dd_series.idxmin()
    MetricsDisplay.info_box(
        f"Maximum drawdown of **{mdd*100:.1f}%** occurred on {mdd_date.date()}. "
        + ra.interpret_drawdown(mdd), "info"
    )

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("📋 View Raw Data"):
        show_cols = [c for c in ["Open", "High", "Low", "Close", "Volume",
                                  "Daily_Return", "RSI", "MACD"] if c in df.columns]
        st.dataframe(
            df[show_cols].tail(100).sort_index(ascending=False)
            .style.format({c: "{:.4f}" for c in show_cols if c != "Volume"}),
            use_container_width=True,
        )
        st.caption(f"Showing last 100 rows of {len(df):,} total records.")