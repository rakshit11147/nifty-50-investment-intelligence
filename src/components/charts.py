"""
src/components/charts.py
────────────────────────
Reusable Plotly chart functions for the Streamlit dashboard.

Each function returns a plotly Figure object — just pass it to st.plotly_chart().

HOW TO USE
----------
from src.components.charts import Charts

fig = Charts.candlestick(df)
st.plotly_chart(fig, use_container_width=True)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Colour palette (dark-theme friendly) ──────────────────────────────────────
COLORS = {
    "primary":   "#3B82F6",   # blue
    "secondary": "#0EA5E9",   # sky
    "accent":    "#F59E0B",   # amber
    "success":   "#10B981",   # green
    "danger":    "#EF4444",   # red
    "muted":     "#64748B",   # slate-500
    "bg":        "#0F172A",   # slate-900
    "surface":   "#1E293B",   # slate-800
    "text":      "#F1F5F9",   # slate-100
}

LAYOUT_DEFAULTS = dict(
    template     = "plotly_dark",
    paper_bgcolor= "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font         = dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
    margin       = dict(l=0, r=0, t=40, b=0),
    legend       = dict(bgcolor="rgba(0,0,0,0)"),
)


class Charts:
    """Factory class for all dashboard charts."""

    # ── Price Charts ──────────────────────────────────────────────────────────

    @staticmethod
    def candlestick(df: pd.DataFrame, symbol: str = "",
                    show_volume: bool = True) -> go.Figure:
        """
        OHLCV candlestick chart with optional volume bars.
        """
        rows = 2 if show_volume and "Volume" in df.columns else 1
        row_heights = [0.75, 0.25] if rows == 2 else [1.0]
        specs = [[{"secondary_y": False}]] * rows

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            row_heights=row_heights, vertical_spacing=0.02)

        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Price",
            increasing_line_color=COLORS["success"],
            decreasing_line_color=COLORS["danger"],
            increasing_fillcolor=COLORS["success"],
            decreasing_fillcolor=COLORS["danger"],
        ), row=1, col=1)

        if rows == 2:
            up_mask   = df["Close"] >= df["Open"]
            bar_colors = [COLORS["success"] if u else COLORS["danger"] for u in up_mask]
            fig.add_trace(go.Bar(
                x=df.index, y=df["Volume"],
                name="Volume", marker_color=bar_colors,
                opacity=0.7,
            ), row=2, col=1)
            fig.update_yaxes(title_text="Volume", row=2)

        fig.update_layout(
            **LAYOUT_DEFAULTS,
            title=f"{symbol} — Candlestick" if symbol else "Candlestick",
            xaxis_rangeslider_visible=False,
            height=500,
        )
        return fig

    @staticmethod
    def line_chart(df: pd.DataFrame, columns: list,
                   title: str = "", y_label: str = "Price") -> go.Figure:
        """Multi-series line chart."""
        palette = [COLORS["primary"], COLORS["accent"], COLORS["success"],
                   COLORS["danger"], COLORS["secondary"]]
        fig = go.Figure()
        for i, col in enumerate(columns):
            if col not in df.columns:
                continue
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                name=col,
                line=dict(color=palette[i % len(palette)], width=1.5),
                mode="lines",
            ))
        fig.update_layout(**LAYOUT_DEFAULTS, title=title,
                          yaxis_title=y_label, height=400)
        return fig

    # ── Indicator Charts ──────────────────────────────────────────────────────

    @staticmethod
    def rsi_chart(df: pd.DataFrame) -> go.Figure:
        """RSI with overbought/oversold bands."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            line=dict(color=COLORS["primary"], width=1.5),
            name="RSI",
        ))
        fig.add_hline(y=70, line=dict(color=COLORS["danger"], dash="dash"), annotation_text="Overbought 70")
        fig.add_hline(y=30, line=dict(color=COLORS["success"], dash="dash"), annotation_text="Oversold 30")
        fig.add_hrect(y0=30, y1=70, fillcolor=COLORS["primary"], opacity=0.05)
        fig.update_layout(**LAYOUT_DEFAULTS, title="RSI (14)", yaxis_title="RSI", height=250)
        return fig

    @staticmethod
    def macd_chart(df: pd.DataFrame) -> go.Figure:
        """MACD with signal line and histogram."""
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                                  line=dict(color=COLORS["primary"], width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                                  line=dict(color=COLORS["accent"], width=1.5, dash="dot")))
        # Histogram
        hist_colors = [COLORS["success"] if v >= 0 else COLORS["danger"]
                       for v in df["MACD_Hist"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"],
                              name="Histogram", marker_color=hist_colors, opacity=0.7))
        fig.update_layout(**LAYOUT_DEFAULTS, title="MACD", height=250)
        return fig

    @staticmethod
    def bollinger_chart(df: pd.DataFrame, symbol: str = "") -> go.Figure:
        """Price with Bollinger Bands."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="Upper Band",
            line=dict(color=COLORS["danger"], width=1, dash="dot"), opacity=0.7))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="Lower Band",
            line=dict(color=COLORS["success"], width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(59,130,246,0.05)", opacity=0.7))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Middle"], name="Middle (SMA 20)",
            line=dict(color=COLORS["muted"], width=1)))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name="Close",
            line=dict(color=COLORS["primary"], width=2)))
        fig.update_layout(**LAYOUT_DEFAULTS,
                          title=f"{symbol} Bollinger Bands" if symbol else "Bollinger Bands",
                          yaxis_title="Price (₹)", height=400)
        return fig

    # ── Returns & Risk Charts ─────────────────────────────────────────────────

    @staticmethod
    def returns_distribution(returns: pd.Series, symbol: str = "") -> go.Figure:
        """Histogram of daily returns with normal distribution overlay."""
        mu, sigma = returns.mean(), returns.std()
        x_range   = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        normal_y  = (np.exp(-0.5 * ((x_range - mu) / sigma) ** 2) /
                     (sigma * np.sqrt(2 * np.pi)))

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns * 100, nbinsx=80,
            name="Daily Returns",
            marker_color=COLORS["primary"],
            opacity=0.7, histnorm="probability density",
        ))
        fig.add_trace(go.Scatter(
            x=x_range * 100, y=normal_y / 100,
            mode="lines", name="Normal Dist",
            line=dict(color=COLORS["accent"], width=2),
        ))
        fig.add_vline(x=0, line=dict(color=COLORS["text"], dash="dash", width=1))
        fig.update_layout(**LAYOUT_DEFAULTS,
                          title=f"{symbol} Return Distribution" if symbol else "Return Distribution",
                          xaxis_title="Daily Return (%)", yaxis_title="Density", height=350)
        return fig

    @staticmethod
    def drawdown_chart(drawdown_series: pd.Series, symbol: str = "") -> go.Figure:
        """Drawdown area chart (shows how deep losses went)."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown_series.index, y=drawdown_series * 100,
            name="Drawdown",
            fill="tozeroy", fillcolor="rgba(239,68,68,0.20)",
            line=dict(color=COLORS["danger"], width=1.5),
        ))
        fig.add_hline(y=-20, line=dict(color=COLORS["accent"], dash="dash", width=1),
                      annotation_text="-20% threshold")
        fig.update_layout(**LAYOUT_DEFAULTS,
                          title=f"{symbol} Drawdown (%)" if symbol else "Drawdown (%)",
                          yaxis_title="Drawdown (%)", height=300)
        return fig

    @staticmethod
    def rolling_sharpe(sharpe_series: pd.Series, symbol: str = "") -> go.Figure:
        """Rolling Sharpe ratio over time."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sharpe_series.index, y=sharpe_series,
            line=dict(color=COLORS["primary"], width=1.5), name="Rolling Sharpe",
        ))
        fig.add_hline(y=1, line=dict(color=COLORS["success"], dash="dash"),
                      annotation_text="Good (1.0)")
        fig.add_hline(y=0, line=dict(color=COLORS["danger"], dash="dash"),
                      annotation_text="Break-even")
        fig.update_layout(**LAYOUT_DEFAULTS,
                          title=f"{symbol} Rolling Sharpe" if symbol else "Rolling Sharpe",
                          yaxis_title="Sharpe Ratio", height=300)
        return fig

    # ── Portfolio Charts ──────────────────────────────────────────────────────

    @staticmethod
    def portfolio_allocation(weights_dict: dict, profile: str = "") -> go.Figure:
        """Donut chart of portfolio weights."""
        labels = list(weights_dict.keys())
        values = [round(v * 100, 2) for v in weights_dict.values()]
        colors = px.colors.qualitative.Set3

        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.55,
            marker_colors=colors,
            textposition="inside",
            textinfo="label+percent",
        ))
        fig.update_layout(**LAYOUT_DEFAULTS,
                          title=f"{profile} Portfolio Allocation" if profile else "Allocation",
                          height=420)
        fig.update_traces(hovertemplate="%{label}: %{value:.1f}%<extra></extra>")
        return fig

    @staticmethod
    def cumulative_return(port_returns: pd.Series,
                          benchmark_returns: pd.Series = None) -> go.Figure:
        """Cumulative return chart vs optional benchmark."""
        cum = (1 + port_returns).cumprod()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum.index, y=(cum - 1) * 100,
            name="Portfolio", line=dict(color=COLORS["primary"], width=2),
        ))
        if benchmark_returns is not None:
            cum_bm = (1 + benchmark_returns.reindex(cum.index).fillna(0)).cumprod()
            fig.add_trace(go.Scatter(
                x=cum_bm.index, y=(cum_bm - 1) * 100,
                name="Benchmark", line=dict(color=COLORS["muted"], width=1.5, dash="dot"),
            ))
        fig.update_layout(**LAYOUT_DEFAULTS,
                          title="Cumulative Return (%)",
                          yaxis_title="Return (%)", height=380)
        return fig

    @staticmethod
    def efficient_frontier(ef_df: pd.DataFrame,
                           highlighted_portfolios: dict = None) -> go.Figure:
        """Scatter plot of the efficient frontier."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ef_df["volatility"] * 100,
            y=ef_df["return"] * 100,
            mode="markers",
            marker=dict(
                size=5,
                color=ef_df["sharpe"],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Sharpe"),
            ),
            name="Random Portfolios",
            hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        ))

        if highlighted_portfolios:
            colors_h = {"Conservative": COLORS["success"],
                        "Balanced":     COLORS["primary"],
                        "Aggressive":   COLORS["accent"]}
            for name, point in highlighted_portfolios.items():
                fig.add_trace(go.Scatter(
                    x=[point["volatility"] * 100],
                    y=[point["return"] * 100],
                    mode="markers+text",
                    marker=dict(size=14, color=colors_h.get(name, COLORS["primary"]),
                                symbol="star", line=dict(color="white", width=2)),
                    text=[name], textposition="top center",
                    name=name,
                ))

        fig.update_layout(**LAYOUT_DEFAULTS,
                          title="Efficient Frontier",
                          xaxis_title="Annual Volatility (%)",
                          yaxis_title="Annual Return (%)",
                          height=450)
        return fig

    @staticmethod
    def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
        """Correlation heatmap for sector/portfolio stocks."""
        fig = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            colorbar=dict(title="Correlation"),
        ))
        fig.update_layout(**LAYOUT_DEFAULTS,
                          title="Correlation Matrix",
                          height=max(350, len(corr_matrix) * 40))
        return fig

    @staticmethod
    def feature_importance_bar(fi_df: pd.DataFrame) -> go.Figure:
        """Horizontal bar chart of model feature importances."""
        fig = go.Figure(go.Bar(
            x=fi_df["Importance"].head(15),
            y=fi_df["Feature"].head(15),
            orientation="h",
            marker_color=COLORS["primary"],
        ))
        fig.update_layout(**LAYOUT_DEFAULTS,
                          title="Feature Importance (XGBoost)",
                          xaxis_title="Importance Score",
                          height=400,
                          yaxis=dict(autorange="reversed"))
        return fig

    @staticmethod
    def sector_performance(sector_returns: dict) -> go.Figure:
        """Bar chart comparing sectors by annual return."""
        sectors = list(sector_returns.keys())
        values  = [round(v * 100, 2) for v in sector_returns.values()]
        colors  = [COLORS["success"] if v >= 0 else COLORS["danger"] for v in values]

        fig = go.Figure(go.Bar(
            x=sectors, y=values,
            marker_color=colors, opacity=0.85,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        ))
        fig.update_layout(**LAYOUT_DEFAULTS,
                          title="Sector Performance (Annual Return)",
                          yaxis_title="Annual Return (%)", height=350)
        return fig