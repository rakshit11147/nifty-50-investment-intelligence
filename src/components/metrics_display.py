"""
src/components/metrics_display.py
──────────────────────────────────
Streamlit metric card helpers and number formatters.

HOW TO USE
----------
from src.components.metrics_display import MetricsDisplay
import streamlit as st

MetricsDisplay.kpi_row(st, metrics_dict)
MetricsDisplay.risk_gauge(st, risk_score=65)
"""

import streamlit as st


def format_pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"

def format_inr(value: float) -> str:
    """Format as Indian Rupees with ₹ symbol."""
    if abs(value) >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    if abs(value) >= 1e5:
        return f"₹{value/1e5:.2f} L"
    return f"₹{value:,.2f}"

def delta_color(value: float) -> str:
    return "normal" if value >= 0 else "inverse"


class MetricsDisplay:
    """Helper class for rendering metric cards and visual indicators."""

    @staticmethod
    def kpi_row(metrics: dict, columns=4):
        """
        Render a row of KPI metric cards from a metrics dict.

        Parameters
        ----------
        metrics : dict
            Keys displayed as labels, values as (display_value, delta_value) tuples
            or just a plain string/float.
        columns : int
            Number of columns in the row
        """
        keys = list(metrics.keys())
        cols = st.columns(columns)
        for i, key in enumerate(keys):
            val = metrics[key]
            with cols[i % columns]:
                if isinstance(val, tuple):
                    display, delta = val
                    st.metric(label=key, value=display, delta=delta)
                else:
                    st.metric(label=key, value=val)

    @staticmethod
    def risk_badge(risk_label: str, risk_score: float):
        """Render a coloured risk badge."""
        color_map = {
            "Low":       "#10B981",
            "Moderate":  "#F59E0B",
            "High":      "#F97316",
            "Very High": "#EF4444",
        }
        color = color_map.get(risk_label, "#64748B")
        st.markdown(f"""
        <div style="
            display:inline-block;
            background:{color}22;
            border:1px solid {color};
            border-radius:6px;
            padding:6px 14px;
            color:{color};
            font-weight:600;
            font-size:0.9rem;
        ">
            Risk: {risk_label} &nbsp; ({risk_score:.0f}/100)
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def metric_card(title: str, value: str, subtitle: str = "",
                    color: str = "#3B82F6", icon: str = ""):
        """Render a styled single metric card."""
        st.markdown(f"""
        <div class="metric-glass" style="
            background: linear-gradient(160deg, rgba(30,41,59,0.85), rgba(15,23,42,0.95));
            border: 1px solid {color}33;
            border-left: 4px solid {color};
            border-radius: 12px;
            padding: 16px 20px;
            margin: 4px 0;
            box-shadow: 0 2px 12px rgba(0,0,0,0.25);
        ">
            <div style="color:#94A3B8; font-size:0.75rem; text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:4px;">
                {icon} {title}
            </div>
            <div style="color:#F1F5F9; font-size:1.5rem; font-weight:700;">
                {value}
            </div>
            {"<div style='color:#64748B; font-size:0.8rem; margin-top:4px;'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def signal_card(signal: str, probability: float, symbol: str):
        """
        Render a Buy/Sell/Hold signal card.

        Parameters
        ----------
        signal      : 'BUY' | 'SELL' | 'NEUTRAL'
        probability : float  Confidence 0-1
        symbol      : str    Stock ticker
        """
        colors = {"BUY": "#10B981", "SELL": "#EF4444", "NEUTRAL": "#F59E0B"}
        icons  = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "🟡"}
        color  = colors.get(signal, "#64748B")
        icon   = icons.get(signal, "⚪")

        st.markdown(f"""
        <div class="metric-glass" style="
            background: linear-gradient(160deg, {color}22, rgba(15,23,42,0.9));
            border: 1px solid {color};
            border-radius: 12px;
            padding: 10px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 0 16px {color}30;
        ">
            <div style="font-size:1.5rem;">{icon}</div>
            <div>
                <span style="color:{color}; font-size:1.05rem; font-weight:800;
                             letter-spacing:0.04em;">{signal}</span>
                <div style="color:#94A3B8; font-size:0.78rem; margin-top:2px;">
                    {symbol} &nbsp;|&nbsp; Confidence: {probability*100:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def sharpe_meter(sharpe: float):
        """Visual Sharpe meter using a progress-bar style."""
        max_sharpe = 3.0
        clamped    = max(0.0, min(sharpe, max_sharpe))
        pct        = int(clamped / max_sharpe * 100)
        color      = "#10B981" if sharpe >= 1 else ("#F59E0B" if sharpe >= 0 else "#EF4444")
        label      = ("Excellent" if sharpe >= 2 else
                      "Good" if sharpe >= 1 else
                      "Below Average" if sharpe >= 0 else "Negative")

        st.markdown(f"""
        <div style="margin: 8px 0;">
            <div style="display:flex; justify-content:space-between;
                        font-size:0.8rem; color:#94A3B8; margin-bottom:4px;">
                <span>Sharpe Ratio</span>
                <span style="color:{color}; font-weight:700;">
                    {sharpe:.2f} — {label}
                </span>
            </div>
            <div style="background:#1E293B; border-radius:4px; height:8px;">
                <div style="background:{color}; width:{pct}%; height:100%;
                            border-radius:4px; transition:width 0.3s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def info_box(text: str, style: str = "info"):
        """Render a styled info/warning/success box."""
        configs = {
            "info":    ("#3B82F6", "ℹ️"),
            "warning": ("#F59E0B", "⚠️"),
            "success": ("#10B981", "✅"),
            "error":   ("#EF4444", "❌"),
        }
        color, icon = configs.get(style, configs["info"])
        st.markdown(f"""
        <div style="
            background: {color}15;
            border-left: 3px solid {color};
            border-radius: 4px;
            padding: 12px 16px;
            margin: 8px 0;
            color: #E2E8F0;
            font-size: 0.9rem;
        ">
            {icon} {text}
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def profile_card(profile: str, description: str,
                     metrics: dict, color: str):
        """Portfolio profile summary card."""
        st.markdown(f"""
        <div style="
            background: #1E293B;
            border: 1px solid {color}44;
            border-radius: 12px;
            padding: 20px;
            margin: 8px 0;
        ">
            <div style="color:{color}; font-size:1.1rem; font-weight:700; margin-bottom:6px;">
                {profile}
            </div>
            <div style="color:#94A3B8; font-size:0.82rem; margin-bottom:12px;">
                {description}
            </div>
            {"".join([
                f'<div style="display:flex; justify-content:space-between; padding:4px 0;'
                f'border-bottom:1px solid #334155;">'
                f'<span style="color:#64748B; font-size:0.8rem;">{k}</span>'
                f'<span style="color:#F1F5F9; font-weight:600; font-size:0.85rem;">{v}</span>'
                f'</div>'
                for k, v in metrics.items()
            ])}
        </div>
        """, unsafe_allow_html=True)