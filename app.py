"""
app.py
──────
Main Streamlit entry point for the NIFTY-50 Investment Intelligence Platform.

Run with:
    streamlit run app.py

Paste this file at the ROOT of the repository (same level as README.md).
"""

import sys
import os

# Make sure local src/ and config/ are importable from the repo root
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (dark theme polish) ────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] p {
    color: #94A3B8 !important;
    font-size: 0.85rem;
}

/* ── Main content area ───────────────────────────────── */
.main .block-container {
    padding-top: 4rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Metric widgets ──────────────────────────────────── */
[data-testid="metric-container"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label {
    color: #94A3B8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #F1F5F9 !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"] {
    color: #64748B;
    font-size: 0.85rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #3B82F6 !important;
    border-bottom-color: #3B82F6 !important;
}

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #3B82F6;
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: background 0.2s;
}
.stButton > button[kind="primary"]:hover {
    background: #2563EB;
}

/* ── DataFrames ──────────────────────────────────────── */
.stDataFrame {
    border-radius: 8px;
    overflow: hidden;
}

/* ── Dividers ────────────────────────────────────────── */
hr {
    border-color: #1E293B !important;
    margin: 1.2rem 0 !important;
}

/* ── Expander ────────────────────────────────────────── */
details summary {
    color: #94A3B8 !important;
    font-size: 0.88rem;
}

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0F172A; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

/* ═══ Visual upgrade layer ═════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.02em; }

/* Layered glow background */
.stApp {
    background:
        radial-gradient(1100px 500px at 85% -10%, rgba(99,102,241,0.18), transparent 60%),
        radial-gradient(900px 420px at -10% 110%, rgba(14,165,233,0.12), transparent 55%),
        #0B1020 !important;
}

/* Sidebar gradient + accent edge */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11162A 0%, #0B1020 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.20);
}

/* Tabs as gradient pills */
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: none !important; }
.stTabs [data-baseweb="tab"] {
    background: rgba(30,41,59,0.6) !important;
    border-radius: 999px !important;
    padding: 6px 18px !important;
    border: 1px solid rgba(148,163,184,0.15) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366F1, #0EA5E9) !important;
    color: #fff !important;
    border: none !important;
}

/* Gradient buttons with glow + lift */
.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #0EA5E9 100%) !important;
    border: none !important; border-radius: 10px !important;
    color: #fff !important; font-weight: 700 !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 26px rgba(99,102,241,0.55);
}

/* Inputs */
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
    background: rgba(30,41,59,0.7) !important;
    border: 1px solid rgba(148,163,184,0.2) !important;
    border-radius: 10px !important;
}

/* Expanders as glass panels */
[data-testid="stExpander"] {
    background: rgba(30,41,59,0.5);
    border: 1px solid rgba(148,163,184,0.12) !important;
    border-radius: 12px !important;
}

/* Card hover effect (used by metric cards) */
.metric-glass { transition: transform .15s ease, box-shadow .15s ease; }
.metric-glass:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }

/* Pill-style radio groups (top navbar & page toggles) */
div[role="radiogroup"] { gap: 8px; }
div[role="radiogroup"] label {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(148,163,184,0.18) !important;
    border-radius: 999px !important;
    padding: 7px 16px !important;
    margin: 0 !important;
    transition: all .15s ease;
    cursor: pointer;
}
div[role="radiogroup"] label:hover { border-color: #6366F1 !important; }
div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, #6366F1, #0EA5E9) !important;
    border-color: transparent !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.40);
}
div[role="radiogroup"] label:has(input:checked) p { color: #fff !important; font-weight: 700; }
div[role="radiogroup"] label > div:first-child { display: none !important; }  /* hide the radio circle */
div[role="radiogroup"] label p { font-size: 0.86rem !important; }

hr { border-color: rgba(148,163,184,0.12) !important; }
</style>
""", unsafe_allow_html=True)

# ── Imports (after path setup) ─────────────────────────────────────────────────
try:
    from src.utils.data_loader import DataLoader
    from src.pages import home, stock_analysis, predictor, portfolio, risk
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Make sure you have installed all requirements: `pip install -r requirements.txt`")
    st.stop()

# ── Cached data loader ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_loader() -> DataLoader:
    return DataLoader()


# ── Sidebar navigation ────────────────────────────────────────────────────────
def render_topnav() -> str:
    """Horizontal pill navigation bar at the top of the page."""
    nav_options = {
        "🏠 Home":          "Home",
        "📊 Analysis":      "Stock Analysis",
        "🤖 AI Predictor":  "AI Predictor",
        "💼 Portfolio":     "Portfolio Builder",
        "🛡️ Risk":         "Risk Assessment",
    }
    col_brand, col_nav = st.columns([1.3, 3.2])
    with col_brand:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; padding-top:6px;">
            <span style="font-size:1.6rem;">📈</span>
            <span style="font-family:'Space Grotesk',Inter,sans-serif; font-weight:700;
                         font-size:1.05rem; color:#F1F5F9; letter-spacing:-0.01em;">
                NIFTY-50&nbsp;<span style="background:linear-gradient(135deg,#6366F1,#0EA5E9);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Intelligence</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col_nav:
        selected = st.radio(
            "Navigate", list(nav_options.keys()),
            horizontal=True, label_visibility="collapsed", key="topnav",
        )
    st.markdown("<hr style='margin:4px 0 18px 0;'>", unsafe_allow_html=True)
    return nav_options[selected]


def render_sidebar():
    with st.sidebar:
        # Logo / brand
        st.markdown("""
        <div style="padding: 8px 0 24px 0; text-align: center;">
            <div style="font-size: 2rem;">📈</div>
            <div style="color: #F1F5F9; font-weight: 800; font-size: 1.05rem;
                        letter-spacing: -0.01em; margin-top: 4px;">
                NIFTY-50 Intelligence
            </div>
            <div style="color: #475569; font-size: 0.72rem; margin-top: 2px;">
                AI Investment Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dataset status indicator
        loader = get_loader()
        n_syms = len(loader.available_symbols)

        st.markdown("---")
        st.markdown("""
        <div style="background:#1E293B; border-radius:8px; padding:12px 14px;">
            <div style="color:#475569; font-size:0.7rem; text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:8px;">Dataset Status</div>
        """, unsafe_allow_html=True)

        if n_syms > 0:
            st.markdown(f"""
            <div style="color:#10B981; font-size:0.85rem; font-weight:600;">
                ✅ {n_syms} stocks loaded
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color:#EF4444; font-size:0.82rem; font-weight:600;">
                ⚠️ No data found
            </div>
            <div style="color:#64748B; font-size:0.75rem; margin-top:4px;">
                Place NIFTY-50 CSVs in data/
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Info links
        st.markdown("---")
        st.markdown("""
        <div style="color:#475569; font-size:0.72rem; line-height:1.8;">
            <div>📦 Dataset: <a href="https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data"
                 style="color:#3B82F6;" target="_blank">Kaggle</a></div>
            <div>📄 Report: <code>reports/</code></div>
            <div>💾 Models: <code>models_saved/</code></div>
        </div>
        """, unsafe_allow_html=True)


# ── Main router ────────────────────────────────────────────────────────────────
def main():
    loader  = get_loader()
    page    = render_topnav()
    render_sidebar()

    if page == "Home":
        home.render(loader)
    elif page == "Stock Analysis":
        stock_analysis.render(loader)
    elif page == "AI Predictor":
        predictor.render(loader)
    elif page == "Portfolio Builder":
        portfolio.render(loader)
    elif page == "Risk Assessment":
        risk.render(loader)
    else:
        home.render(loader)


if __name__ == "__main__":
    main()
