"""
src/pages/predictor.py
──────────────────────
AI Stock Predictor Engine page — Mandatory Task A

Paste this file at:  src/pages/predictor.py
Called from:         app.py  (sidebar nav → "AI Predictor")
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    from src.utils.data_loader import DataLoader
    from src.utils.indicators import TechnicalIndicators
    from src.models.predictor import StockPredictor
    from src.components.charts import Charts, COLORS
    from src.components.metrics_display import MetricsDisplay
except ImportError:
    from utils.data_loader import DataLoader
    from utils.indicators import TechnicalIndicators
    from models.predictor import StockPredictor
    from components.charts import Charts, COLORS
    from components.metrics_display import MetricsDisplay


def render(loader: DataLoader):
    """Render the AI Predictor Engine page."""

    st.markdown("## 🤖 AI Stock Predictor Engine")
    st.caption(
        "Train ML models on historical data to forecast stock direction, plus a "
        "Monte Carlo price outlook built from each stock's own volatility profile. "
        "Models: XGBoost & Random Forest."
    )

    symbols = loader.available_symbols
    if not symbols:
        MetricsDisplay.info_box("No data files found in `data/`. See Home page.", "warning")
        return

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Model Configuration")
        symbol       = st.selectbox("Stock Symbol", symbols, key="pred_sym")
        horizon      = st.slider("Forecast Horizon (days)", 5, 30, 10)
        do_train     = st.button("🚀 Train Models", use_container_width=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    try:
        df_raw = loader.load_stock(symbol)
    except FileNotFoundError as e:
        MetricsDisplay.info_box(str(e), "error")
        return

    ti = TechnicalIndicators()
    df = ti.add_all(df_raw)

    if len(df) < 200:
        MetricsDisplay.info_box("Insufficient history (< 200 trading days) for reliable model training.", "warning")
        return

    # ── Train on button click ─────────────────────────────────────────────────
    # Cache the predictor across Streamlit reruns (slider moves, nav, etc.)
    cache_key = f"predictor_{symbol}"
    if cache_key in st.session_state:
        pred = st.session_state[cache_key]
    else:
        pred = StockPredictor(symbol=symbol)
        pred.load()  # Try loading saved models first
        st.session_state[cache_key] = pred

    if do_train or not pred.is_trained:
        if do_train:
            with st.spinner(f"Training models on {symbol}…"):
                results = pred.train(df, train_lstm=False, horizon=horizon)
            st.success("✅ Models trained and saved to `models_saved/`")
            st.session_state[f"pred_results_{symbol}"] = results
        else:
            MetricsDisplay.info_box(
                f"No saved models for {symbol}. Click **Train Models** in the sidebar to start.", "info"
            )
            # Show a demo with synthetic evaluation to illustrate the UI
            _render_explainer()
            return

    # ── Direction Prediction ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📡 Direction Forecast (Next {horizon} Trading Days)")

    trained_h = getattr(pred, "clf_horizon", None)
    if trained_h is not None and trained_h != horizon:
        MetricsDisplay.info_box(
            f"Direction models were trained for a **{trained_h}-day** horizon. "
            f"Click **Train Models** in the sidebar to retrain for {horizon} days.", "info"
        )

    col_rf, col_xgb = st.columns(2)
    for col, model_name, label in [(col_rf, "rf", "Random Forest"), (col_xgb, "xgb", "XGBoost")]:
        with col:
            try:
                pred_dir, proba = pred.predict_direction(df, model=model_name)
                signal = "BUY" if pred_dir == 1 else "SELL"
                shown_h = trained_h if trained_h is not None else horizon
                MetricsDisplay.signal_card(signal, proba, f"{symbol} · {label} · {shown_h}-day")
            except Exception as e:
                st.warning(f"{label}: {e}")

    # ── Price Forecast ────────────────────────────────────────────────────────
    if True:  # Monte Carlo price outlook — no deep-learning model required
        st.markdown("---")
        st.markdown(f"### 📈 Price Outlook — Next {horizon} Days (Monte Carlo Simulation)")

        try:
            last_date      = df.index[-1]
            future_dates   = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

            # Historical tail + forecast
            hist_tail  = df["Close"].iloc[-60:]
            last_close = float(df["Close"].iloc[-1])

            # ── ML price target (XGBoost regressor — Mandatory Task A) ────────
            if getattr(pred, "reg_horizon", None) == horizon:
                ml_ret, ml_price = pred.predict_return(df, horizon=horizon)
                mlc1, mlc2, mlc3 = st.columns(3)
                with mlc1:
                    MetricsDisplay.metric_card(f"ML {horizon}-Day Target", f"₹{ml_price:,.2f}",
                                               "XGBoost return regressor", "#F59E0B", "🎯")
                with mlc2:
                    color_ret = "#10B981" if ml_ret >= 0 else "#EF4444"
                    MetricsDisplay.metric_card("Expected Return", f"{ml_ret*100:+.2f}%",
                                               f"over {horizon} trading days", color_ret, "📈")
                with mlc3:
                    MetricsDisplay.metric_card("Last Close", f"₹{last_close:,.2f}",
                                               str(last_date.date()), "#3B82F6", "💹")
                st.markdown("")
            else:
                MetricsDisplay.info_box(
                    f"Click **Train Models** to fit the {horizon}-day price model "
                    "(evaluated with MAE, RMSE, R² and Directional Accuracy).", "info"
                )

            # ── Monte Carlo simulation: 500 GBM paths from historical drift/vol
            rets   = df["Daily_Return"].dropna().tail(504)
            mu, sd = float(rets.mean()), float(rets.std())
            rng    = np.random.default_rng(42)
            shocks = rng.normal(mu, sd, size=(500, horizon))
            paths  = last_close * np.exp(np.cumsum(shocks, axis=1))
            p05, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)

            fig = go.Figure()
            # 90% probability fan
            fig.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(p95) + list(p05[::-1]),
                fill="toself", fillcolor="rgba(99,102,241,0.10)",
                line=dict(color="rgba(0,0,0,0)"), name="90% range (Monte Carlo)",
            ))
            # 50% probability fan
            fig.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(p75) + list(p25[::-1]),
                fill="toself", fillcolor="rgba(99,102,241,0.22)",
                line=dict(color="rgba(0,0,0,0)"), name="50% range (Monte Carlo)",
            ))
            fig.add_trace(go.Scatter(
                x=hist_tail.index, y=hist_tail.values,
                name="Historical", line=dict(color=COLORS["primary"], width=2),
            ))
            fig.add_trace(go.Scatter(
                x=future_dates, y=p50,
                name="Expected Path (median)",
                line=dict(color=COLORS["accent"], width=2.5),
                mode="lines+markers", marker=dict(size=5),
            ))
            # Overlay the ML target as a star at the end of the horizon
            try:
                fig.add_trace(go.Scatter(
                    x=[future_dates[-1]], y=[ml_price],
                    mode="markers+text", text=["ML target"], textposition="top center",
                    marker=dict(size=14, symbol="star", color="#F59E0B",
                                line=dict(color="white", width=1)),
                    name=f"XGB {horizon}-day target",
                ))
            except NameError:
                pass
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                title=f"{symbol} — {horizon}-Day Monte Carlo Outlook (500 simulations)",
                xaxis_title="Date", yaxis_title="Price (₹)",
                margin=dict(l=0, r=0, t=40, b=0), height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table (Monte Carlo percentiles)
            st.markdown("**Projected Price Ranges**")
            forecast_df = pd.DataFrame({
                "Date":              [d.strftime("%Y-%m-%d") for d in future_dates],
                "Pessimistic (5%)":  [f"₹{p:,.2f}" for p in p05],
                "Expected (median)": [f"₹{p:,.2f}" for p in p50],
                "Optimistic (95%)":  [f"₹{p:,.2f}" for p in p95],
                "Expected Change":   [f"{(p / last_close - 1) * 100:+.2f}%" for p in p50],
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
            st.caption(
                "Simulated from the stock's own historical drift and volatility "
                "(500 geometric Brownian motion paths). Ranges quantify uncertainty "
                "instead of pretending to know a single future price."
            )

        except Exception as e:
            MetricsDisplay.info_box(f"Forecast error: {e}", "warning")

    # ── Model Evaluation ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Model Evaluation Metrics")

    results = st.session_state.get(f"pred_results_{symbol}", {})

    tab_rf, tab_xgb, tab_reg = st.tabs(["Random Forest", "XGBoost", "Price Forecast (Regressor)"])

    for tab, key, label in [(tab_rf, "rf", "RF"), (tab_xgb, "xgb", "XGB"), (tab_reg, "reg", "REG")]:
        with tab:
            res = results.get(key)
            if res is None:
                st.info(f"{label} model not yet trained in this session. Train to see metrics.")
                continue

            if key in ("rf", "xgb"):
                acc = res.get("accuracy", 0)
                rep = res.get("report", {})
                c1, c2, c3 = st.columns(3)
                with c1:
                    MetricsDisplay.metric_card("Accuracy", f"{acc*100:.1f}%", color="#3B82F6")
                with c2:
                    precision = rep.get("1", {}).get("precision", 0)
                    MetricsDisplay.metric_card("Precision (UP)", f"{precision:.3f}", color="#10B981")
                with c3:
                    recall = rep.get("1", {}).get("recall", 0)
                    MetricsDisplay.metric_card("Recall (UP)", f"{recall:.3f}", color="#F59E0B")

                # Actual vs Predicted
                y_true = res.get("y_test")
                y_pred = res.get("y_pred")
                if y_true is not None and y_pred is not None:
                    _plot_direction_chart(y_true.values, y_pred)

            elif key == "reg":
                c1, c2, c3, c4 = st.columns(4)
                metrics_map = [
                    ("MAE", f"₹{res['mae']:.2f}", "#EF4444"),
                    ("RMSE", f"₹{res['rmse']:.2f}", "#F59E0B"),
                    ("R² Score", f"{res['r2']:.4f}", "#3B82F6"),
                    ("Directional Acc.", f"{res['directional_accuracy']*100:.1f}%", "#10B981"),
                ]
                for col, (name, val, color) in zip([c1, c2, c3, c4], metrics_map):
                    with col:
                        MetricsDisplay.metric_card(name, val, color=color)

                # Training loss curve
                history = res.get("history", {})
                if history and "loss" in history:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(y=history["loss"], name="Training Loss",
                                                   line=dict(color=COLORS["primary"])))
                    if "val_loss" in history:
                        fig_loss.add_trace(go.Scatter(y=history["val_loss"], name="Val Loss",
                                                       line=dict(color=COLORS["accent"], dash="dot")))
                    fig_loss.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        title="LSTM Training Loss",
                        xaxis_title="Epoch", yaxis_title="Loss",
                        margin=dict(l=0, r=0, t=40, b=0), height=300,
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    if pred.xgb_model is not None:
        st.markdown("---")
        st.markdown("### 🔍 Feature Importance (Explainability)")
        try:
            fi_df = pred.feature_importance()
            st.plotly_chart(Charts.feature_importance_bar(fi_df), use_container_width=True)
            MetricsDisplay.info_box(
                f"Top driver: **{fi_df.iloc[0]['Feature']}** — "
                "This is the most influential indicator in the XGBoost direction model.",
                "info"
            )
        except Exception as e:
            st.warning(f"Feature importance unavailable: {e}")


# ── Helper functions ──────────────────────────────────────────────────────────

def _plot_direction_chart(y_true: np.ndarray, y_pred: np.ndarray):
    """Visualise classification quality: confusion matrix + hit/miss timeline."""
    n  = min(100, len(y_true))
    yt = np.asarray(y_true[-n:]).astype(int)
    yp = np.asarray(y_pred[-n:]).astype(int)
    correct = yt == yp

    col_cm, col_hits = st.columns([1, 1.6])

    # ── Confusion matrix heatmap ──────────────────────────────────────────
    with col_cm:
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(yt, yp):
            cm[t, p] += 1
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Pred Down", "Pred Up"],
            y=["Actual Down", "Actual Up"],
            text=cm, texttemplate="%{text}",
            textfont=dict(size=18),
            colorscale="Blues", showscale=False,
        ))
        fig_cm.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title="Confusion Matrix (last 100 samples)",
            margin=dict(l=0, r=0, t=40, b=0), height=300,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Hit/miss markers + rolling accuracy ───────────────────────────────
    with col_hits:
        roll_acc = pd.Series(correct.astype(int)).rolling(20, min_periods=5).mean() * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(n), y=roll_acc,
            name="Rolling Accuracy (20)",
            line=dict(color=COLORS["primary"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(n), y=np.where(correct, 100, 0),
            mode="markers", name="Hit (top) / Miss (bottom)",
            marker=dict(size=6,
                        color=np.where(correct, COLORS["success"], COLORS["danger"])),
        ))
        fig.add_hline(y=50, line=dict(color=COLORS["muted"], dash="dash", width=1),
                      annotation_text="Coin flip (50%)")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title="Prediction Hits & Rolling Accuracy (last 100 samples)",
            xaxis_title="Test Sample", yaxis_title="Accuracy (%)",
            yaxis=dict(range=[-5, 110]),
            margin=dict(l=0, r=0, t=40, b=0), height=300,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_explainer():
    """Show explanatory content when models aren't yet trained."""
    st.markdown("---")
    st.markdown("### How the Predictor Works")
    cols = st.columns(3)
    with cols[0]:
        MetricsDisplay.metric_card("Step 1", "Feature Engineering",
                                   "30+ technical indicators computed from OHLCV data", "#3B82F6", "📐")
    with cols[1]:
        MetricsDisplay.metric_card("Step 2", "Model Training",
                                   "XGBoost + RF for direction; LSTM for price levels", "#F59E0B", "🧠")
    with cols[2]:
        MetricsDisplay.metric_card("Step 3", "Explainability",
                                   "SHAP values explain which indicators drove each prediction", "#10B981", "🔍")
    MetricsDisplay.info_box(
        "Select a stock in the sidebar and click **Train Models** to begin. "
        "XGBoost and Random Forest train in seconds; LSTM may take 1–2 minutes.", "info"
    )