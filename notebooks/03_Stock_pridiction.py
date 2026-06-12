"""
notebooks/03_Stock_Prediction.py
──────────────────────────────────
Model training, evaluation, and comparison for stock prediction.

Paste this file at:  notebooks/03_Stock_Prediction.py
"""

# %% [markdown]
# # 🤖 Stock Prediction — Model Training & Evaluation
# Trains LSTM, XGBoost, and Random Forest models for a selected stock.
# Reports MAE, RMSE, R², and Directional Accuracy.

# %%
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from src.utils.data_loader import DataLoader
from src.utils.indicators import TechnicalIndicators
from src.models.predictor import StockPredictor, TF_AVAILABLE
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)

loader = DataLoader()
ti     = TechnicalIndicators()

# ── Choose stock to train on ──────────────────────────────────────────────────
SYMBOL = "RELIANCE"   # Change to any available symbol

df_raw = loader.load_stock(SYMBOL)
df     = ti.add_all(df_raw)
print(f"[{SYMBOL}] Data shape after feature engineering: {df.shape}")
print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")

# %% [markdown]
# ## 1. Random Forest — Direction Classifier

# %%
pred = StockPredictor(symbol=SYMBOL)
rf_results = pred.train_random_forest(df)

print("\n── Random Forest Evaluation ──────────────────────")
print(f"Accuracy:  {rf_results['accuracy']:.4f}")
print("\nClassification Report:")
rep = rf_results["report"]
for cls, vals in rep.items():
    if isinstance(vals, dict):
        print(f"  Class {cls}: Precision={vals['precision']:.3f}  "
              f"Recall={vals['recall']:.3f}  F1={vals['f1-score']:.3f}")

# %%
# Plot confusion-style direction chart
y_true = rf_results["y_test"].values
y_pred = rf_results["y_pred"]

fig, ax = plt.subplots(figsize=(14, 3))
n = min(200, len(y_true))
ax.plot(y_true[-n:], label="Actual", color="#3B82F6", linewidth=1.5)
ax.plot(y_pred[-n:], label="Predicted (RF)", color="#F59E0B", linestyle="--", linewidth=1.5)
ax.set_title(f"{SYMBOL} — Random Forest: Predicted vs Actual Direction (last {n} test samples)")
ax.set_yticks([0, 1])
ax.set_yticklabels(["Down", "Up"])
ax.legend()
plt.tight_layout()
plt.savefig("../reports/03_rf_predictions.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. XGBoost — Direction Classifier

# %%
xgb_results = pred.train_xgboost(df)
print("\n── XGBoost Evaluation ──────────────────────────")
print(f"Accuracy:  {xgb_results['accuracy']:.4f}")

# %%
# Feature importance
fi_df = pred.feature_importance()
fig, ax = plt.subplots(figsize=(8, 6))
fi_df.head(15).sort_values("Importance").plot(
    kind="barh", x="Feature", y="Importance", ax=ax,
    color="#3B82F6", edgecolor="none", legend=False,
)
ax.set_title("XGBoost Feature Importance (Top 15)")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("../reports/03_feature_importance.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. LSTM — Price Forecasting

# %%
if TF_AVAILABLE:
    lstm_results = pred.train_lstm(df)
    print("\n── LSTM Evaluation ────────────────────────────")
    print(f"MAE:                 ₹{lstm_results['mae']:.2f}")
    print(f"RMSE:                ₹{lstm_results['rmse']:.2f}")
    print(f"R² Score:            {lstm_results['r2']:.4f}")
    print(f"Directional Acc.:    {lstm_results['directional_accuracy']*100:.2f}%")

    # Training loss
    history = lstm_results["history"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["loss"],     label="Training Loss",   color="#3B82F6")
    ax.plot(history["val_loss"], label="Validation Loss", color="#F59E0B", linestyle="--")
    ax.set_title("LSTM Training Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../reports/03_lstm_loss.png", dpi=150)
    plt.show()

    # Predicted vs Actual prices
    y_true_p = lstm_results["y_true"]
    y_pred_p = lstm_results["y_pred"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true_p[-300:], label="Actual Close",  color="#3B82F6", linewidth=1.5)
    ax.plot(y_pred_p[-300:], label="LSTM Forecast", color="#F59E0B", linestyle="--", linewidth=1.5)
    ax.set_title(f"{SYMBOL} — LSTM: Predicted vs Actual Close Price (last 300 test points)")
    ax.set_ylabel("Price (₹)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../reports/03_lstm_predictions.png", dpi=150)
    plt.show()
else:
    print("TensorFlow not installed — LSTM training skipped.")
    print("Install with: pip install tensorflow")

# %% [markdown]
# ## 4. Model Comparison Summary

# %%
comparison = {
    "Model":    ["Random Forest", "XGBoost"],
    "Task":     ["Direction", "Direction"],
    "Accuracy": [rf_results["accuracy"], xgb_results["accuracy"]],
}
if TF_AVAILABLE and "lstm_results" in dir():
    comparison["Model"].append("LSTM")
    comparison["Task"].append("Price Level")
    comparison["Accuracy"].append(lstm_results["directional_accuracy"])

comp_df = pd.DataFrame(comparison)
print("\n── Model Comparison ───────────────────────────────────")
print(comp_df.to_string(index=False))

# %% [markdown]
# ## 5. Future Price Forecast (LSTM)

# %%
if TF_AVAILABLE:
    HORIZON = 10
    future_prices = pred.predict_prices(df, horizon=HORIZON)
    last_close    = df["Close"].iloc[-1]
    future_dates  = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=HORIZON)

    print(f"\n── {HORIZON}-Day LSTM Price Forecast for {SYMBOL} ──")
    for date, price in zip(future_dates, future_prices):
        change = (price / last_close - 1) * 100
        print(f"  {date.date()}  ₹{price:,.2f}  ({change:+.2f}%)")

    fig, ax = plt.subplots(figsize=(12, 5))
    tail = df["Close"].iloc[-60:]
    ax.plot(tail.index, tail.values, label="Historical", color="#3B82F6", linewidth=2)
    ax.plot(future_dates, future_prices, label="Forecast", color="#F59E0B",
            linewidth=2, linestyle="--", marker="o", markersize=5)
    ax.set_title(f"{SYMBOL} — {HORIZON}-Day LSTM Price Forecast")
    ax.set_ylabel("Close Price (₹)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../reports/03_future_forecast.png", dpi=150)
    plt.show()

print("\n✅ Notebook 03 complete — all charts saved to reports/")