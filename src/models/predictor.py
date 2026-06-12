"""
src/models/predictor.py
───────────────────────
Stock Predictor Engine — Mandatory Task A

Implements three prediction approaches:
  1. LSTM         — deep learning for price-level forecasting
  2. RandomForest — direction (up/down) classification
  3. XGBoost      — direction (up/down) classification

HOW TO USE
----------
from src.models.predictor import StockPredictor

pred = StockPredictor(symbol="RELIANCE")
pred.train(df_with_features)

# Price forecast
future_prices = pred.predict_prices(df, horizon=10)

# Direction forecast (1 = up, 0 = down)
direction, proba = pred.predict_direction(df)

# Evaluation metrics
metrics = pred.evaluate(df)
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

try:
    from config.settings import MODELS_DIR, LSTM_CONFIG, RF_CONFIG, XGB_CONFIG
    from src.utils.indicators import TechnicalIndicators
except ImportError:
    MODELS_DIR  = Path("models_saved")
    LSTM_CONFIG = {"lookback": 60, "horizon": 30, "units": [128, 64], "dropout": 0.2,
                   "epochs": 50, "batch_size": 32, "test_split": 0.2}
    RF_CONFIG   = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
    XGB_CONFIG  = {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05,
                   "subsample": 0.8, "random_state": 42}
    from src.utils.indicators import TechnicalIndicators

MODELS_DIR = Path(MODELS_DIR)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Optional TensorFlow import — graceful fallback if not installed
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
   


class StockPredictor:
    """
    Unified stock prediction engine.

    Parameters
    ----------
    symbol : str
        NIFTY-50 ticker, e.g. 'RELIANCE'
    """

    FEATURE_COLS = [
        "Daily_Return", "Volatility_20", "RSI", "MACD", "MACD_Hist",
        "BB_Pct_B", "BB_Width", "ATR_Pct", "ROC",
        "Price_vs_SMA20", "Price_vs_SMA50",
        "Momentum_5", "Momentum_20", "HL_Ratio", "OC_Ratio",
    ]

    def __init__(self, symbol: str = "RELIANCE"):
        self.symbol      = symbol
        self.scaler      = MinMaxScaler()
        self.price_scaler = MinMaxScaler()
        self.rf_model    = None
        self.xgb_model   = None
        self.lstm_model  = None
        self.ti          = TechnicalIndicators()
        self.is_trained  = False
        self.trained_features = None
        self.reg_model   = None
        self.reg_horizon = None
        self.clf_horizon = None

    # ── Feature preparation ───────────────────────────────────────────────────

    def _prepare_features(self, df: pd.DataFrame):
        """Extract and scale classification features."""
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        X = df[available].copy()
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        return X

    def _create_labels(self, df: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """Binary label: 1 if close `horizon` days ahead > today's close, else 0."""
        return (df["Close"].shift(-horizon) > df["Close"]).astype(int)

    def _train_test_split(self, X: pd.DataFrame, y: pd.Series):
        """Time-series split (no shuffle)."""
        n     = len(X)
        split = int(n * (1 - LSTM_CONFIG["test_split"]))
        return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    # ── Random Forest ─────────────────────────────────────────────────────────

    def train_random_forest(self, df: pd.DataFrame, horizon: int = 1):
        """Train Random Forest classifier for `horizon`-day direction."""
        X = self._prepare_features(df)
        y = self._create_labels(df, horizon).reindex(X.index).dropna().astype(int)
        X = X.reindex(y.index)

        X_train, X_test, y_train, y_test = self._train_test_split(X, y)

        self.rf_model = RandomForestClassifier(**RF_CONFIG)
        self.rf_model.fit(X_train, y_train)

        y_pred = self.rf_model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.clf_horizon = horizon
        # Save
        joblib.dump(self.rf_model, MODELS_DIR / f"rf_{self.symbol}.pkl")
        print(f"[RF-{self.symbol}] {horizon}-day Accuracy: {acc:.4f}")
        return {"accuracy": acc, "report": report, "X_test": X_test, "y_test": y_test, "y_pred": y_pred}

    # ── XGBoost ───────────────────────────────────────────────────────────────

    def train_xgboost(self, df: pd.DataFrame, horizon: int = 1):
        """Train XGBoost classifier for `horizon`-day direction."""
        X = self._prepare_features(df)
        y = self._create_labels(df, horizon).reindex(X.index).dropna().astype(int)
        X = X.reindex(y.index)

        X_train, X_test, y_train, y_test = self._train_test_split(X, y)

        self.xgb_model = XGBClassifier(
            **XGB_CONFIG,
            eval_metric="logloss",
            verbosity=0,
        )
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        self.trained_features = list(X_train.columns)

        y_pred = self.xgb_model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        self.clf_horizon = horizon
        joblib.dump(self.xgb_model, MODELS_DIR / f"xgb_{self.symbol}.pkl")
        print(f"[XGB-{self.symbol}] {horizon}-day Accuracy: {acc:.4f}")
        return {"accuracy": acc, "report": report, "X_test": X_test, "y_test": y_test, "y_pred": y_pred}

    # ── Price/Return Regressor (Mandatory Task A: horizon forecasting) ────────

    def train_regressor(self, df: pd.DataFrame, horizon: int = 10):
        """
        Train an XGBoost regressor to predict the `horizon`-day forward return.
        Evaluated with MAE, RMSE, R2 and Directional Accuracy on a
        time-ordered holdout (last 20%, no lookahead leakage).
        """
        from xgboost import XGBRegressor

        X = self._prepare_features(df)
        fwd_ret = (df["Close"].shift(-horizon) / df["Close"] - 1).reindex(X.index)
        y = fwd_ret.dropna()
        X = X.reindex(y.index)

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        self.reg_model = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42, verbosity=0,
        )
        self.reg_model.fit(X_train, y_train)
        self.reg_horizon = horizon

        # Convert predicted returns to price levels for interpretable metrics
        pred_ret   = self.reg_model.predict(X_test)
        close_te   = df["Close"].reindex(y_test.index).values
        pred_price = close_te * (1 + pred_ret)
        true_price = close_te * (1 + y_test.values)

        mae    = mean_absolute_error(true_price, pred_price)
        rmse   = np.sqrt(mean_squared_error(true_price, pred_price))
        r2     = r2_score(true_price, pred_price)
        diracc = float(np.mean(np.sign(pred_ret) == np.sign(y_test.values)))

        joblib.dump(self.reg_model, MODELS_DIR / f"reg_{self.symbol}_{horizon}d.pkl")
        print(f"[REG-{self.symbol}] MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f} Dir={diracc:.4f}")
        return {
            "mae": mae, "rmse": rmse, "r2": r2, "directional_accuracy": diracc,
            "y_true": true_price, "y_pred": pred_price,
            "dates": y_test.index, "horizon": horizon,
        }

    def predict_return(self, df: pd.DataFrame, horizon: int = 10):
        """Predict the `horizon`-day forward return and implied target price."""
        if self.reg_model is None:
            raise RuntimeError("Regressor not trained yet.")
        X = self._prepare_features(df)
        ret = float(self.reg_model.predict(X.iloc[[-1]])[0])
        last_close = float(df["Close"].iloc[-1])
        return ret, last_close * (1 + ret)

    # ── LSTM ──────────────────────────────────────────────────────────────────

    def _build_lstm_sequences(self, prices: np.ndarray, lookback: int, horizon: int = 1):
        """Create (X, y) sequences. y holds the next `horizon` prices
        (direct multi-horizon forecasting — avoids the error compounding of
        recursive one-step prediction)."""
        X, y = [], []
        for i in range(lookback, len(prices) - horizon + 1):
            X.append(prices[i - lookback: i])
            y.append(prices[i: i + horizon].flatten())
        return np.array(X), np.array(y)

    def train_lstm(self, df: pd.DataFrame):
        """Train LSTM for price-level forecasting."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for LSTM training.")

        cfg      = LSTM_CONFIG
        lookback = cfg["lookback"]
        prices   = df["Close"].values.reshape(-1, 1)
        scaled   = self.price_scaler.fit_transform(prices)
        # Save fitted scaler
        joblib.dump(
            self.price_scaler,
            MODELS_DIR / f"lstm_scaler_{self.symbol}.pkl"
        )


        horizon = cfg.get("horizon", 30)
        X, y = self._build_lstm_sequences(scaled, lookback, horizon)
        split = int(len(X) * (1 - cfg["test_split"]))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Reshape for LSTM: (samples, timesteps, features)
        X_train = X_train.reshape(*X_train.shape, 1)
        X_test  = X_test.reshape(*X_test.shape, 1)

        model = Sequential([
            Input(shape=(lookback, 1)),
            LSTM(cfg["units"][0], return_sequences=True),
            Dropout(cfg["dropout"]),
            LSTM(cfg["units"][1], return_sequences=False),
            Dropout(cfg["dropout"]),
            Dense(64, activation="relu"),
            Dense(horizon),   # direct multi-horizon output: all future days at once
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")

        ckpt_path = str(MODELS_DIR / f"lstm_{self.symbol}.keras")
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(ckpt_path, save_best_only=True),
        ]
        history = model.fit(
            X_train, y_train,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=0,
        )

        self.lstm_model = model

        # Evaluate on the 1-step-ahead output (first forecast column)
        y_pred_scaled = model.predict(X_test, verbose=0)[:, 0]
        y_pred = self.price_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.price_scaler.inverse_transform(y_test[:, 0].reshape(-1, 1)).flatten()

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)

        directional = np.mean(np.sign(np.diff(y_pred)) == np.sign(np.diff(y_true)))

        print(f"[LSTM-{self.symbol}] MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  Dir={directional:.4f}")
        return {
            "mae": mae, "rmse": rmse, "r2": r2, "directional_accuracy": directional,
            "history": history.history, "y_true": y_true, "y_pred": y_pred,
        }

    # ── Unified train ─────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, train_lstm: bool = True, horizon: int = 10):
        """
        Train all models (RF, XGBoost, and optionally LSTM).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with technical indicators already added.
        train_lstm : bool
            Whether to train LSTM (requires TensorFlow).
        """
        results = {}
        results["rf"]  = self.train_random_forest(df, horizon=horizon)
        results["xgb"] = self.train_xgboost(df, horizon=horizon)
        results["reg"] = self.train_regressor(df, horizon=horizon)
        if train_lstm and TF_AVAILABLE:
            results["lstm"] = self.train_lstm(df)
        self.is_trained = True
        return results

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_direction(self, df: pd.DataFrame, model: str = "xgb"):
        """
        Predict whether the stock will go up (1) or down (0) next day.

        Returns
        -------
        tuple (prediction: int, probability: float)
        """
        X = self._prepare_features(df)
        if len(X) == 0:
            return 0, 0.5

        x_last = X.iloc[[-1]]  # Most recent row

        if model == "rf" and self.rf_model:
            pred  = int(self.rf_model.predict(x_last)[0])
            proba = float(self.rf_model.predict_proba(x_last)[0][1])
        elif model == "xgb" and self.xgb_model:
            pred  = int(self.xgb_model.predict(x_last)[0])
            proba = float(self.xgb_model.predict_proba(x_last)[0][1])
        else:
            raise ValueError(f"Model '{model}' not trained or unrecognised.")

        return pred, proba

    def predict_prices(self, df: pd.DataFrame, horizon: int = 10):
        """
        Forecast future closing prices using LSTM.

        Parameters
        ----------
        horizon : int
            Number of future trading days to forecast.

        Returns
        -------
        np.ndarray of shape (horizon,)
        """
        if not TF_AVAILABLE or self.lstm_model is None:
            raise RuntimeError("LSTM model not available.")

        cfg      = LSTM_CONFIG
        lookback = cfg["lookback"]
        max_h    = cfg.get("horizon", 30)
        horizon  = min(horizon, max_h)
        prices   = df["Close"].values[-lookback:].reshape(-1, 1)
        scaled   = self.price_scaler.transform(prices)

        # Direct multi-horizon: one forward pass predicts every future day at
        # once — no recursive feedback, no compounding errors.
        x_in  = scaled.reshape(1, lookback, 1)
        preds = self.lstm_model.predict(x_in, verbose=0)[0][:horizon]

        preds_unscaled = self.price_scaler.inverse_transform(
            np.array(preds).reshape(-1, 1)
        ).flatten()
        return preds_unscaled

    # ── Feature Importance ────────────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances from the XGBoost model as a DataFrame."""
        if self.xgb_model is None:
            raise RuntimeError("XGBoost not trained yet.")

        importances = self.xgb_model.feature_importances_
        # Use the exact feature names seen at training time to avoid label misalignment
        features = self.trained_features or list(
            getattr(self.xgb_model, "feature_names_in_", self.FEATURE_COLS[:len(importances)])
        )
        n = min(len(features), len(importances))
        fi = pd.DataFrame({
            "Feature":    features[:n],
            "Importance": importances[:n],
        }).sort_values("Importance", ascending=False)
        return fi

    # ── Load saved models ─────────────────────────────────────────────────────

    def load(self):
        """Load previously saved models from disk."""
        rf_path   = MODELS_DIR / f"rf_{self.symbol}.pkl"
        xgb_path  = MODELS_DIR / f"xgb_{self.symbol}.pkl"
        # Support both .keras (new) and .h5 (legacy)
        lstm_path = MODELS_DIR / f"lstm_{self.symbol}.keras"
        if not lstm_path.exists():
            lstm_path = MODELS_DIR / f"lstm_{self.symbol}.h5"

        if rf_path.exists():
            self.rf_model  = joblib.load(rf_path)
        if xgb_path.exists():
            self.xgb_model = joblib.load(xgb_path)
        if lstm_path.exists() and TF_AVAILABLE:
            try:
                self.lstm_model = load_model(str(lstm_path), compile=False)
                # Sanity-check: models saved by a different Keras/TF version can
                # load but fail at inference with unknown-rank shape errors.
                lookback = LSTM_CONFIG["lookback"]
                out = self.lstm_model.predict(
                    np.zeros((1, lookback, 1), dtype=np.float32), verbose=0
                )
                if out.shape[-1] != LSTM_CONFIG.get("horizon", 30):
                    raise ValueError("legacy single-output LSTM — retrain required")
            except Exception:
                print(f"[Predictor] Saved LSTM for {self.symbol} is incompatible "
                      "with this TensorFlow version. Please retrain.")
                self.lstm_model = None
        scaler_path = MODELS_DIR / f"lstm_scaler_{self.symbol}.pkl"
        if scaler_path.exists():
            self.price_scaler = joblib.load(scaler_path)

        self.is_trained = any([self.rf_model, self.xgb_model, self.lstm_model])


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from src.utils.data_loader import DataLoader
    from src.utils.indicators import TechnicalIndicators

    parser = argparse.ArgumentParser(description="Train stock prediction models")
    parser.add_argument("--stock",    default="RELIANCE")
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--train-all",action="store_true", dest="train_all")
    args = parser.parse_args()

    loader = DataLoader()
    ti     = TechnicalIndicators()

    symbols = loader.available_symbols if args.train_all else [args.stock]
    for sym in symbols:
        try:
            df   = loader.load_stock(sym)
            df   = ti.add_all(df)
            pred = StockPredictor(symbol=sym)
            pred.train(df, train_lstm=True)
            print(f"[✓] {sym} trained successfully")
        except Exception as e:
            print(f"[✗] {sym} failed: {e}")
