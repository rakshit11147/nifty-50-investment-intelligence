"""
src/utils/indicators.py
───────────────────────
Compute all technical indicators used as model features.

All functions accept a DataFrame with at least [Open, High, Low, Close, Volume]
and return the same DataFrame with new indicator columns appended.

HOW TO USE
----------
from src.utils.indicators import TechnicalIndicators

ti = TechnicalIndicators()
df_with_features = ti.add_all(df)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from config.settings import TECHNICAL_INDICATORS
except ImportError:
    TECHNICAL_INDICATORS = {
        "sma_windows":  [20, 50, 200],
        "ema_windows":  [12, 26],
        "rsi_period":   14,
        "macd_fast":    12,
        "macd_slow":    26,
        "macd_signal":  9,
        "bb_window":    20,
        "bb_std":       2,
        "atr_period":   14,
        "roc_period":   10,
        "vol_window":   20,
    }


class TechnicalIndicators:
    """Compute and attach technical indicators to a stock DataFrame."""

    def __init__(self, config: dict = None):
        self.cfg = config or TECHNICAL_INDICATORS

    # ── Public API ────────────────────────────────────────────────────────────

    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add every indicator.  Safe to call even if some indicators already exist.
        Returns a new DataFrame (original unchanged).
        """
        df = df.copy()
        df = self.add_moving_averages(df)
        df = self.add_ema(df)
        df = self.add_macd(df)
        df = self.add_rsi(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        df = self.add_obv(df)
        df = self.add_roc(df)
        df = self.add_volatility(df)
        df = self.add_momentum(df)
        df = self.add_price_ratios(df)
        df = df.dropna()
        return df

    # ── Moving Averages ───────────────────────────────────────────────────────

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in self.cfg["sma_windows"]:
            df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
            df[f"Price_vs_SMA{w}"] = df["Close"] / df[f"SMA_{w}"] - 1
        return df

    # ── Exponential Moving Averages ───────────────────────────────────────────

    def add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in self.cfg["ema_windows"]:
            df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
        return df

    # ── MACD ─────────────────────────────────────────────────────────────────

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        fast   = self.cfg["macd_fast"]
        slow   = self.cfg["macd_slow"]
        signal = self.cfg["macd_signal"]

        ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

        df["MACD"]        = ema_fast - ema_slow
        df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
        df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
        return df

    # ── RSI ───────────────────────────────────────────────────────────────────

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.cfg["rsi_period"]
        delta  = df["Close"].diff()
        gain   = delta.clip(lower=0).rolling(window=period).mean()
        loss   = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs     = gain / (loss + 1e-10)
        df["RSI"] = 100 - (100 / (1 + rs))
        return df

    # ── Bollinger Bands ───────────────────────────────────────────────────────

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        w   = self.cfg["bb_window"]
        std = self.cfg["bb_std"]
        mid = df["Close"].rolling(window=w).mean()
        dev = df["Close"].rolling(window=w).std()

        df["BB_Upper"]  = mid + std * dev
        df["BB_Middle"] = mid
        df["BB_Lower"]  = mid - std * dev
        df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
        df["BB_Pct_B"]  = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-10)
        return df

    # ── Average True Range ────────────────────────────────────────────────────

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.cfg["atr_period"]
        prev_close = df["Close"].shift(1)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["ATR"]      = tr.rolling(window=period).mean()
        df["ATR_Pct"]  = df["ATR"] / df["Close"]   # normalised ATR
        return df

    # ── On-Balance Volume ─────────────────────────────────────────────────────

    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Volume" not in df.columns:
            return df
        direction = np.sign(df["Close"].diff()).fillna(0)
        df["OBV"] = (direction * df["Volume"]).cumsum()
        return df

    # ── Rate of Change ────────────────────────────────────────────────────────

    def add_roc(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.cfg["roc_period"]
        df["ROC"] = df["Close"].pct_change(periods=period) * 100
        return df

    # ── Historical Volatility ─────────────────────────────────────────────────

    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self.cfg["vol_window"]
        if "Daily_Return" not in df.columns:
            df["Daily_Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Volatility_20"]  = df["Daily_Return"].rolling(window=w).std() * np.sqrt(252)
        df["Volatility_50"]  = df["Daily_Return"].rolling(window=50).std() * np.sqrt(252)
        return df

    # ── Momentum ──────────────────────────────────────────────────────────────

    def add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Momentum_5"]  = df["Close"] - df["Close"].shift(5)
        df["Momentum_20"] = df["Close"] - df["Close"].shift(20)
        return df

    # ── Price Ratios ──────────────────────────────────────────────────────────

    def add_price_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        df["HL_Ratio"]  = df["High"] / df["Low"]
        df["OC_Ratio"]  = df["Open"] / df["Close"]
        df["Daily_Range"] = (df["High"] - df["Low"]) / df["Close"]
        return df

    # ── Feature list ──────────────────────────────────────────────────────────

    @property
    def feature_columns(self) -> list:
        """Return all indicator column names (used for model input selection)."""
        return [
            "SMA_20", "SMA_50", "SMA_200",
            "Price_vs_SMA20", "Price_vs_SMA50", "Price_vs_SMA200",
            "EMA_12", "EMA_26",
            "MACD", "MACD_Signal", "MACD_Hist",
            "RSI",
            "BB_Upper", "BB_Lower", "BB_Width", "BB_Pct_B",
            "ATR", "ATR_Pct",
            "OBV",
            "ROC",
            "Volatility_20", "Volatility_50",
            "Momentum_5", "Momentum_20",
            "HL_Ratio", "OC_Ratio", "Daily_Range",
        ]


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate synthetic data for testing
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    np.random.seed(42)
    price = 1000 * np.cumprod(1 + np.random.randn(300) * 0.01)
    df = pd.DataFrame({
        "Open":   price * (1 + np.random.randn(300) * 0.002),
        "High":   price * (1 + abs(np.random.randn(300)) * 0.005),
        "Low":    price * (1 - abs(np.random.randn(300)) * 0.005),
        "Close":  price,
        "Volume": np.random.randint(1_000_000, 10_000_000, 300).astype(float),
    }, index=idx)

    ti = TechnicalIndicators()
    df_feat = ti.add_all(df)
    print(f"Features added: {len(ti.feature_columns)}")
    print(df_feat[["Close", "RSI", "MACD", "BB_Pct_B", "Volatility_20"]].tail())