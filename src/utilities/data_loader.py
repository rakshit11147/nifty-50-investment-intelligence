"""
src/utils/data_loader.py
────────────────────────
Loads and preprocesses NIFTY-50 stock CSV files from the data/ directory.

HOW TO USE
----------
from src.utils.data_loader import DataLoader

loader = DataLoader()
df = loader.load_stock("RELIANCE")        # single stock
all_df = loader.load_all_stocks()          # all stocks merged
sectors = loader.get_sector_data("IT")    # all IT sector stocks
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

warnings.filterwarnings("ignore")

# ── Adjust this import path if running from repo root ────────────────────────
try:
    from config.settings import DATA_DIR, NIFTY50_SYMBOLS, SECTOR_MAP
except ImportError:
    from pathlib import Path
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
    NIFTY50_SYMBOLS = []
    SECTOR_MAP = {}


class DataLoader:
    """
    Central data-loading class for the platform.

    The Kaggle dataset has one CSV per stock named <SYMBOL>.csv
    with columns: Date, Symbol, Series, Prev Close, Open, High, Low,
                  Last, Close, VWAP, Volume, Turnover, Trades,
                  Deliverable Volume, %Deliverble
    We normalise to: Date, Open, High, Low, Close, Volume, Turnover
    """

    REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self._cache: dict = {}

    # ── Core loaders ─────────────────────────────────────────────────────────

    def load_stock(self, symbol: str, fill_missing: bool = True) -> pd.DataFrame:
        """
        Load a single stock CSV and return a clean DataFrame indexed by Date.

        Parameters
        ----------
        symbol : str
            NIFTY-50 ticker symbol, e.g. 'RELIANCE'
        fill_missing : bool
            Forward-fill missing trading days

        Returns
        -------
        pd.DataFrame
            Columns: Open, High, Low, Close, Volume, Turnover, Daily_Return
        """
        if symbol in self._cache:
            return self._cache[symbol].copy()

        filepath = self.data_dir / f"{symbol}.csv"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Data file not found: {filepath}\n"
                f"Please download the dataset from Kaggle and place CSVs in {self.data_dir}"
            )

        df = pd.read_csv(filepath, parse_dates=["Date"])
        df = self._clean(df, symbol, fill_missing)
        self._cache[symbol] = df
        return df.copy()

    def load_all_stocks(self, symbols: list = None) -> dict:
        """
        Load multiple stocks and return a dict of {symbol: DataFrame}.

        Parameters
        ----------
        symbols : list, optional
            Subset of symbols to load.  Defaults to all available CSVs.

        Returns
        -------
        dict
        """
        symbols = symbols or self._available_symbols()
        data = {}
        for sym in symbols:
            try:
                data[sym] = self.load_stock(sym)
            except FileNotFoundError:
                pass  # Skip missing files gracefully
        return data

    def load_combined(self, symbols: list = None) -> pd.DataFrame:
        """
        Return a single DataFrame with all stocks' Close prices as columns.
        Useful for correlation analysis and portfolio optimisation.

        Returns
        -------
        pd.DataFrame
            Index = Date, Columns = symbols
        """
        stocks = self.load_all_stocks(symbols)
        closes = {sym: df["Close"] for sym, df in stocks.items()}
        combined = pd.DataFrame(closes)
        combined.index.name = "Date"
        combined = combined.sort_index()
        # Keep each stock's full history; drop only rows where every stock is NaN.
        # (A bare dropna() truncated ALL history to the newest stock's listing date.)
        combined = combined.ffill().dropna(how="all")
        return combined

    def get_sector_data(self, sector: str) -> dict:
        """Return all stocks belonging to a sector as {symbol: DataFrame}."""
        sector_symbols = SECTOR_MAP.get(sector, [])
        return self.load_all_stocks(sector_symbols)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame, symbol: str, fill_missing: bool) -> pd.DataFrame:
        """Standardise columns, sort by date, compute daily returns."""

        # Rename common Kaggle column variants
        rename_map = {
            "date":           "Date",
            "open":           "Open",
            "high":           "High",
            "low":            "Low",
            "close":          "Close",
            "vwap":           "VWAP",
            "volume":         "Volume",
            "turnover":       "Turnover",
            "prev close":     "Prev_Close",
            "prev_close":     "Prev_Close",
            "%deliverble":    "Pct_Deliverable",
            "% deliverble":   "Pct_Deliverable",
        }
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Ensure Date is the index
        if "Date" in df.columns:
            df = df.set_index("Date")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Keep only useful price/volume columns
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "Turnover", "VWAP"] if c in df.columns]
        df = df[keep]

        # Cast to float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where Close is NaN
        df = df.dropna(subset=["Close"])

        # Forward-fill remaining gaps
        if fill_missing:
            df = df.ffill()

        # Add symbol column
        df["Symbol"] = symbol

        # Daily log return
        df["Daily_Return"] = np.log(df["Close"] / df["Close"].shift(1))

        return df

    def _available_symbols(self) -> list:
        """Scan data/ directory for all CSVs and return symbol names."""
        files = glob.glob(str(self.data_dir / "*.csv"))
        symbols = [Path(f).stem for f in files if Path(f).stem not in ("NIFTY50_all", "metadata")]
        return sorted(symbols)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def available_symbols(self) -> list:
        return self._available_symbols()

    def describe_dataset(self) -> pd.DataFrame:
        """Return a summary table of all available stocks (date range, records)."""
        rows = []
        for sym in self.available_symbols:
            try:
                df = self.load_stock(sym)
                rows.append({
                    "Symbol":   sym,
                    "Start":    df.index.min().date(),
                    "End":      df.index.max().date(),
                    "Records":  len(df),
                    "Avg_Close":round(df["Close"].mean(), 2),
                })
            except Exception:
                pass
        return pd.DataFrame(rows)


# ── Demo usage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DataLoader()
    print("Available symbols:", loader.available_symbols[:5], "…")
    try:
        df = loader.load_stock("RELIANCE")
        print(df.tail())
    except FileNotFoundError as e:
        print(f"[Demo] {e}")