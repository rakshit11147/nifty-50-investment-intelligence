"""
src/models/risk.py
──────────────────
Risk Assessment Module — Mandatory Task C

Computes all required risk metrics for individual stocks or portfolios:
  • Historical Volatility
  • Sharpe Ratio
  • Sortino Ratio
  • Maximum Drawdown
  • Value at Risk (VaR) — Historical & Parametric
  • Conditional VaR (CVaR / Expected Shortfall)
  • Beta vs NIFTY-50 index (if benchmark available)
  • Calmar Ratio
  • Risk-Adjusted Return scoring

HOW TO USE
----------
from src.models.risk import RiskAssessor

ra = RiskAssessor()
metrics = ra.assess_stock(df)           # single stock
pf_metrics = ra.assess_portfolio(port_returns)  # portfolio series
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

try:
    from config.settings import RISK_THRESHOLDS
except ImportError:
    RISK_THRESHOLDS = {
        "sharpe_good":       1.0,
        "sharpe_excellent":  2.0,
        "max_drawdown_warn": 0.20,
        "var_confidence":    0.95,
    }


class RiskAssessor:
    """
    Compute and interpret risk metrics for stocks and portfolios.

    Parameters
    ----------
    risk_free_rate : float   Annual risk-free rate (default 6% — India Gsec)
    trading_days   : int     Trading days per year (default 252)
    var_conf       : float   Confidence level for VaR (default 0.95)
    """

    def __init__(self,
                 risk_free_rate: float = 0.06,
                 trading_days:   int   = 252,
                 var_conf:       float = None):
        self.rfr    = risk_free_rate
        self.td     = trading_days
        self.var_conf = var_conf or RISK_THRESHOLDS.get("var_confidence", 0.95)

    # ── Core metric calculations ───────────────────────────────────────────────

    def volatility(self, returns: pd.Series) -> float:
        """Annualised historical volatility."""
        return float(returns.std() * np.sqrt(self.td))

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """Annualised Sharpe Ratio."""
        ann_ret = float(returns.mean() * self.td)
        ann_vol = self.volatility(returns)
        return (ann_ret - self.rfr) / (ann_vol + 1e-10)

    def sortino_ratio(self, returns: pd.Series) -> float:
        """
        Sortino Ratio — penalises only downside volatility.
        """
        ann_ret    = float(returns.mean() * self.td)
        downside   = returns[returns < 0]
        down_vol   = float(downside.std() * np.sqrt(self.td)) if len(downside) > 0 else 1e-10
        return (ann_ret - self.rfr) / (down_vol + 1e-10)

    def max_drawdown(self, prices_or_cumret: pd.Series) -> float:
        """
        Maximum peak-to-trough drawdown.

        Parameters
        ----------
        prices_or_cumret : pd.Series
            Either raw Close prices or cumulative return series.
        """
        cumulative = prices_or_cumret / prices_or_cumret.iloc[0]
        rolling_max = cumulative.cummax()
        drawdown    = cumulative / rolling_max - 1
        return float(drawdown.min())

    def drawdown_series(self, prices: pd.Series) -> pd.Series:
        """Return the full drawdown time series."""
        cumulative  = prices / prices.iloc[0]
        rolling_max = cumulative.cummax()
        return cumulative / rolling_max - 1

    def value_at_risk(self, returns: pd.Series, method: str = "historical") -> float:
        """
        Daily Value at Risk at self.var_conf confidence level.

        Parameters
        ----------
        method : 'historical' | 'parametric'
        """
        if method == "parametric":
            z    = scipy_stats.norm.ppf(1 - self.var_conf)
            mean = returns.mean()
            std  = returns.std()
            return float(mean + z * std)
        else:  # historical
            return float(np.percentile(returns, (1 - self.var_conf) * 100))

    def cvar(self, returns: pd.Series) -> float:
        """
        Conditional VaR (Expected Shortfall) — mean of losses beyond VaR.
        """
        var   = self.value_at_risk(returns, method="historical")
        tail  = returns[returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    def calmar_ratio(self, returns: pd.Series, prices: pd.Series) -> float:
        """Annual return / |Max Drawdown|."""
        ann_ret = float(returns.mean() * self.td)
        mdd     = abs(self.max_drawdown(prices))
        return ann_ret / (mdd + 1e-10)

    def beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Beta relative to a market/benchmark return series.
        """
        aligned     = pd.concat([stock_returns, market_returns], axis=1).dropna()
        if len(aligned) < 30:
            return np.nan
        cov_matrix  = aligned.cov().values
        return float(cov_matrix[0, 1] / cov_matrix[1, 1])

    def treynor_ratio(self, returns: pd.Series, beta_val: float) -> float:
        """(Portfolio Return - Rf) / Beta"""
        ann_ret = float(returns.mean() * self.td)
        return (ann_ret - self.rfr) / (abs(beta_val) + 1e-10)

    # ── Composite assessment ───────────────────────────────────────────────────

    def assess_stock(self, df: pd.DataFrame,
                     market_returns: pd.Series = None) -> dict:
        """
        Compute all risk metrics for a single stock.

        Parameters
        ----------
        df : pd.DataFrame
            Must have at least 'Close' and 'Daily_Return' columns.
        market_returns : pd.Series, optional
            Market/benchmark daily returns for Beta calculation.

        Returns
        -------
        dict of all risk metrics + a risk_score (0–100) and risk_label
        """
        returns = df["Daily_Return"].dropna()
        prices  = df["Close"]

        if len(returns) < 30:
            return {"error": "Insufficient data (< 30 days)"}

        ann_ret  = float(returns.mean() * self.td)
        vol      = self.volatility(returns)
        sharpe   = self.sharpe_ratio(returns)
        sortino  = self.sortino_ratio(returns)
        mdd      = self.max_drawdown(prices)
        var_h    = self.value_at_risk(returns, "historical")
        var_p    = self.value_at_risk(returns, "parametric")
        cvar_val = self.cvar(returns)
        calmar   = self.calmar_ratio(returns, prices)

        beta_val = np.nan
        treynor  = np.nan
        if market_returns is not None:
            beta_val = self.beta(returns, market_returns)
            treynor  = self.treynor_ratio(returns, beta_val)

        # ── Rolling metrics (1-year window) ───────────────────────────────────
        roll_w    = min(252, len(returns))
        roll_mean = returns.rolling(roll_w).mean() * self.td
        roll_std  = returns.rolling(roll_w).std() * np.sqrt(self.td)
        rolling_sharpe = ((roll_mean - self.rfr) / (roll_std + 1e-10)).dropna()

        # ── Risk score (composite 0–100, higher = riskier) ────────────────────
        # Normalise key risk inputs
        norm_vol  = min(vol / 0.60, 1.0)          # 60% vol = max
        norm_mdd  = min(abs(mdd) / 0.80, 1.0)    # 80% drawdown = max
        norm_var  = min(abs(var_h) / 0.10, 1.0)  # 10% daily VaR = max
        risk_score = norm_vol * 40 + norm_mdd * 40 + norm_var * 20  # weighted sum is already 0-100

        risk_label = (
            "Low"       if risk_score < 30 else
            "Moderate"  if risk_score < 55 else
            "High"      if risk_score < 75 else
            "Very High"
        )

        metrics = {
            # Return
            "annual_return":     round(ann_ret, 4),
            # Risk
            "annual_volatility": round(vol, 4),
            "max_drawdown":      round(mdd, 4),
            "var_95_daily":      round(var_h, 6),
            "var_95_parametric": round(var_p, 6),
            "cvar_95_daily":     round(cvar_val, 6),
            # Ratios
            "sharpe_ratio":      round(sharpe, 4),
            "sortino_ratio":     round(sortino, 4),
            "calmar_ratio":      round(calmar, 4),
            "beta":              round(beta_val, 4) if not np.isnan(beta_val) else None,
            "treynor_ratio":     round(treynor, 4)  if not np.isnan(treynor) else None,
            # Composite
            "risk_score":        round(risk_score, 1),
            "risk_label":        risk_label,
            # Extras
            "positive_days_pct": round(float((returns > 0).mean() * 100), 2),
            "avg_gain":          round(float(returns[returns > 0].mean() * 100), 4),
            "avg_loss":          round(float(returns[returns < 0].mean() * 100), 4),
            "skewness":          round(float(scipy_stats.skew(returns)), 4),
            "kurtosis":          round(float(scipy_stats.kurtosis(returns)), 4),
            # Series for charts
            "_drawdown_series":  self.drawdown_series(prices),
            "_rolling_sharpe":   rolling_sharpe,
        }
        return metrics

    def assess_portfolio(self, port_returns: pd.Series,
                         port_prices:  pd.Series = None,
                         market_returns: pd.Series = None) -> dict:
        """
        Compute all risk metrics for a portfolio return series.

        Parameters
        ----------
        port_returns  : pd.Series  Daily portfolio returns
        port_prices   : pd.Series  Optional; cumulative portfolio value
        market_returns: pd.Series  Optional; market benchmark

        Returns
        -------
        dict of risk metrics
        """
        if port_prices is None:
            port_prices = (1 + port_returns).cumprod()

        df_mock = pd.DataFrame({
            "Close":        port_prices,
            "Daily_Return": port_returns,
        })
        return self.assess_stock(df_mock, market_returns)

    # ── Interpretation helpers ────────────────────────────────────────────────

    def interpret_sharpe(self, sharpe: float) -> str:
        if sharpe < 0:   return "❌ Negative (worse than risk-free)"
        if sharpe < 1.0: return "⚠️  Below average"
        if sharpe < 2.0: return "✅ Good"
        return "🏆 Excellent"

    def interpret_drawdown(self, mdd: float) -> str:
        mdd = abs(mdd)
        if mdd < 0.10: return "✅ Low (< 10%)"
        if mdd < 0.20: return "⚠️  Moderate (10–20%)"
        if mdd < 0.40: return "❌ High (20–40%)"
        return "🔴 Severe (> 40%)"

    def compare_stocks(self, metrics_dict: dict) -> pd.DataFrame:
        """
        Compare risk metrics across multiple stocks.

        Parameters
        ----------
        metrics_dict : dict
            {symbol: metrics_dict_from_assess_stock()}

        Returns
        -------
        pd.DataFrame sorted by Sharpe ratio
        """
        rows = []
        for sym, m in metrics_dict.items():
            if "error" in m:
                continue
            rows.append({
                "Symbol":      sym,
                "Ann Return":  f"{m['annual_return']*100:.1f}%",
                "Volatility":  f"{m['annual_volatility']*100:.1f}%",
                "Sharpe":      round(m["sharpe_ratio"], 2),
                "Sortino":     round(m["sortino_ratio"], 2),
                "Max DD":      f"{m['max_drawdown']*100:.1f}%",
                "VaR 95%":     f"{m['var_95_daily']*100:.2f}%",
                "Risk Score":  m["risk_score"],
                "Risk Label":  m["risk_label"],
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Sharpe", ascending=False)
        return df


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    idx = pd.date_range("2015-01-01", periods=1500, freq="B")
    price = 1000 * np.cumprod(1 + np.random.randn(1500) * 0.012)
    ret   = np.log(price[1:] / price[:-1])

    df_demo = pd.DataFrame({
        "Close":        price,
        "Daily_Return": np.concatenate([[np.nan], ret]),
    }, index=idx)

    ra      = RiskAssessor()
    metrics = ra.assess_stock(df_demo)

    print("── Risk Metrics ──────────────────────────────────")
    for k, v in metrics.items():
        if not k.startswith("_"):
            print(f"  {k:25s}: {v}")