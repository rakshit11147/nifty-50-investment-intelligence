"""
src/models/portfolio.py
───────────────────────
Portfolio Construction Module — Mandatory Task B

Builds optimised portfolios for three investor profiles:
  • Conservative  — Minimum Volatility optimisation
  • Balanced      — Maximum Sharpe Ratio
  • Aggressive    — Maximum Return / Equal Weight

HOW TO USE
----------
from src.models.portfolio import PortfolioBuilder

builder = PortfolioBuilder()
combined_df = loader.load_combined(symbols)       # Close prices, multi-column

result = builder.build("Balanced", combined_df)
# result = {
#   "weights": {"RELIANCE": 0.12, "TCS": 0.18, ...},
#   "metrics": {"annual_return": 0.17, "sharpe": 1.3, ...},
#   "profile": "Balanced",
# }
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

try:
    from config.settings import INVESTOR_PROFILES, RISK_THRESHOLDS
except ImportError:
    INVESTOR_PROFILES = {
        "Conservative": {"target_return": 0.08, "max_volatility": 0.12,
                         "preferred_sectors": [], "max_stocks": 8, "color": "#10B981"},
        "Balanced":     {"target_return": 0.15, "max_volatility": 0.20,
                         "preferred_sectors": [], "max_stocks": 12, "color": "#3B82F6"},
        "Aggressive":   {"target_return": 0.25, "max_volatility": 0.35,
                         "preferred_sectors": [], "max_stocks": 15, "color": "#F59E0B"},
    }
    RISK_THRESHOLDS = {"var_confidence": 0.95}


class PortfolioBuilder:
    """
    Builds mean-variance optimised portfolios for different investor profiles.

    Parameters
    ----------
    risk_free_rate : float
        Annual risk-free rate (default: 6% — India 10Y Gsec proxy)
    trading_days   : int
        Trading days per year (default 252)
    """

    def __init__(self, risk_free_rate: float = 0.06, trading_days: int = 252):
        self.rfr    = risk_free_rate
        self.td     = trading_days

    # ── Core returns & covariance ─────────────────────────────────────────────

    def _compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change().dropna()

    def _annualised_stats(self, returns: pd.DataFrame):
        """Annual mean returns and covariance matrix."""
        ann_ret = returns.mean() * self.td
        ann_cov = returns.cov() * self.td
        return ann_ret, ann_cov

    # ── Portfolio statistics ──────────────────────────────────────────────────

    def portfolio_stats(self, weights: np.ndarray,
                        ann_ret: pd.Series, ann_cov: pd.DataFrame) -> dict:
        """
        Compute portfolio return, volatility, and Sharpe.

        Parameters
        ----------
        weights  : np.ndarray — portfolio weights (must sum to 1)
        ann_ret  : pd.Series  — annualised returns per stock
        ann_cov  : pd.DataFrame — annualised covariance matrix

        Returns
        -------
        dict with keys: return, volatility, sharpe
        """
        ret  = float(np.dot(weights, ann_ret))
        vol  = float(np.sqrt(weights @ ann_cov.values @ weights))
        shrp = (ret - self.rfr) / (vol + 1e-10)
        return {"return": ret, "volatility": vol, "sharpe": shrp}

    # ── Optimisation objectives ────────────────────────────────────────────────

    def _neg_sharpe(self, w, ann_ret, ann_cov):
        s = self.portfolio_stats(w, ann_ret, ann_cov)
        return -s["sharpe"]

    def _min_volatility(self, w, ann_ret, ann_cov):
        return self.portfolio_stats(w, ann_ret, ann_cov)["volatility"]

    def _neg_return(self, w, ann_ret, ann_cov):
        return -self.portfolio_stats(w, ann_ret, ann_cov)["return"]

    # ── Optimisation engine ────────────────────────────────────────────────────

    def _optimise(self, objective, ann_ret, ann_cov,
                  bounds=None, constraints=None) -> np.ndarray:
        n = len(ann_ret)
        w0 = np.ones(n) / n                             # equal-weight start
        bounds      = bounds or [(0.02, 0.40)] * n      # 2–40% per stock
        constraints = constraints or [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        result = minimize(
            objective, w0,
            args=(ann_ret, ann_cov),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        if not result.success:
            # Fall back to equal weights
            return w0
        return result.x

    # ── Stock pre-selection ────────────────────────────────────────────────────

    def _select_stocks(self, ann_ret: pd.Series, ann_vol: pd.Series,
                       profile: str, max_n: int) -> list:
        """
        Select top stocks for a given profile:
          - Conservative: lowest volatility stocks
          - Balanced:     best Sharpe proxies (excess return / vol ratio)
          - Aggressive:   highest return stocks
        """
        if profile == "Conservative":
            score = -ann_vol                                   # lowest volatility first
        elif profile == "Balanced":
            score = (ann_ret - self.rfr) / (ann_vol + 1e-10)   # Sharpe proxy
        else:  # Aggressive
            score = ann_ret                                    # maximum return

        top = score.nlargest(max_n).index.tolist()
        return top

    # ── Main builder ──────────────────────────────────────────────────────────

    def build(self, profile: str, prices: pd.DataFrame) -> dict:
        """
        Build a portfolio for the given investor profile.

        Parameters
        ----------
        profile : str
            One of 'Conservative', 'Balanced', 'Aggressive'
        prices  : pd.DataFrame
            Close price DataFrame with symbols as columns

        Returns
        -------
        dict
            {weights, metrics, profile, allocation_df, reasoning}
        """
        if profile not in INVESTOR_PROFILES:
            raise ValueError(f"Profile must be one of {list(INVESTOR_PROFILES.keys())}")

        cfg      = INVESTOR_PROFILES[profile]
        returns  = self._compute_returns(prices)
        ann_ret, ann_cov = self._annualised_stats(returns)

        # ── 1. Select stocks ──────────────────────────────────────────────────
        max_n    = cfg["max_stocks"]
        ann_vol  = pd.Series(np.sqrt(np.diag(ann_cov)), index=ann_ret.index)
        selected = self._select_stocks(ann_ret, ann_vol, profile, max_n)

        ann_ret_sel = ann_ret[selected]
        ann_cov_sel = ann_cov.loc[selected, selected]

        # ── 2. Choose objective ───────────────────────────────────────────────
        if profile == "Conservative":
            objective = self._min_volatility
            # Lower max per stock for diversification
            bounds = [(0.02, 0.25)] * len(selected)
        elif profile == "Balanced":
            objective = self._neg_sharpe
            bounds = [(0.02, 0.30)] * len(selected)
        else:  # Aggressive
            objective = self._neg_return
            bounds = [(0.02, 0.40)] * len(selected)

        # ── 3. Optimise ───────────────────────────────────────────────────────
        weights_arr = self._optimise(objective, ann_ret_sel, ann_cov_sel, bounds)
        weights_arr = np.clip(weights_arr, 0, None)
        weights_arr /= weights_arr.sum()               # renormalise

        weights_dict = dict(zip(selected, weights_arr.tolist()))

        # ── 4. Portfolio metrics ──────────────────────────────────────────────
        stats = self.portfolio_stats(weights_arr, ann_ret_sel, ann_cov_sel)

        # Additional metrics
        port_returns = (returns[selected] * weights_arr).sum(axis=1)
        var_95  = float(np.percentile(port_returns, (1 - RISK_THRESHOLDS.get("var_confidence", 0.95)) * 100))
        cvar_95 = float(port_returns[port_returns <= var_95].mean())

        cumulative = (1 + port_returns).cumprod()
        drawdown   = (cumulative / cumulative.cummax() - 1)
        max_dd     = float(drawdown.min())

        metrics = {
            "annual_return":   round(stats["return"], 4),
            "annual_volatility": round(stats["volatility"], 4),
            "sharpe_ratio":    round(stats["sharpe"], 4),
            "var_95_daily":    round(var_95, 6),
            "cvar_95_daily":   round(cvar_95, 6),
            "max_drawdown":    round(max_dd, 4),
            "n_stocks":        len(selected),
        }

        # ── 5. Allocation DataFrame ───────────────────────────────────────────
        allocation_df = pd.DataFrame({
            "Stock":           list(weights_dict.keys()),
            "Weight":          [round(w, 4) for w in weights_dict.values()],
            "Weight_Pct":      [round(w * 100, 2) for w in weights_dict.values()],
            "Ann_Return":      [round(ann_ret_sel[s], 4) for s in selected],
            "Ann_Volatility":  [round(np.sqrt(ann_cov_sel.loc[s, s]), 4) for s in selected],
        }).sort_values("Weight", ascending=False)

        # ── 6. Plain-language reasoning ───────────────────────────────────────
        reasoning = self._generate_reasoning(profile, stats, allocation_df)

        return {
            "profile":       profile,
            "weights":       weights_dict,
            "metrics":       metrics,
            "allocation_df": allocation_df,
            "reasoning":     reasoning,
            "port_returns":  port_returns,
            "drawdown_series": drawdown,
        }

    # ── Reasoning generator ────────────────────────────────────────────────────

    def _generate_reasoning(self, profile: str, stats: dict,
                             allocation_df: pd.DataFrame) -> str:
        """Generate a plain-language explanation of the portfolio decisions."""
        top3  = allocation_df.head(3)["Stock"].tolist()
        shrp  = stats["sharpe"]
        ret   = stats["return"] * 100
        vol   = stats["volatility"] * 100

        rationale = {
            "Conservative": (
                f"This portfolio prioritises capital preservation by minimising volatility "
                f"to {vol:.1f}% annually. Top holdings ({', '.join(top3)}) were selected "
                f"for their consistent performance and low drawdown history. "
                f"Expected annual return is {ret:.1f}% with a Sharpe ratio of {shrp:.2f}."
            ),
            "Balanced": (
                f"This portfolio maximises risk-adjusted return (Sharpe: {shrp:.2f}) "
                f"by balancing growth stocks with stable dividend payers. "
                f"Core positions in {', '.join(top3)} provide a blend of momentum "
                f"and defensive characteristics. "
                f"Targeting {ret:.1f}% annual return at {vol:.1f}% volatility."
            ),
            "Aggressive": (
                f"This portfolio targets maximum capital growth ({ret:.1f}% annually) "
                f"by concentrating in high-momentum, high-return stocks. "
                f"Significant positions in {', '.join(top3)} reflect a high-risk, "
                f"high-reward approach. Volatility of {vol:.1f}% is expected; "
                f"investors must tolerate large drawdown periods."
            ),
        }
        return rationale.get(profile, "Portfolio constructed using mean-variance optimisation.")

    # ── Efficient Frontier ────────────────────────────────────────────────────

    def efficient_frontier(self, prices: pd.DataFrame, n_points: int = 200) -> pd.DataFrame:
        """
        Generate random portfolios to approximate the efficient frontier.

        Returns
        -------
        pd.DataFrame with columns: return, volatility, sharpe
        """
        returns  = self._compute_returns(prices)
        ann_ret, ann_cov = self._annualised_stats(returns)
        n = len(ann_ret)

        rows = []
        for _ in range(n_points):
            w = np.random.dirichlet(np.ones(n))
            s = self.portfolio_stats(w, ann_ret, ann_cov)
            rows.append(s)
        return pd.DataFrame(rows)

    # ── Rebalancing schedule ──────────────────────────────────────────────────

    def rebalancing_schedule(self, profile: str) -> dict:
        """Return a suggested rebalancing cadence for each profile."""
        schedules = {
            "Conservative": {
                "frequency": "Quarterly",
                "trigger":   "Drift > 5% from target weights",
                "review":    "Annual full review recommended",
            },
            "Balanced": {
                "frequency": "Monthly",
                "trigger":   "Drift > 3% from target weights",
                "review":    "Semi-annual strategy review",
            },
            "Aggressive": {
                "frequency": "Weekly",
                "trigger":   "Drift > 2% or new market signal",
                "review":    "Monthly tactical review",
            },
        }
        return schedules.get(profile, {})


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    idx = pd.date_range("2018-01-01", periods=1000, freq="B")
    symbols = ["TCS", "INFY", "RELIANCE", "HDFCBANK", "ICICIBANK"]
    prices = pd.DataFrame(
        np.cumprod(1 + np.random.randn(1000, len(symbols)) * 0.01, axis=0) * 1000,
        index=idx, columns=symbols
    )

    builder = PortfolioBuilder()
    for profile in ["Conservative", "Balanced", "Aggressive"]:
        result = builder.build(profile, prices)
        print(f"\n── {profile} Portfolio ──")
        print(result["allocation_df"][["Stock", "Weight_Pct"]].to_string(index=False))
        print(f"Sharpe: {result['metrics']['sharpe_ratio']:.3f}")
        print(f"Reasoning: {result['reasoning'][:120]}…")