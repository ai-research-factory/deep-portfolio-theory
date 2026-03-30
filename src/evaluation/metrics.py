"""Performance evaluation metrics for portfolio strategies.

Computes annualized return, annualized volatility, Sharpe ratio,
Sortino ratio, and maximum drawdown from a daily return series.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_return(daily_returns: pd.Series) -> float:
    """Compute annualized return from daily returns."""
    total = (1 + daily_returns).prod()
    n_years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Compute annualized volatility from daily returns."""
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        daily_returns: Series of daily portfolio returns.
        risk_free_rate: Annual risk-free rate (default 0).

    Returns:
        Annualized Sharpe ratio.
    """
    ann_ret = annualized_return(daily_returns)
    ann_vol = annualized_volatility(daily_returns)
    if ann_vol == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / ann_vol)


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sortino ratio (downside deviation only).

    Args:
        daily_returns: Series of daily portfolio returns.
        risk_free_rate: Annual risk-free rate (default 0).

    Returns:
        Annualized Sortino ratio.
    """
    ann_ret = annualized_return(daily_returns)
    downside = daily_returns[daily_returns < 0]
    if len(downside) == 0:
        return float("inf") if ann_ret > 0 else 0.0
    downside_std = float(downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    if downside_std == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / downside_std)


def max_drawdown(daily_returns: pd.Series) -> float:
    """Compute maximum drawdown from daily returns.

    Returns:
        Maximum drawdown as a negative float (e.g., -0.15 for 15% drawdown).
    """
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1
    return float(drawdowns.min())


def compute_all_metrics(daily_returns: pd.Series) -> dict:
    """Compute all five performance metrics for a strategy.

    Args:
        daily_returns: Series of daily portfolio returns.

    Returns:
        Dict with keys: annualized_return, annualized_volatility,
        sharpe_ratio, sortino_ratio, max_drawdown.
    """
    return {
        "annualized_return": round(annualized_return(daily_returns), 4),
        "annualized_volatility": round(annualized_volatility(daily_returns), 4),
        "sharpe_ratio": round(sharpe_ratio(daily_returns), 4),
        "sortino_ratio": round(sortino_ratio(daily_returns), 4),
        "max_drawdown": round(max_drawdown(daily_returns), 4),
    }
