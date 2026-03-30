"""Run walk-forward backtests for all strategies.

Executes Equal-Weight, Minimum Variance, and Deep Portfolio strategies
using the WalkForwardValidator and saves results to reports/cycle_4/.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.framework import WalkForwardValidator
from src.strategies.benchmarks import EqualWeightStrategy, MinimumVarianceStrategy
from src.strategies.deep_portfolio import DeepPortfolioStrategy


RETURNS_PATH = "data/processed/sp100_daily_returns.csv"
CYCLE = 4
OUTPUT_DIR = f"reports/cycle_{CYCLE}"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "benchmark_returns.json")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")


def compute_strategy_metrics(daily_returns: pd.Series, fee_bps: float = 10, slippage_bps: float = 5) -> dict:
    """Compute performance metrics for a return series."""
    if len(daily_returns) == 0:
        return {"sharpe": 0.0, "annual_return": 0.0, "max_drawdown": 0.0, "volatility": 0.0}

    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std()

    sharpe = mean_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0.0
    annual_return = mean_daily * 252
    annual_vol = std_daily * np.sqrt(252)

    # Max drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # Hit rate
    hit_rate = (daily_returns > 0).mean()

    # Net of costs (approximate: apply cost at each rebalance = monthly)
    total_cost_bps = fee_bps + slippage_bps
    monthly_cost = total_cost_bps / 10000  # per rebalance
    annual_cost = monthly_cost * 12
    net_annual_return = annual_return - annual_cost
    net_sharpe = (net_annual_return / annual_vol) if annual_vol > 0 else 0.0

    return {
        "sharpe": round(sharpe, 4),
        "annual_return": round(annual_return, 4),
        "max_drawdown": round(max_dd, 4),
        "volatility": round(annual_vol, 4),
        "hit_rate": round(hit_rate, 4),
        "net_sharpe": round(net_sharpe, 4),
        "net_annual_return": round(net_annual_return, 4),
        "n_days": len(daily_returns),
    }


def compute_turnover(weights_history: list) -> float:
    """Compute average monthly turnover from weights history."""
    if len(weights_history) < 2:
        return 0.0
    turnovers = []
    for i in range(1, len(weights_history)):
        prev_w = weights_history[i - 1][1]
        curr_w = weights_history[i][1]
        turnover = np.sum(np.abs(curr_w - prev_w))
        turnovers.append(turnover)
    return round(float(np.mean(turnovers)), 4)


def main():
    # Load data
    print("Loading returns data...")
    if not os.path.exists(RETURNS_PATH):
        print(f"Error: {RETURNS_PATH} not found. Run scripts/prepare_data.py first.")
        sys.exit(1)

    returns = pd.read_csv(RETURNS_PATH, index_col=0, parse_dates=True)
    print(f"Data shape: {returns.shape}")
    print(f"Date range: {returns.index.min().date()} to {returns.index.max().date()}")

    # Initialize validator with 60-month train, 1-month test
    validator = WalkForwardValidator(
        returns=returns,
        train_months=60,
        test_months=1,
        rebalance_freq="MS",
    )

    # Define strategies including Deep Portfolio
    strategies = {
        "equal_weight": EqualWeightStrategy(),
        "min_variance": MinimumVarianceStrategy(),
        "deep_portfolio": DeepPortfolioStrategy(
            hidden_dim=128,
            latent_dim=32,
            epochs=100,
            learning_rate=0.001,
            batch_size=64,
            seed=42,
        ),
    }

    results = {}
    all_metrics = {}
    all_turnovers = {}

    for name, strategy in strategies.items():
        print(f"\n{'='*60}")
        print(f"Running {name} strategy...")
        print(f"{'='*60}")
        result = validator.run(strategy)
        daily_rets = result["daily_returns"]
        print(f"  OOS days: {len(daily_rets)}")
        print(f"  Windows: {len(result['windows'])}")

        # Compute metrics
        metrics = compute_strategy_metrics(daily_rets)
        all_metrics[name] = metrics
        all_turnovers[name] = compute_turnover(result["weights_history"])
        print(f"  Sharpe: {metrics['sharpe']:.4f}")
        print(f"  Annual Return: {metrics['annual_return']:.4f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"  Net Sharpe: {metrics['net_sharpe']:.4f}")
        print(f"  Turnover: {all_turnovers[name]:.4f}")

        # Store returns as serializable format
        results[name] = {
            "daily_returns": {
                str(date.date()): round(float(ret), 8)
                for date, ret in daily_rets.items()
            },
            "n_days": len(daily_rets),
            "windows": result["windows"],
        }

    # Save benchmark returns
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark returns saved to {OUTPUT_PATH}")

    # Generate metrics.json — Deep Portfolio is the primary strategy
    dp_metrics = all_metrics["deep_portfolio"]
    ew_metrics = all_metrics["equal_weight"]
    mv_metrics = all_metrics["min_variance"]

    dp_windows = results["deep_portfolio"]["windows"]
    ew_windows = results["equal_weight"]["windows"]
    mv_windows = results["min_variance"]["windows"]

    dp_positive = sum(1 for w in dp_windows if w["sharpe"] > 0)
    ew_positive = sum(1 for w in ew_windows if w["sharpe"] > 0)
    mv_positive = sum(1 for w in mv_windows if w["sharpe"] > 0)

    metrics_json = {
        "sharpeRatio": dp_metrics["sharpe"],
        "annualReturn": dp_metrics["annual_return"],
        "maxDrawdown": dp_metrics["max_drawdown"],
        "hitRate": dp_metrics["hit_rate"],
        "totalTrades": len(dp_windows),
        "transactionCosts": {
            "feeBps": 10,
            "slippageBps": 5,
            "netSharpe": dp_metrics["net_sharpe"],
        },
        "walkForward": {
            "windows": len(dp_windows),
            "positiveWindows": dp_positive,
            "avgOosSharpe": round(
                np.mean([w["sharpe"] for w in dp_windows]), 4
            ) if dp_windows else 0.0,
        },
        "customMetrics": {
            "phase": 4,
            "task": "Deep Portfolio Strategy Implementation",
            # Deep Portfolio metrics
            "deep_portfolio_sharpe": dp_metrics["sharpe"],
            "deep_portfolio_return": dp_metrics["annual_return"],
            "deep_portfolio_drawdown": dp_metrics["max_drawdown"],
            "deep_portfolio_volatility": dp_metrics["volatility"],
            "deep_portfolio_hit_rate": dp_metrics["hit_rate"],
            "deep_portfolio_net_sharpe": dp_metrics["net_sharpe"],
            "deep_portfolio_turnover": all_turnovers["deep_portfolio"],
            "deep_portfolio_n_days": dp_metrics["n_days"],
            "deep_portfolio_windows": len(dp_windows),
            "deep_portfolio_positive_windows": dp_positive,
            # Baseline: Equal Weight
            "baseline_1n_sharpe": ew_metrics["sharpe"],
            "baseline_1n_return": ew_metrics["annual_return"],
            "baseline_1n_drawdown": ew_metrics["max_drawdown"],
            "baseline_1n_volatility": ew_metrics["volatility"],
            "baseline_1n_hit_rate": ew_metrics["hit_rate"],
            "baseline_1n_net_sharpe": ew_metrics["net_sharpe"],
            "baseline_1n_turnover": all_turnovers["equal_weight"],
            "baseline_1n_n_days": ew_metrics["n_days"],
            "baseline_1n_windows": len(ew_windows),
            "baseline_1n_positive_windows": ew_positive,
            # Baseline: Minimum Variance
            "baseline_minvar_sharpe": mv_metrics["sharpe"],
            "baseline_minvar_return": mv_metrics["annual_return"],
            "baseline_minvar_drawdown": mv_metrics["max_drawdown"],
            "baseline_minvar_volatility": mv_metrics["volatility"],
            "baseline_minvar_hit_rate": mv_metrics["hit_rate"],
            "baseline_minvar_net_sharpe": mv_metrics["net_sharpe"],
            "baseline_minvar_turnover": all_turnovers["min_variance"],
            "baseline_minvar_n_days": mv_metrics["n_days"],
            "baseline_minvar_windows": len(mv_windows),
            "baseline_minvar_positive_windows": mv_positive,
            # Strategy vs 1/N comparison
            "strategy_vs_1n_sharpe_diff": round(dp_metrics["sharpe"] - ew_metrics["sharpe"], 4),
            "strategy_vs_1n_return_diff": round(dp_metrics["annual_return"] - ew_metrics["annual_return"], 4),
            "strategy_vs_1n_drawdown_diff": round(dp_metrics["max_drawdown"] - ew_metrics["max_drawdown"], 4),
            "strategy_vs_1n_turnover_ratio": round(
                all_turnovers["deep_portfolio"] / all_turnovers["equal_weight"], 2
            ) if all_turnovers["equal_weight"] > 0 else 0.0,
            # Data summary
            "data_start": str(returns.index.min().date()),
            "data_end": str(returns.index.max().date()),
            "n_assets": returns.shape[1],
            "train_months": 60,
            "test_months": 1,
            "rebalance_freq": "monthly",
            # Model config
            "model_hidden_dim": 128,
            "model_latent_dim": 32,
            "model_epochs": 100,
            "model_learning_rate": 0.001,
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
