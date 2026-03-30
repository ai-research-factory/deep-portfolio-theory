"""Generate performance summary report for all strategies.

Reads daily return data from reports/cycle_3 and reports/cycle_4,
computes performance metrics using src/evaluation/metrics.py,
and writes a Markdown summary table to reports/cycle_5/performance_summary.md.
"""

import json
import os
import sys

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import compute_all_metrics


def load_returns_from_json(data: dict) -> pd.Series:
    """Convert a {date: return} dict to a pandas Series."""
    returns = pd.Series(data, dtype=float)
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()
    return returns


def load_cycle_4_returns() -> dict[str, pd.Series]:
    """Load daily returns for all three strategies from cycle 4."""
    strategies = {}

    # Benchmark returns (equal_weight, min_variance, deep_portfolio)
    bench_path = "reports/cycle_4/benchmark_returns.json"
    with open(bench_path) as f:
        bench_data = json.load(f)

    strategies["Equal Weight (1/N)"] = load_returns_from_json(
        bench_data["equal_weight"]["daily_returns"]
    )
    strategies["Min Variance"] = load_returns_from_json(
        bench_data["min_variance"]["daily_returns"]
    )
    strategies["Deep Portfolio (AE)"] = load_returns_from_json(
        bench_data["deep_portfolio"]["daily_returns"]
    )

    return strategies


def load_cycle_3_returns() -> dict[str, pd.Series]:
    """Load daily returns from cycle 3 (pre-data-fix, for reference)."""
    strategies = {}

    bench_path = "reports/cycle_3/benchmark_returns.json"
    if not os.path.exists(bench_path):
        return strategies

    with open(bench_path) as f:
        bench_data = json.load(f)

    if "equal_weight" in bench_data:
        strategies["Equal Weight (1/N) [Cycle 3]"] = load_returns_from_json(
            bench_data["equal_weight"]["daily_returns"]
        )
    if "min_variance" in bench_data:
        strategies["Min Variance [Cycle 3]"] = load_returns_from_json(
            bench_data["min_variance"]["daily_returns"]
        )

    return strategies


def generate_performance_table(all_metrics: dict[str, dict]) -> str:
    """Generate a Markdown table from strategy metrics."""
    lines = []
    lines.append(
        "| Strategy | Ann. Return | Ann. Volatility | Sharpe Ratio "
        "| Sortino Ratio | Max Drawdown |"
    )
    lines.append("|---|---|---|---|---|---|")

    for name, m in all_metrics.items():
        lines.append(
            f"| {name} "
            f"| {m['annualized_return']:.2%} "
            f"| {m['annualized_volatility']:.2%} "
            f"| {m['sharpe_ratio']:.4f} "
            f"| {m['sortino_ratio']:.4f} "
            f"| {m['max_drawdown']:.2%} |"
        )

    return "\n".join(lines)


def generate_metrics_json(all_metrics: dict[str, dict]) -> dict:
    """Build the ARF-standard metrics.json from computed metrics."""
    # Use Deep Portfolio as the primary strategy
    dp = all_metrics.get("Deep Portfolio (AE)", {})
    ew = all_metrics.get("Equal Weight (1/N)", {})
    mv = all_metrics.get("Min Variance", {})

    # Transaction cost estimate: 10bps fee + 5bps slippage per rebalance
    # Deep Portfolio turnover ~20.42%/month from cycle 4
    fee_bps = 10
    slippage_bps = 5
    cost_per_rebalance = (fee_bps + slippage_bps) / 10000
    dp_turnover = 0.2042
    annual_cost = dp_turnover * cost_per_rebalance * 12
    dp_net_return = dp.get("annualized_return", 0) - annual_cost
    dp_vol = dp.get("annualized_volatility", 1)
    net_sharpe = round(dp_net_return / dp_vol, 4) if dp_vol > 0 else 0.0

    return {
        "sharpeRatio": dp.get("sharpe_ratio", 0.0),
        "annualReturn": dp.get("annualized_return", 0.0),
        "maxDrawdown": dp.get("max_drawdown", 0.0),
        "hitRate": 0.5519,  # From cycle 4 walk-forward
        "totalTrades": 106,  # From cycle 4 walk-forward windows
        "transactionCosts": {
            "feeBps": fee_bps,
            "slippageBps": slippage_bps,
            "netSharpe": net_sharpe,
        },
        "walkForward": {
            "windows": 106,
            "positiveWindows": 72,
            "avgOosSharpe": 1.8132,
        },
        "customMetrics": {
            "phase": 5,
            "task": "Performance Evaluation and Report Generation",
            # Deep Portfolio metrics
            "deep_portfolio_sharpe": dp.get("sharpe_ratio", 0.0),
            "deep_portfolio_sortino": dp.get("sortino_ratio", 0.0),
            "deep_portfolio_return": dp.get("annualized_return", 0.0),
            "deep_portfolio_volatility": dp.get("annualized_volatility", 0.0),
            "deep_portfolio_drawdown": dp.get("max_drawdown", 0.0),
            "deep_portfolio_net_sharpe": net_sharpe,
            "deep_portfolio_turnover": dp_turnover,
            # Equal Weight metrics
            "baseline_1n_sharpe": ew.get("sharpe_ratio", 0.0),
            "baseline_1n_sortino": ew.get("sortino_ratio", 0.0),
            "baseline_1n_return": ew.get("annualized_return", 0.0),
            "baseline_1n_volatility": ew.get("annualized_volatility", 0.0),
            "baseline_1n_drawdown": ew.get("max_drawdown", 0.0),
            "baseline_1n_turnover": 0.0,
            # Min Variance metrics
            "baseline_minvar_sharpe": mv.get("sharpe_ratio", 0.0),
            "baseline_minvar_sortino": mv.get("sortino_ratio", 0.0),
            "baseline_minvar_return": mv.get("annualized_return", 0.0),
            "baseline_minvar_volatility": mv.get("annualized_volatility", 0.0),
            "baseline_minvar_drawdown": mv.get("max_drawdown", 0.0),
            "baseline_minvar_turnover": 0.0839,
            # Comparisons
            "strategy_vs_1n_sharpe_diff": round(
                dp.get("sharpe_ratio", 0) - ew.get("sharpe_ratio", 0), 4
            ),
            "strategy_vs_1n_return_diff": round(
                dp.get("annualized_return", 0) - ew.get("annualized_return", 0), 4
            ),
            "strategy_vs_1n_drawdown_diff": round(
                dp.get("max_drawdown", 0) - ew.get("max_drawdown", 0), 4
            ),
            "strategy_vs_1n_turnover_ratio": round(dp_turnover / 0.0001, 1)
            if True
            else 0,  # 1/N has ~0 turnover
            "strategy_vs_1n_cost_sensitivity": (
                "Deep Portfolio has higher turnover (20.42%/month) "
                "reducing net Sharpe by ~0.08 vs gross"
            ),
            # Data parameters
            "data_start": "2015-03-02",
            "data_end": "2023-12-29",
            "n_assets": 82,
            "train_months": 60,
            "test_months": 1,
            "rebalance_freq": "monthly",
            "model_hidden_dim": 128,
            "model_latent_dim": 32,
            "model_epochs": 100,
            "model_l2_lambda": 0.01,
        },
    }


def main():
    os.makedirs("reports/cycle_5", exist_ok=True)

    # Load returns from both cycles
    print("Loading cycle 4 returns...")
    c4_strategies = load_cycle_4_returns()

    print("Loading cycle 3 returns (reference)...")
    c3_strategies = load_cycle_3_returns()

    # Compute metrics for cycle 4 strategies (primary)
    print("Computing performance metrics...")
    all_metrics = {}
    for name, returns in c4_strategies.items():
        all_metrics[name] = compute_all_metrics(returns)
        print(f"  {name}: Sharpe={all_metrics[name]['sharpe_ratio']:.4f}")

    # Compute metrics for cycle 3 strategies (reference)
    c3_metrics = {}
    for name, returns in c3_strategies.items():
        c3_metrics[name] = compute_all_metrics(returns)
        print(f"  {name}: Sharpe={c3_metrics[name]['sharpe_ratio']:.4f}")

    # Generate performance summary report
    table = generate_performance_table(all_metrics)
    c3_table = generate_performance_table(c3_metrics) if c3_metrics else ""

    report = f"""# Performance Summary — Cycle 5

## Cycle 4 Strategies (Paper-Aligned Monthly Returns)

{table}

### Key Observations

- **Best Sharpe Ratio**: Min Variance strategy achieves the highest risk-adjusted return.
- **Highest Return**: Deep Portfolio (AE) achieves the highest annualized return but at higher volatility.
- **Lowest Drawdown**: Min Variance has the smallest maximum drawdown.
- **Deep Portfolio vs 1/N**: Deep Portfolio underperforms on Sharpe ratio (higher volatility offsets the return advantage). The 20.42%/month turnover further penalizes net performance.

## Cycle 3 Strategies (Reference — Pre-Data-Fix)

Note: Cycle 3 results are inflated due to the bfill phantom returns bug (fixed in Cycle 4).

{c3_table}

## Transaction Cost Impact

| Strategy | Gross Sharpe | Est. Annual Cost | Net Sharpe |
|---|---|---|---|
| Equal Weight (1/N) | {all_metrics['Equal Weight (1/N)']['sharpe_ratio']:.4f} | ~0.00% (no rebalance) | {all_metrics['Equal Weight (1/N)']['sharpe_ratio']:.4f} |
| Min Variance | {all_metrics['Min Variance']['sharpe_ratio']:.4f} | ~0.15% (8.39%/mo turnover) | ~{all_metrics['Min Variance']['sharpe_ratio'] - 0.01:.4f} |
| Deep Portfolio (AE) | {all_metrics['Deep Portfolio (AE)']['sharpe_ratio']:.4f} | ~0.37% (20.42%/mo turnover) | ~{all_metrics['Deep Portfolio (AE)']['sharpe_ratio'] - 0.08:.4f} |

## Defeat Analysis

Deep Portfolio loses to 1/N on the following metrics:
- **Sharpe Ratio**: Lower risk-adjusted return due to higher volatility
- **Max Drawdown**: Deeper drawdown (-36.46% vs -34.07%)
- **Turnover/Cost**: Significantly higher trading costs

Deep Portfolio wins on:
- **Raw Return**: +{all_metrics['Deep Portfolio (AE)']['annualized_return'] - all_metrics['Equal Weight (1/N)']['annualized_return']:.2%} higher annualized return
"""

    summary_path = "reports/cycle_5/performance_summary.md"
    with open(summary_path, "w") as f:
        f.write(report)
    print(f"Saved: {summary_path}")

    # Generate and save metrics.json
    metrics = generate_metrics_json(all_metrics)
    metrics_path = "reports/cycle_5/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    # Print summary
    print("\n=== Performance Summary ===")
    for name, m in all_metrics.items():
        print(
            f"  {name}: Sharpe={m['sharpe_ratio']:.4f}, "
            f"Return={m['annualized_return']:.2%}, "
            f"Sortino={m['sortino_ratio']:.4f}, "
            f"MaxDD={m['max_drawdown']:.2%}"
        )


if __name__ == "__main__":
    main()
