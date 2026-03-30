# Performance Summary — Cycle 5

## Cycle 4 Strategies (Paper-Aligned Monthly Returns)

| Strategy | Ann. Return | Ann. Volatility | Sharpe Ratio | Sortino Ratio | Max Drawdown |
|---|---|---|---|---|---|
| Equal Weight (1/N) | 19.68% | 19.78% | 0.9947 | 1.2175 | -34.07% |
| Min Variance | 14.31% | 13.75% | 1.0407 | 1.2721 | -22.72% |
| Deep Portfolio (AE) | 20.36% | 21.74% | 0.9366 | 1.1705 | -36.46% |

### Key Observations

- **Best Sharpe Ratio**: Min Variance strategy achieves the highest risk-adjusted return.
- **Highest Return**: Deep Portfolio (AE) achieves the highest annualized return but at higher volatility.
- **Lowest Drawdown**: Min Variance has the smallest maximum drawdown.
- **Deep Portfolio vs 1/N**: Deep Portfolio underperforms on Sharpe ratio (higher volatility offsets the return advantage). The 20.42%/month turnover further penalizes net performance.

## Cycle 3 Strategies (Reference — Pre-Data-Fix)

Note: Cycle 3 results are inflated due to the bfill phantom returns bug (fixed in Cycle 4).

| Strategy | Ann. Return | Ann. Volatility | Sharpe Ratio | Sortino Ratio | Max Drawdown |
|---|---|---|---|---|---|
| Equal Weight (1/N) [Cycle 3] | 62.52% | 19.79% | 3.1591 | 3.7279 | -32.97% |
| Min Variance [Cycle 3] | 1450.62% | 41.49% | 34.9663 | 132.5790 | -10.89% |

## Transaction Cost Impact

| Strategy | Gross Sharpe | Est. Annual Cost | Net Sharpe |
|---|---|---|---|
| Equal Weight (1/N) | 0.9947 | ~0.00% (no rebalance) | 0.9947 |
| Min Variance | 1.0407 | ~0.15% (8.39%/mo turnover) | ~1.0307 |
| Deep Portfolio (AE) | 0.9366 | ~0.37% (20.42%/mo turnover) | ~0.8566 |

## Defeat Analysis

Deep Portfolio loses to 1/N on the following metrics:
- **Sharpe Ratio**: Lower risk-adjusted return due to higher volatility
- **Max Drawdown**: Deeper drawdown (-36.46% vs -34.07%)
- **Turnover/Cost**: Significantly higher trading costs

Deep Portfolio wins on:
- **Raw Return**: +0.68% higher annualized return
