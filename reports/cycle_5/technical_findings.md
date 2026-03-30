# Technical Findings — Cycle 5

## Phase 5: Performance Evaluation and Report Generation

### Implementation Summary

1. **`src/evaluation/metrics.py`**: Created a metrics module with five functions:
   - `annualized_return()`: Computes geometric annualized return from daily returns.
   - `annualized_volatility()`: Daily std * sqrt(252).
   - `sharpe_ratio()`: Annualized return / annualized volatility.
   - `sortino_ratio()`: Annualized return / downside deviation (negative returns only).
   - `max_drawdown()`: Peak-to-trough maximum drawdown.
   - `compute_all_metrics()`: Convenience wrapper returning all five metrics.

2. **`scripts/generate_report.py`**: Report generator that:
   - Reads daily returns from `reports/cycle_4/benchmark_returns.json` (all 3 strategies).
   - Reads cycle 3 returns for reference comparison.
   - Computes all metrics via `src/evaluation/metrics.py`.
   - Outputs `reports/cycle_5/performance_summary.md` with Markdown tables.
   - Outputs `reports/cycle_5/metrics.json` in ARF standard schema.

3. **Reviewer feedback addressed**:
   - Added `TestFactorPortfolioWeights` test class verifying paper equation (4): `w_f = W2 @ W1`.
   - Added L2 regularization (`l2_lambda=0.01`) to the Autoencoder model to address `n < p` overfitting.

### Performance Results (from metrics.json)

| Strategy | Sharpe | Sortino | Ann. Return | Ann. Vol | Max DD |
|---|---|---|---|---|---|
| Deep Portfolio (AE) | 0.9366 | 1.1705 | 20.36% | 21.74% | -36.46% |
| Equal Weight (1/N) | 0.9947 | 1.2175 | 19.68% | 19.78% | -34.07% |
| Min Variance | 1.0407 | 1.2721 | 14.31% | 13.75% | -22.72% |

### Defeat Analysis

Deep Portfolio loses to 1/N on:
- **Sharpe**: -0.0581 lower (0.9366 vs 0.9947)
- **Sortino**: -0.047 lower (1.1705 vs 1.2175)
- **Max Drawdown**: -2.39% deeper (-36.46% vs -34.07%)
- **Turnover/Cost**: 20.42%/month vs ~0% — significant cost drag

Deep Portfolio wins on:
- **Raw Return**: +0.68% higher annualized return (20.36% vs 19.68%)

### Cycle 3 vs Cycle 4 Comparison

Cycle 3 results were severely inflated by the `bfill()` phantom returns bug:
- Cycle 3 Equal Weight Sharpe: 3.16 (artificially inflated)
- Cycle 4 Equal Weight Sharpe: 0.99 (corrected)
- Cycle 3 Min Variance Sharpe: 34.97 (absurdly inflated — pre-IPO phantom returns)
- Cycle 4 Min Variance Sharpe: 1.04 (corrected)

This confirms the data fix in Cycle 4 was essential for realistic performance evaluation.

### L2 Regularization (Reviewer Feedback)

Added L2 regularization with `l2_lambda=0.01` to the Autoencoder:
- Penalty term: `lambda * sum(W^2)` for all linear layer weights.
- Rationale: With 60 monthly samples and 82 assets, the autoencoder has more parameters than training samples (n < p problem). L2 regularization constrains weight magnitudes to reduce overfitting.
- The regularization parameter is provisional and will be tuned in Phase 7 (hyperparameter optimization).

### Observations

1. All three Sharpe ratios are in the reasonable range (0.9–1.1), consistent with realistic equity portfolio performance over 2015–2023.
2. The Sortino ratios (1.17–1.27) indicate moderate downside risk management across all strategies.
3. Min Variance achieves the best risk-adjusted performance but at the cost of lower absolute returns.
4. Deep Portfolio's value proposition is higher raw return, but the added volatility and turnover costs negate this advantage on a risk-adjusted basis.
5. All metrics are computed from the same OOS daily return series (2015-03 to 2023-12, 2225 trading days).
