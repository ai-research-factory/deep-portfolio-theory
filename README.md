# Deep Portfolio Theory

> **Note**: This project is a reduced-scale verification using publicly available data, not a full reproduction of the original paper. The implementation serves as a learning exercise and exploratory validation of the core ideas.

This paper proposes using autoencoders and denoising autoencoders to construct investment portfolios. The autoencoder learns a non-linear factor model of asset returns, aiming to create portfolios that are more robust to noise and market shocks compared to traditional methods like minimum variance.

## Current Status — Cycle 5, Phase 5

Performance evaluation and reporting phase. Computed comprehensive metrics (Sharpe, Sortino, MaxDD, annualized return/volatility) for all strategies. Added L2 regularization to the autoencoder. Added tests verifying paper equation (4): `w_f = W2 @ W1`.

### Results (from `reports/cycle_5/metrics.json`)

| Strategy | Sharpe | Sortino | Ann. Return | Ann. Vol | Max Drawdown |
|---|---|---|---|---|---|
| **Deep Portfolio (AE)** | 0.9366 | 1.1705 | 20.36% | 21.74% | -36.46% |
| Equal Weight (1/N) | 0.9947 | 1.2175 | 19.68% | 19.78% | -34.07% |
| Min Variance (5% cap) | 1.0407 | 1.2721 | 14.31% | 13.75% | -22.72% |

Walk-forward: 106 windows, 2,225 OOS days (2015-03 to 2023-12). Deep Portfolio positive in 72/106 windows.

Deep Portfolio vs 1/N: Sharpe -0.058 (underperforms), Return +0.68pp (outperforms on raw return).

### Prior Cycles

| Cycle | Phase | Key Outcome |
|---|---|---|
| 4 | Deep Portfolio Strategy | Monthly training, data fix, W2@W1 weight extraction |
| 3 | Walk-Forward & Benchmarks | 1/N and MinVar backtests, 107 windows |
| 2 | Data Pipeline | 82 S&P 100 tickers, 3521 daily returns |
| 1 | Core Autoencoder | Architecture validated, loss reduction 51.4% |

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Prepare data (fetch from ARF Data API)
python scripts/prepare_data.py

# Run Deep Portfolio walk-forward backtest (monthly training)
python scripts/run_deep_portfolio.py

# Run tests
pytest tests/
```

## Project Structure

```
src/
  models/
    autoencoder.py       # Autoencoder model (encoder-decoder with bottleneck)
  data/
    loader.py            # S&P 100 data fetching, daily and monthly returns
  evaluation/
    framework.py         # WalkForwardValidator for OOS backtesting
  strategies/
    benchmarks.py        # Equal-Weight and Minimum Variance strategies
    deep_portfolio.py    # Deep Portfolio strategy (autoencoder-based)
  train.py               # Training script (config-driven)
configs/
  default.yaml           # Hyperparameters and data configuration
scripts/
  prepare_data.py        # Data pipeline runner
  run_benchmarks.py      # Walk-forward backtest runner (legacy)
  run_deep_portfolio.py  # Deep Portfolio backtest with monthly training
  generate_report.py     # Performance evaluation report generator
reports/
  cycle_5/               # Phase 5 results (performance evaluation)
docs/
  paper_spec.md          # Paper parameter specification
  open_questions.md      # Outstanding questions and limitations
tests/
  test_data_integrity.py # Data integrity, leakage, and strategy tests
  test_models.py         # Autoencoder unit tests
```

## Taxonomy
PCA, ResidualFactors

---
Managed by [AI Research Factory](https://ai.1s.xyz)
