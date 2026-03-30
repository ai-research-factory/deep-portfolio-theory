# Deep Portfolio Theory

> **Note**: This project is a reduced-scale verification using publicly available data, not a full reproduction of the original paper. The implementation serves as a learning exercise and exploratory validation of the core ideas.

This paper proposes using autoencoders and denoising autoencoders to construct investment portfolios. The autoencoder learns a non-linear factor model of asset returns, aiming to create portfolios that are more robust to noise and market shocks compared to traditional methods like minimum variance.

## Current Status — Cycle 4, Phase 4

Deep Portfolio strategy implemented and evaluated via walk-forward backtest against benchmarks.

### Results (from `reports/cycle_4/metrics.json`)

| Strategy | Sharpe | Ann. Return | Max Drawdown | Net Sharpe | Turnover |
|---|---|---|---|---|---|
| **Deep Portfolio (AE)** | 2.4420 | 49.62% | -33.28% | 2.3534 | 3.42% |
| Equal Weight (1/N) | 2.5556 | 50.57% | -32.97% | 2.4646 | 0.00% |
| Min Variance | 15.4913 | 613.70% | -16.38% | 15.4459 | 5.69% |

Walk-forward: 107 windows, 2,244 OOS days (2015-02 to 2023-12). Deep Portfolio positive in 87/107 windows.

Note: Min Variance results are inflated due to extreme concentration — to be addressed in future phases.

### Prior Cycles

| Cycle | Phase | Key Outcome |
|---|---|---|
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

# Train the autoencoder
python -m src.train

# Run walk-forward backtest (all strategies)
python scripts/run_benchmarks.py

# Run tests
pytest tests/
```

## Project Structure

```
src/
  models/
    autoencoder.py       # Autoencoder model (encoder-decoder with bottleneck)
  data/
    loader.py            # S&P 100 data fetching and returns calculation
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
  run_benchmarks.py      # Walk-forward backtest runner
reports/
  cycle_1/               # Phase 1 results
  cycle_2/               # Phase 2 results
  cycle_3/               # Phase 3 results (benchmarks)
  cycle_4/               # Phase 4 results (Deep Portfolio)
tests/
  test_data_integrity.py # Data integrity, leakage, and strategy tests
  test_models.py         # Autoencoder unit tests
```

## Taxonomy
PCA, ResidualFactors

---
Managed by [AI Research Factory](https://ai.1s.xyz)
