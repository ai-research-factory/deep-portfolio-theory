# Deep Portfolio Theory

> **Note**: This project is a reduced-scale verification using publicly available data, not a full reproduction of the original paper. The implementation serves as a learning exercise and exploratory validation of the core ideas.

This paper proposes using autoencoders and denoising autoencoders to construct investment portfolios. The autoencoder learns a non-linear factor model of asset returns, aiming to create portfolios that are more robust to noise and market shocks compared to traditional methods like minimum variance.

## Current Status — Cycle 2, Phase 2

Data pipeline implemented. S&P 100 daily returns fetched from the ARF Data API.

### Results (from `reports/cycle_2/metrics.json`)

| Metric | Value |
|---|---|
| Phase | 2 — Data Pipeline Construction |
| Tickers Fetched | 82 / 82 |
| Data Rows | 3521 |
| Date Range | 2010-01-05 to 2023-12-29 |
| NaN Values | 0 |
| Config Externalized | Yes |
| Unit Tests Added | Yes |

Portfolio-level metrics (Sharpe, returns, drawdown) are placeholders (0.0) for Phase 2 and will be populated in Phase 3+.

### Cycle 1 Results (from `reports/cycle_1/metrics.json`)

| Metric | Value |
|---|---|
| Training Loss (epoch 1) | 0.339305 |
| Training Loss (epoch 10) | 0.164840 |
| Loss Reduction | 51.4% |
| Architecture | 256 → 128 → 32 → 128 → 256 |

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
  train.py               # Training script (config-driven)
configs/
  default.yaml           # Hyperparameters and data configuration
scripts/
  prepare_data.py        # Data pipeline runner
reports/
  cycle_1/               # Phase 1 results
  cycle_2/               # Phase 2 results
    metrics.json         # ARF standard metrics
    preflight.md         # Pre-implementation checks
    technical_findings.md
tests/
  test_data_integrity.py # Data integrity and leakage tests
  test_models.py         # Autoencoder unit tests
```

## Taxonomy
PCA, ResidualFactors

---
Managed by [AI Research Factory](https://ai.1s.xyz)
