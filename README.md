# Deep Portfolio Theory

> **Note**: This project is a reduced-scale verification using publicly available data, not a full reproduction of the original paper. The implementation serves as a learning exercise and exploratory validation of the core ideas.

This paper proposes using autoencoders and denoising autoencoders to construct investment portfolios. The autoencoder learns a non-linear factor model of asset returns, aiming to create portfolios that are more robust to noise and market shocks compared to traditional methods like minimum variance.

## Current Status — Cycle 1, Phase 1

Core Autoencoder model implemented and validated on synthetic data.

### Results (from `reports/cycle_1/metrics.json`)

| Metric | Value |
|---|---|
| Training Loss (epoch 1) | 0.339305 |
| Training Loss (epoch 10) | 0.164840 |
| Loss Reduction | 51.4% |
| Architecture | 256 → 128 → 32 → 128 → 256 |
| Optimizer | Adam (lr=0.001) |

Portfolio-level metrics (Sharpe, returns, drawdown) are placeholders (0.0) for Phase 1 and will be populated in later phases.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Train the autoencoder
python -m src.train

# Run tests
pytest tests/
```

## Project Structure

```
src/
  models/
    autoencoder.py   # Autoencoder model (encoder-decoder with bottleneck)
  train.py           # Training script
reports/
  cycle_1/
    metrics.json     # ARF standard metrics
    loss_log.json    # Per-epoch training loss
    preflight.md     # Pre-implementation checks
    technical_findings.md  # Implementation notes
tests/
  test_data_integrity.py  # Data integrity and leakage tests
```

## Taxonomy
PCA, ResidualFactors

---
Managed by [AI Research Factory](https://ai.1s.xyz)
