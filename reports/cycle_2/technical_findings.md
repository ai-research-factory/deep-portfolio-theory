# Technical Findings — Cycle 2 (Phase 2: Data Pipeline)

## Task
Build a data pipeline to fetch S&P 100 stock data and compute daily returns.

## Reviewer Feedback Addressed

### 1. Config Externalization (最優先)
- Refactored `src/train.py` to load all hyperparameters from `configs/default.yaml` via PyYAML
- Added model parameters (`hidden_dim`, `latent_dim`), training parameters (`epochs`, `batch_size`, `learning_rate`, `seed`), and data parameters (`api_base_url`, `start_date`, `end_date`) to the config
- No hardcoded values remain in training logic

### 2. Unit Tests (重要)
- Created `tests/test_models.py` with comprehensive Autoencoder tests:
  - Output dimension matching for multiple configurations (default, small, large, single-sample)
  - Encoder latent dimension verification for various latent sizes
  - Training convergence test
  - Gradient flow verification through all parameters

### 3. Preflight (推奨)
- Created `reports/cycle_2/preflight.md` with data boundary table, feature timestamp contract, and paper spec diff

## Implementation

### Data Pipeline (`src/data/loader.py`)
- **Data Source**: ARF Data API (`https://ai.1s.xyz/api/data/ohlcv`)
- **Tickers**: 82 US equities available via the API, covering S&P 100 constituents
- **Date Range**: 2010-01-05 to 2023-12-29 (trading days within 2010-01-01 to 2023-12-31)
- **Returns Calculation**: `pct_change()` on adjusted close prices (backward-looking, no future leakage)
- **NaN Handling**: `ffill().bfill()` after dropping first row (NaN from pct_change)

### Output
- **File**: `data/processed/sp100_daily_returns.csv`
- **Shape**: 3521 rows × 82 columns
- **NaN values**: 0
- **Acceptance criteria**: All met (82 >= 80 assets, 3521 >= 3000 rows, 0 NaN)

## Data Quality Observations
- Mean daily returns are near zero as expected for daily data
- Standard deviations range from ~1% (utilities) to ~3.5% (high-vol tech/crypto-adjacent)
- Max single-day returns reach ~30-50% for some volatile names (e.g., OXY, AMD)
- Min single-day returns show drawdowns up to -52% (OXY during oil crash)

## Limitations
- **Universe**: 82 tickers vs. S&P 100 (paper uses S&P 500/TOPIX). Constrained by ARF API availability. Documented in `docs/open_questions.md`.
- **Data frequency**: Daily returns fetched; paper uses monthly returns for walk-forward. Aggregation to monthly will be done in Phase 3.
- **No survivorship bias correction**: Current universe is based on current constituents, not historical.

## Key Metrics (from metrics.json)
- Phase: 2
- Tickers fetched: 82/82
- Data rows: 3521
- NaN count: 0
- Portfolio-level metrics (Sharpe, returns, drawdown): 0.0 (not yet implemented, Phase 3+)
