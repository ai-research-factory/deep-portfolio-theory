# Open Questions

## Phase 1

1. **Activation function choice**: The paper does not specify exact activation functions. ReLU was used as a reasonable default. May need to experiment with sigmoid or tanh for the encoder output in later phases.
2. **Latent dimension**: 32 factors for 256 inputs (8:1 ratio) is a starting point. The optimal number of factors for S&P 100 (~100 assets) will need tuning in Phase 7.

## Phase 2

3. **Universe size**: The ARF Data API provides 82 US equities. The paper uses S&P 500 or TOPIX (~500 assets). Our reduced universe (82 assets) may affect the richness of the learned factor model. This is an inherent constraint of the available data source.
4. **Survivorship bias**: The current ticker list is based on present-day constituents. Stocks that were delisted or dropped from S&P 100 during 2010-2023 are not included. This is a known limitation.
5. **Monthly vs. daily returns**: The paper's walk-forward uses monthly returns (60 monthly observations per 5-year window). Phase 2 fetches daily returns; aggregation to monthly will be done in Phase 3.
6. **Some tickers have limited history**: Tickers like PLTR (IPO 2020), SNOW (IPO 2020), CRWD (IPO 2019) have shorter histories. The `ffill().bfill()` fill handles the NaN gap, but these tickers contribute zeros for pre-IPO dates, which may distort early-period factor estimation.
