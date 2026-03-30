# Open Questions

## Phase 1

1. **Activation function choice**: The paper does not specify exact activation functions. ReLU was used as a reasonable default. May need to experiment with sigmoid or tanh for the encoder output in later phases.
2. **Latent dimension**: 32 factors for 256 inputs (8:1 ratio) is a starting point. The optimal number of factors for S&P 100 (~100 assets) will need tuning in Phase 7.

## Phase 2

3. **Universe size**: The ARF Data API provides 82 US equities. The paper uses S&P 500 or TOPIX (~500 assets). Our reduced universe (82 assets) may affect the richness of the learned factor model. This is an inherent constraint of the available data source.
4. **Survivorship bias**: The current ticker list is based on present-day constituents. Stocks that were delisted or dropped from S&P 100 during 2010-2023 are not included. This is a known limitation.
5. **Monthly vs. daily returns**: The paper's walk-forward uses monthly returns (60 monthly observations per 5-year window). Phase 2 fetches daily returns; aggregation to monthly will be done in Phase 3.
6. **Some tickers have limited history**: Tickers like PLTR (IPO 2020), SNOW (IPO 2020), CRWD (IPO 2019) have shorter histories. The `ffill().bfill()` fill handles the NaN gap, but these tickers contribute zeros for pre-IPO dates, which may distort early-period factor estimation.

## Phase 4

7. **Decoder weight interpretation**: The current approach averages absolute decoder weights across all latent factors to derive portfolio weights. The paper suggests selecting specific factor portfolios or using risk-parity combinations. Alternative weight extraction methods should be explored.
8. **Deep Portfolio vs 1/N**: The Deep Portfolio strategy (Sharpe 2.44) slightly underperforms Equal Weight (Sharpe 2.56). This may be due to (a) the small universe (82 vs 500 assets), (b) using daily rather than monthly returns for training, or (c) the weight extraction method.
9. **Min Variance concentration**: The Minimum Variance strategy produces anomalously high returns (Sharpe ~15.5) due to extreme concentration in a few assets. Adding per-asset weight caps (e.g., max 5%) would produce more realistic results.
10. **Fixed seed across windows**: Using the same random seed (42) for all 107 walk-forward windows means the autoencoder starts from the same initialization each time. This reduces variability but may also limit the model's ability to adapt to different market regimes.
