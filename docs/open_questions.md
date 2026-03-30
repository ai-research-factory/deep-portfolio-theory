# Open Questions

## Phase 1

1. **Activation function choice**: The paper does not specify exact activation functions. ReLU was used as a reasonable default. May need to experiment with sigmoid or tanh for the encoder output in later phases.
2. **Latent dimension**: 32 factors for 256 inputs (8:1 ratio) is a starting point. The optimal number of factors for S&P 100 (~100 assets) will need tuning in Phase 7.

## Phase 2

3. **Universe size**: The ARF Data API provides 82 US equities (out of ~182 total tickers including indices, ETFs, crypto, and JP stocks). The paper uses S&P 500 (~500 assets). Our reduced universe (82 assets) limits the richness of the learned factor model. Expanding to 500 is not possible with the current API.
4. **Survivorship bias**: The current ticker list is based on present-day constituents. Stocks that were delisted or dropped from S&P 100 during 2010-2023 are not included. This is a known limitation.

## Phase 4

5. **bfill data quality issue (RESOLVED in Cycle 4)**: The `ffill().bfill()` in the data loader was propagating pre-IPO phantom returns (e.g., CRWD +16.48%/day for 2015-2019). Fixed by using `fillna(0.0)` instead. This caused inflated Sharpe ratios in Cycles 1-3.
6. **n < p problem**: With 60 monthly samples and 82 assets, the autoencoder is overparameterized (more weights than training samples). This is a fundamental statistical challenge for the paper's approach with small universes.
7. **Deep Portfolio vs 1/N**: The Deep Portfolio strategy (Sharpe 0.96) slightly underperforms Equal Weight (Sharpe 1.01) on risk-adjusted basis. The higher turnover (20.42%/month) also penalizes net performance. This may improve with:
   - Larger universe (more structure for the AE to learn)
   - Denoising Autoencoder (Phase 6)
   - Factor portfolio selection rather than equal-weight combination
8. **MinVar with monthly covariance**: The covariance matrix from 60 monthly observations for 82 assets is rank-deficient (rank ≤ 60). Higher shrinkage (0.5) is used when n_samples < n_assets, but results may still be suboptimal.
9. **Decoder weight interpretation**: The current approach computes factor_portfolios = W2 @ W1 (full decoder mapping) and equal-weight combines them. Alternative approaches include: (a) selecting top-k factors by variance explained, (b) risk-parity weighting of factors.
