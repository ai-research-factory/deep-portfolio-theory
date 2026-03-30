# Open Questions

## Phase 1

1. **Activation function choice**: The paper does not specify exact activation functions. ReLU was used as a reasonable default. May need to experiment with sigmoid or tanh for the encoder output in later phases.
2. **Latent dimension**: 32 factors for 256 inputs (8:1 ratio) is a starting point. The optimal number of factors for S&P 100 (~100 assets) will need tuning in Phase 7.
3. **Data universe**: The paper targets S&P 100. The ARF Data API's available tickers need to be checked in Phase 2 to confirm coverage.
