# Paper Specification — Deep Portfolio Theory

## Paper Reference
Heaton, Polson, Witte (2016): "Deep Portfolio Theory"

## Key Parameters

| Parameter | Paper Value | Notes |
|---|---|---|
| Universe | S&P 500 constituents | ~500 large-cap US equities |
| Return frequency | Monthly | Monthly log/arithmetic returns |
| Lookback window | 60 months (5 years) | Training period for each WF window |
| Rebalance frequency | Monthly | Re-estimate weights each month |
| Model | Autoencoder (AE) and Denoising AE (DAE) | Encoder-Decoder with bottleneck |
| Bottleneck size | Variable (paper tests multiple) | Latent factors = learned market structure |
| Loss function | MSE (reconstruction error) | Standard autoencoder loss |
| Weight extraction | Decoder first layer weights | Each row = factor portfolio |
| Portfolio construction | Equal-weight combination of factor portfolios | Normalize to sum=1, long-only |

## Decoder Weight Interpretation

The paper extracts portfolio weights from the **first layer of the decoder** (decoder[0]):
- Shape: (hidden_dim, n_assets) in PyTorch convention, or equivalently each column maps a latent factor to asset weights
- Each row of decoder[0].weight.T represents a "factor portfolio"
- Factor portfolios are combined with equal weights
- Final weights are normalized to be long-only and sum to 1

## Walk-Forward Protocol

1. Use 60 months of historical **monthly** returns as training data
2. Train autoencoder to minimize reconstruction error
3. Extract decoder weights as portfolio weights
4. Hold portfolio for 1 month (out-of-sample)
5. Roll forward and repeat

## Implementation Deviations

| Deviation | Reason | Impact |
|---|---|---|
| 82 assets vs 500 | ARF API provides ~84 US equities | Reduced factor diversity |
| Fixed architecture | Paper tests multiple; we use 128→32 | Simplified; may miss optimal config |
