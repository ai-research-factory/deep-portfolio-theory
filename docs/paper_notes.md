# Paper Notes

## Deep Portfolio Theory

### Summary
This paper proposes using autoencoders and denoising autoencoders to construct investment portfolios. The autoencoder learns a non-linear factor model of asset returns, aiming to create portfolios that are more robust to noise and market shocks compared to traditional methods like minimum variance.

### Key Strengths
- Proposes a novel, non-linear approach to portfolio construction using autoencoders, moving beyond traditional linear factor models.
- Focuses on risk modeling and robustness (denoising), which is a practical and often overlooked aspect of portfolio management.
- Conceptually simple and can be applied to any market with sufficient historical return data.

### Risks / Concerns
- Autoencoder architecture and hyperparameters are not theoretically grounded and require extensive tuning, risking overfitting.
- The model's performance may be sensitive to the training window and market regime, potentially lacking stability out-of-sample.
- The paper is relatively old (2017); more recent deep learning architectures might offer better performance.

### Implementation Notes
v0.1: Build a simple, single-hidden-layer autoencoder using PyTorch/TensorFlow. Train it on daily returns of TOPIX 100 constituents over a 5-year rolling window. Use the reconstructed (denoised) returns to compute a covariance matrix and then solve for the minimum variance portfolio weights.
v1.0: Compare the performance (Sharpe, max drawdown) of this 'Deep MV' portfolio against benchmarks: 1/N (equal weight), traditional sample covariance MV, and a Ledoit-Wolf shrinkage MV portfolio.
Data needed: Daily OHLCV data for a broad universe of Japanese stocks (e.g., TOPIX 500) for the last 15-20 years.
