"""Deep Portfolio strategy using Autoencoder factor portfolios.

Implements the core idea from 'Deep Portfolio Theory': train an autoencoder
on asset returns, then extract the decoder's last-layer weights as factor
portfolios. The portfolio is constructed by combining these factor portfolios
into a long-only allocation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.autoencoder import Autoencoder


class DeepPortfolioStrategy:
    """Deep Portfolio strategy based on autoencoder decoder weights.

    For each rebalancing window:
    1. Train an autoencoder on the training period's daily returns.
    2. Extract the decoder's last linear layer weights (shape: hidden_dim x n_assets).
       Each row is a 'factor portfolio' mapping a latent factor to asset weights.
    3. Compute the absolute contribution of each factor portfolio and combine them.
    4. Normalize to produce long-only weights summing to 1.

    Args:
        hidden_dim: Hidden layer size for the autoencoder.
        latent_dim: Bottleneck (latent) layer size.
        epochs: Training epochs per window.
        learning_rate: Learning rate for Adam optimizer.
        batch_size: Mini-batch size for training.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed

    def generate_weights(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Train autoencoder and extract portfolio weights from decoder.

        Args:
            returns_data: DataFrame of daily returns (training period only).

        Returns:
            Array of long-only portfolio weights summing to 1.
        """
        n_assets = returns_data.shape[1]
        data_tensor = torch.tensor(returns_data.values, dtype=torch.float32)

        torch.manual_seed(self.seed)

        model = Autoencoder(
            input_dim=n_assets,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Train the autoencoder
        n_samples = data_tensor.shape[0]
        model.train()
        for epoch in range(self.epochs):
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                batch = data_tensor[batch_idx]

                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()

        # Extract decoder last layer weights
        # decoder[-1] is Linear(hidden_dim, n_assets)
        # weight shape: (n_assets, hidden_dim) — PyTorch stores as (out_features, in_features)
        decoder_weights = model.decoder[-1].weight.detach().numpy()
        # decoder_weights shape: (n_assets, hidden_dim)

        # Each column of decoder_weights is a factor portfolio's contribution to assets.
        # Combine by taking the mean absolute weight across factors, which captures
        # how important each asset is across all learned factors.
        # This follows the paper's approach of using decoder weights as factor loadings.
        asset_importance = np.abs(decoder_weights).mean(axis=1)

        # Normalize to long-only weights summing to 1
        weights = asset_importance / asset_importance.sum()

        return weights
