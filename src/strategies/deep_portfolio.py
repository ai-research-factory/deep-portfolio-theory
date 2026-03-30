"""Deep Portfolio strategy using Autoencoder factor portfolios.

Implements the core idea from 'Deep Portfolio Theory': train an autoencoder
on asset returns, then extract the decoder's first-layer weights as factor
portfolios. The portfolio is constructed by equal-weight combining these
factor portfolios into a long-only allocation.
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.autoencoder import Autoencoder

logger = logging.getLogger(__name__)


class DeepPortfolioStrategy:
    """Deep Portfolio strategy based on autoencoder decoder weights.

    For each rebalancing window:
    1. Train an autoencoder on the training period's returns (monthly).
    2. Extract the decoder's first linear layer weights (decoder[0]).
       Shape: (hidden_dim, latent_dim) — each column is a factor portfolio
       mapping from latent space to hidden space, but we need the mapping
       from latent space to asset space. Per the paper, we use decoder[0].weight
       which has shape (hidden_dim, latent_dim).
    3. Combine factor portfolios with equal weights.
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
        l2_lambda: float = 0.01,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.l2_lambda = l2_lambda

    def generate_weights(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Train autoencoder and extract portfolio weights from decoder.

        Per the paper, portfolio weights are derived from the decoder's first
        layer (decoder[0]), which maps latent factors to the hidden layer.
        Combined with decoder's second layer, this gives factor portfolios
        in asset space.

        Args:
            returns_data: DataFrame of returns (training period only).

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
            l2_lambda=self.l2_lambda,
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Train the autoencoder
        n_samples = data_tensor.shape[0]
        model.train()
        final_loss = 0.0
        for epoch in range(self.epochs):
            indices = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                batch = data_tensor[batch_idx]

                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch) + model.l2_penalty()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            final_loss = epoch_loss / max(n_batches, 1)

        logger.info(
            f"AE trained on {n_samples} samples, {n_assets} assets, "
            f"final loss={final_loss:.6f}"
        )

        # Extract factor portfolios from decoder
        # decoder structure: Linear(latent, hidden) -> ReLU -> Linear(hidden, n_assets)
        # decoder[0].weight shape: (hidden_dim, latent_dim)
        # decoder[2].weight shape: (n_assets, hidden_dim)
        #
        # The full decoder mapping: z -> W1*z + b1 -> ReLU -> W2*h + b2
        # Factor portfolios in asset space: W2 @ W1 gives (n_assets, latent_dim)
        # Each column is a factor portfolio showing how each latent factor
        # maps to asset weights through the full decoder.
        W1 = model.decoder[0].weight.detach().numpy()  # (hidden_dim, latent_dim)
        W2 = model.decoder[2].weight.detach().numpy()  # (n_assets, hidden_dim)
        factor_portfolios = W2 @ W1  # (n_assets, latent_dim)

        # Equal-weight combination of all factor portfolios
        # Average across latent factors to get a single set of asset weights
        raw_weights = factor_portfolios.mean(axis=1)  # (n_assets,)

        # Convert to long-only: take absolute values and normalize
        weights = np.abs(raw_weights)
        weights = weights / weights.sum()

        return weights
