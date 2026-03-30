"""Autoencoder model for Deep Portfolio Theory.

Implements an encoder-decoder architecture that learns a low-dimensional
factor representation of asset returns via a bottleneck layer.
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Autoencoder with configurable layer sizes.

    Architecture: Input -> hidden_dim -> latent_dim -> hidden_dim -> Input
    The latent_dim bottleneck captures the learned 'factors'.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        l2_lambda: float = 0.0,
    ):
        super().__init__()
        self.l2_lambda = l2_lambda
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def l2_penalty(self) -> torch.Tensor:
        """Compute L2 regularization penalty over all linear layer weights."""
        penalty = torch.tensor(0.0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                penalty = penalty + module.weight.pow(2).sum()
        return self.l2_lambda * penalty
