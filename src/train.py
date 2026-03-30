"""Training script for the Autoencoder model.

Trains an Autoencoder on synthetic data (Phase 1) to validate that the
architecture learns to reconstruct inputs through the bottleneck.
"""

import json
import os

import torch
import torch.nn as nn

from src.models.autoencoder import Autoencoder

# Paths
MODELS_DIR = "models"
REPORTS_DIR = "reports/cycle_1"


def train(
    input_dim: int = 256,
    hidden_dim: int = 128,
    latent_dim: int = 32,
    n_samples: int = 1000,
    epochs: int = 10,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Train the autoencoder and return the loss log.

    Returns:
        dict mapping epoch number (str) to loss value.
    """
    torch.manual_seed(seed)

    # Synthetic data for architecture validation
    data = torch.rand(n_samples, input_dim)

    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_log = {}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_log[str(epoch)] = round(loss_val, 6)
        print(f"Epoch {epoch}/{epochs} — Loss: {loss_val:.6f}")

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "autoencoder_v1.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save loss log
    os.makedirs(REPORTS_DIR, exist_ok=True)
    log_path = os.path.join(REPORTS_DIR, "loss_log.json")
    with open(log_path, "w") as f:
        json.dump(loss_log, f, indent=2)
    print(f"Loss log saved to {log_path}")

    return loss_log


if __name__ == "__main__":
    train()
