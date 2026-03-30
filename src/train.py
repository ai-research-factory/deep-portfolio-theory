"""Training script for the Autoencoder model.

Loads hyperparameters from configs/default.yaml.
"""

import json
import os

import torch
import torch.nn as nn
import yaml

from src.models.autoencoder import Autoencoder

# Paths
MODELS_DIR = "models"
CONFIG_PATH = "configs/default.yaml"


def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config_path: str = CONFIG_PATH, cycle: int = 1) -> dict:
    """Train the autoencoder and return the loss log.

    Args:
        config_path: Path to YAML config file.
        cycle: Current cycle number (for report output directory).

    Returns:
        dict mapping epoch number (str) to loss value.
    """
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    input_dim = model_cfg.get("input_dim") or 256
    hidden_dim = model_cfg["hidden_dim"]
    latent_dim = model_cfg["latent_dim"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["learning_rate"]
    seed = train_cfg["seed"]

    reports_dir = f"reports/cycle_{cycle}"

    torch.manual_seed(seed)

    # Synthetic data for architecture validation (Phase 1)
    n_samples = 1000
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
    os.makedirs(reports_dir, exist_ok=True)
    log_path = os.path.join(reports_dir, "loss_log.json")
    with open(log_path, "w") as f:
        json.dump(loss_log, f, indent=2)
    print(f"Loss log saved to {log_path}")

    return loss_log


if __name__ == "__main__":
    train()
