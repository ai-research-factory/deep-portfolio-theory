"""Training script for the Autoencoder model.

Loads hyperparameters from configs/default.yaml.
Uses real S&P 100 daily returns from the data pipeline.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.models.autoencoder import Autoencoder

# Paths
MODELS_DIR = "models"
CONFIG_PATH = "configs/default.yaml"
RETURNS_PATH = "data/processed/sp100_daily_returns.csv"


def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_training_data(train_end_date: str = "2019-12-31") -> torch.Tensor:
    """Load real returns data for training.

    Args:
        train_end_date: End date for the first training period.

    Returns:
        Tensor of daily returns for the training period.
    """
    if not os.path.exists(RETURNS_PATH):
        raise FileNotFoundError(
            f"{RETURNS_PATH} not found. Run scripts/prepare_data.py first."
        )

    df = pd.read_csv(RETURNS_PATH, index_col=0, parse_dates=True)
    # Use data up to train_end_date for initial training
    train_df = df.loc[:train_end_date]
    return torch.tensor(train_df.values, dtype=torch.float32)


def train(config_path: str = CONFIG_PATH, cycle: int = 3) -> dict:
    """Train the autoencoder on real S&P 100 returns data.

    Args:
        config_path: Path to YAML config file.
        cycle: Current cycle number (for report output directory).

    Returns:
        dict mapping epoch number (str) to loss value.
    """
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    hidden_dim = model_cfg["hidden_dim"]
    latent_dim = model_cfg["latent_dim"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["learning_rate"]
    batch_size = train_cfg["batch_size"]
    seed = train_cfg["seed"]

    reports_dir = f"reports/cycle_{cycle}"

    torch.manual_seed(seed)

    # Load real data from data pipeline
    data = load_training_data(train_end_date="2019-12-31")
    input_dim = data.shape[1]
    print(f"Training data: {data.shape[0]} samples × {input_dim} assets")

    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_log = {}
    n_samples = data.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Mini-batch training
        indices = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch = data[batch_idx]

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_log[str(epoch)] = round(avg_loss, 6)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.6f}")

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
