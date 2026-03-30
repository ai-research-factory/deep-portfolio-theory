"""Unit tests for Autoencoder model."""

import torch
import torch.nn as nn

from src.models.autoencoder import Autoencoder


class TestAutoencoderOutputDimensions:
    """Verify that output dimensions match input dimensions for various configs."""

    def test_output_matches_input_default(self):
        """Output shape equals input shape with default parameters."""
        model = Autoencoder(input_dim=256, hidden_dim=128, latent_dim=32)
        x = torch.rand(16, 256)
        out = model(x)
        assert out.shape == x.shape

    def test_output_matches_input_small(self):
        """Output shape equals input shape with small dimensions."""
        model = Autoencoder(input_dim=10, hidden_dim=8, latent_dim=3)
        x = torch.rand(4, 10)
        out = model(x)
        assert out.shape == x.shape

    def test_output_matches_input_large(self):
        """Output shape equals input shape with larger dimensions."""
        model = Autoencoder(input_dim=500, hidden_dim=256, latent_dim=64)
        x = torch.rand(8, 500)
        out = model(x)
        assert out.shape == x.shape

    def test_single_sample(self):
        """Works with batch size of 1."""
        model = Autoencoder(input_dim=100, hidden_dim=64, latent_dim=16)
        x = torch.rand(1, 100)
        out = model(x)
        assert out.shape == x.shape


class TestAutoencoderEncode:
    """Verify encoder produces correct latent dimensions."""

    def test_encode_shape(self):
        """Encoded output has latent_dim dimensions."""
        model = Autoencoder(input_dim=256, hidden_dim=128, latent_dim=32)
        x = torch.rand(16, 256)
        z = model.encode(x)
        assert z.shape == (16, 32)

    def test_encode_various_latent(self):
        """Latent dimension is respected for different configs."""
        for latent_dim in [4, 8, 16, 64]:
            model = Autoencoder(input_dim=100, hidden_dim=64, latent_dim=latent_dim)
            x = torch.rand(5, 100)
            z = model.encode(x)
            assert z.shape == (5, latent_dim)


class TestAutoencoderTraining:
    """Verify the model can learn."""

    def test_training_reduces_loss(self):
        """Training over several epochs reduces reconstruction loss."""
        torch.manual_seed(0)
        model = Autoencoder(input_dim=64, hidden_dim=32, latent_dim=8)
        data = torch.rand(100, 64)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        initial_loss = criterion(model(data), data).item()
        for _ in range(50):
            optimizer.zero_grad()
            loss = criterion(model(data), data)
            loss.backward()
            optimizer.step()
        final_loss = criterion(model(data), data).item()

        assert final_loss < initial_loss

    def test_gradient_flow(self):
        """Gradients flow through all parameters."""
        model = Autoencoder(input_dim=32, hidden_dim=16, latent_dim=4)
        x = torch.rand(8, 32)
        loss = nn.MSELoss()(model(x), x)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
