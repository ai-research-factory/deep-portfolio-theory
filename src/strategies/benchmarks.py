"""Benchmark portfolio strategies for comparison with Deep Portfolio.

Implements Equal-Weight (1/N) and Minimum Variance strategies.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class EqualWeightStrategy:
    """Equal-weight (1/N) portfolio strategy.

    Allocates equal weight to all assets regardless of training data.
    """

    def generate_weights(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Return equal weights for all assets.

        Args:
            returns_data: DataFrame of historical returns (used only for n_assets).

        Returns:
            Array of equal weights summing to 1.
        """
        n_assets = returns_data.shape[1]
        return np.ones(n_assets) / n_assets


class MinimumVarianceStrategy:
    """Long-only global minimum variance portfolio strategy.

    Computes the portfolio that minimizes variance subject to weights >= 0
    and sum(weights) == 1, using the sample covariance matrix estimated
    from training data only.
    """

    def generate_weights(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Compute long-only minimum variance portfolio weights.

        Uses scipy.optimize.minimize with SLSQP to solve the constrained
        quadratic program: min w^T Σ w, s.t. w >= 0, sum(w) = 1.

        Args:
            returns_data: DataFrame of historical returns for covariance estimation.

        Returns:
            Array of non-negative minimum variance weights summing to 1.
        """
        cov_matrix = returns_data.cov().values
        n_assets = cov_matrix.shape[0]

        # Regularize covariance matrix for numerical stability
        diag = np.diag(np.diag(cov_matrix))
        shrinkage = 0.1
        cov_reg = (1 - shrinkage) * cov_matrix + shrinkage * diag

        def portfolio_variance(w):
            return w @ cov_reg @ w

        # Initial guess: equal weight
        w0 = np.ones(n_assets) / n_assets

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # Bounds: long-only (0 <= w_i <= 1)
        bounds = [(0.0, 1.0)] * n_assets

        result = minimize(
            portfolio_variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if result.success:
            weights = result.x
            # Clip tiny negatives from numerical noise
            weights = np.maximum(weights, 0.0)
            weights = weights / weights.sum()
        else:
            # Fallback to equal weight
            weights = np.ones(n_assets) / n_assets

        return weights
