"""Walk-forward validation framework for portfolio strategies.

Implements a rolling-window walk-forward validator that trains strategies
on a fixed lookback period and evaluates on the subsequent month.
"""

import pandas as pd
import numpy as np


class WalkForwardValidator:
    """Walk-forward out-of-sample validator for portfolio strategies.

    Args:
        returns: DataFrame of daily asset returns (date index, ticker columns).
        train_months: Number of months in the training window (default 60).
        test_months: Number of months in the test window (default 1).
        rebalance_freq: Rebalance frequency string for pd.offsets (default 'MS' = month start).
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        train_months: int = 60,
        test_months: int = 1,
        rebalance_freq: str = "MS",
    ):
        self.returns = returns
        self.train_months = train_months
        self.test_months = test_months
        self.rebalance_freq = rebalance_freq

    def _generate_windows(self) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate (train_start, test_start, test_end) tuples for walk-forward splits.

        Each window uses `train_months` of history ending just before `test_start`,
        and `test_months` of out-of-sample data starting at `test_start`.
        """
        data_start = self.returns.index.min()
        data_end = self.returns.index.max()

        # First possible test start: after train_months of data
        first_test_start = data_start + pd.DateOffset(months=self.train_months)

        # Generate monthly rebalance dates
        rebalance_dates = pd.date_range(
            start=first_test_start,
            end=data_end,
            freq=self.rebalance_freq,
        )

        windows = []
        for test_start in rebalance_dates:
            train_start = test_start - pd.DateOffset(months=self.train_months)
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # Ensure we have data in both periods
            train_data = self.returns.loc[train_start:test_start - pd.Timedelta(days=1)]
            test_data = self.returns.loc[test_start:test_end - pd.Timedelta(days=1)]

            if len(train_data) < 20 or len(test_data) == 0:
                continue

            windows.append((train_start, test_start, test_end))

        return windows

    def run(self, strategy) -> dict:
        """Run walk-forward backtest for a given strategy.

        Args:
            strategy: Object with a `generate_weights(returns_data)` method
                that returns a numpy array or Series of portfolio weights.

        Returns:
            dict with keys:
                - 'daily_returns': Series of daily portfolio returns (full OOS period)
                - 'windows': list of window metadata dicts
                - 'weights_history': list of (date, weights) tuples
        """
        windows = self._generate_windows()

        all_portfolio_returns = []
        window_metadata = []
        weights_history = []

        for train_start, test_start, test_end in windows:
            # Strict temporal separation: train ends before test begins
            train_returns = self.returns.loc[
                train_start : test_start - pd.Timedelta(days=1)
            ]
            test_returns = self.returns.loc[
                test_start : test_end - pd.Timedelta(days=1)
            ]

            if len(test_returns) == 0:
                continue

            # Generate weights using only training data
            weights = strategy.generate_weights(train_returns)

            if isinstance(weights, pd.Series):
                weights = weights.values

            weights = np.asarray(weights, dtype=np.float64)

            # Compute daily portfolio returns for the test period
            port_returns = test_returns.values @ weights
            port_series = pd.Series(port_returns, index=test_returns.index)
            all_portfolio_returns.append(port_series)

            weights_history.append((test_start, weights))

            # Window-level metrics
            window_sharpe = (
                port_series.mean() / port_series.std() * np.sqrt(252)
                if port_series.std() > 0 else 0.0
            )
            window_metadata.append({
                "train_start": str(train_start.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_returns.index[-1].date()),
                "n_train_days": len(train_returns),
                "n_test_days": len(test_returns),
                "sharpe": round(window_sharpe, 4),
            })

        # Concatenate all OOS returns
        if all_portfolio_returns:
            daily_returns = pd.concat(all_portfolio_returns)
            # Remove any duplicate indices (overlapping month boundaries)
            daily_returns = daily_returns[~daily_returns.index.duplicated(keep="first")]
            daily_returns = daily_returns.sort_index()
        else:
            daily_returns = pd.Series(dtype=float)

        return {
            "daily_returns": daily_returns,
            "windows": window_metadata,
            "weights_history": weights_history,
        }
