"""Prepare S&P 100 daily returns data.

Calls the data loader to fetch stock data from the ARF Data API,
compute daily returns, and save to CSV.
"""

import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_stock_data, save_returns


def main():
    config_path = "configs/default.yaml"
    output_path = "data/processed/sp100_daily_returns.csv"

    print("=" * 60)
    print("S&P 100 Daily Returns Data Pipeline")
    print("=" * 60)

    returns_df = load_stock_data(config_path=config_path)

    # Validation
    n_assets = returns_df.shape[1]
    n_rows = returns_df.shape[0]
    n_nan = returns_df.isna().sum().sum()

    print(f"\nValidation:")
    print(f"  Assets: {n_assets} (requirement: >= 80)")
    print(f"  Rows:   {n_rows} (requirement: >= 3000)")
    print(f"  NaN:    {n_nan} (requirement: 0)")

    assert n_assets >= 80, f"Only {n_assets} assets, need >= 80"
    assert n_rows >= 3000, f"Only {n_rows} rows, need >= 3000"
    assert n_nan == 0, f"Found {n_nan} NaN values"

    save_returns(returns_df, output_path)

    print(f"\nPipeline complete.")
    print(f"  Output: {output_path}")
    print(f"  Shape:  {returns_df.shape}")


if __name__ == "__main__":
    main()
