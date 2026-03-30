"""Data loader for S&P 100 stock returns via ARF Data API.

Fetches OHLCV data, computes daily returns, and handles missing values.
"""

import os
import time

import pandas as pd
import yaml

CONFIG_PATH = "configs/default.yaml"

# S&P 100 constituents available via ARF Data API (~84 tickers)
SP100_TICKERS = [
    # Mega Cap
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    # Semiconductors (excluding duplicates)
    "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "MRVL", "LRCX", "KLAC", "AMAT",
    "ASML", "TSM", "ADI",
    # AI & Software
    "PLTR", "CRM", "NOW", "SNOW", "PATH", "CDNS", "SNPS",
    # Finance (excluding JPM duplicate)
    "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "CME", "ICE",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "OXY",
    # Healthcare & Pharma
    "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "UNH", "ABT",
    # Defense
    "RTX", "LMT", "NOC", "GD", "BA", "LHX",
    # Cybersecurity
    "CRWD", "PANW", "FTNT", "ZS", "NET", "CYBR",
    # Consumer & Retail
    "WMT", "COST", "HD", "MCD", "SBUX", "NKE", "TGT",
    # Utilities & Infrastructure
    "NEE", "DUK", "SO", "CEG",
    # Transportation
    "UNP", "UPS", "FDX", "DAL",
]


def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_ticker_data(
    ticker: str,
    api_base_url: str,
    start_date: str,
    end_date: str,
) -> pd.Series | None:
    """Fetch adjusted close prices for a single ticker from the ARF Data API.

    Returns a Series of adjusted close prices indexed by date, or None on failure.
    """
    url = f"{api_base_url}?ticker={ticker}&interval=1d&period=max"
    try:
        df = pd.read_csv(url)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        # Filter to requested date range
        df = df.loc[start_date:end_date]

        if df.empty or "close" not in df.columns:
            return None

        return df["close"].rename(ticker)
    except Exception as e:
        print(f"  Warning: Failed to fetch {ticker}: {e}")
        return None


def load_stock_data(config_path: str = CONFIG_PATH) -> pd.DataFrame:
    """Load S&P 100 stock data and compute daily returns.

    Fetches daily adjusted close prices from the ARF Data API,
    computes daily percentage returns, and handles NaN values.

    Returns:
        DataFrame of daily returns with date index and ticker columns.
    """
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    api_base_url = data_cfg["api_base_url"]
    start_date = data_cfg["start_date"]
    end_date = data_cfg["end_date"]

    print(f"Fetching data for {len(SP100_TICKERS)} tickers...")
    print(f"Date range: {start_date} to {end_date}")

    prices = {}
    failed = []

    for i, ticker in enumerate(SP100_TICKERS):
        print(f"  [{i+1}/{len(SP100_TICKERS)}] Fetching {ticker}...")
        series = fetch_ticker_data(ticker, api_base_url, start_date, end_date)
        if series is not None and len(series) > 100:
            prices[ticker] = series
        else:
            failed.append(ticker)
        # Small delay to avoid overwhelming the API
        if (i + 1) % 10 == 0:
            time.sleep(1)

    print(f"\nSuccessfully fetched: {len(prices)} tickers")
    if failed:
        print(f"Failed/insufficient data: {len(failed)} tickers — {failed}")

    # Combine into a single DataFrame
    price_df = pd.DataFrame(prices)

    # Compute daily returns (pct_change uses t-1 to t)
    returns_df = price_df.pct_change()

    # Drop the first row (NaN from pct_change)
    returns_df = returns_df.iloc[1:]

    # Fill NaN returns with 0 (stock not yet trading = no return).
    # Do NOT use bfill, which would propagate the first real return
    # backward into pre-IPO periods, creating phantom returns.
    returns_df = returns_df.fillna(0.0)

    # Drop any columns that are entirely zero (no real data at all)
    all_zero_cols = returns_df.columns[(returns_df == 0).all()]
    if len(all_zero_cols) > 0:
        print(f"Dropping columns with no data: {list(all_zero_cols)}")
        returns_df = returns_df.drop(columns=all_zero_cols)

    print(f"\nFinal dataset: {returns_df.shape[0]} rows × {returns_df.shape[1]} columns")
    return returns_df


def daily_to_monthly(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Convert daily returns to monthly returns using month-end prices.

    Extracts the last business day price each month from the underlying
    daily returns, then computes month-over-month percentage changes.
    This aligns with the paper's use of monthly return data.

    Args:
        returns_df: DataFrame of daily returns with DatetimeIndex.

    Returns:
        DataFrame of monthly returns indexed by month-end date.
    """
    # Reconstruct price levels from daily returns (cumulative product)
    price_levels = (1 + returns_df).cumprod()
    # Resample to month-end: take the last available observation each month
    monthly_prices = price_levels.resample("ME").last()
    # Compute monthly returns as pct_change of month-end prices
    monthly_returns = monthly_prices.pct_change().dropna()
    return monthly_returns


def load_monthly_returns(config_path: str = CONFIG_PATH) -> pd.DataFrame:
    """Load daily returns from cache and convert to monthly.

    Args:
        config_path: Path to YAML config file.

    Returns:
        DataFrame of monthly returns with month-end DatetimeIndex.
    """
    cfg = load_config(config_path)
    output_dir = cfg["data"]["output_dir"]
    daily_path = os.path.join(output_dir, "sp100_daily_returns.csv")

    if not os.path.exists(daily_path):
        raise FileNotFoundError(
            f"{daily_path} not found. Run scripts/prepare_data.py first."
        )

    daily_returns = pd.read_csv(daily_path, index_col=0, parse_dates=True)
    return daily_to_monthly(daily_returns)


def save_returns(returns_df: pd.DataFrame, output_path: str) -> None:
    """Save returns DataFrame to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    returns_df.to_csv(output_path)
    print(f"Returns saved to {output_path}")
