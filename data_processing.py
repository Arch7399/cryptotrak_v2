import pandas as pd
import talib
import numpy as np


def liquidity(df):
    df["quote.USD.volume_24h"] = pd.to_numeric(
        df["quote.USD.volume_24h"], errors="coerce"
    )
    df["quote.USD.market_cap"] = pd.to_numeric(
        df["quote.USD.market_cap"], errors="coerce"
    )
    df["num_market_pairs"] = pd.to_numeric(df["num_market_pairs"], errors="coerce")
    df["volume_to_market_cap_ratio"] = (
        df["quote.USD.volume_24h"] / df["quote.USD.market_cap"]
    )
    return df


def momentum(df):
    df["momentum_score"] = df[
        [
            "quote.USD.percent_change_1h",
            "quote.USD.percent_change_24h",
            "quote.USD.percent_change_7d",
            "quote.USD.percent_change_30d",
        ]
    ].mean(axis=1)
    return df


def calculate_volatility(row):
    percent_changes = [
        row["quote.USD.percent_change_1h"],
        row["quote.USD.percent_change_24h"],
        row["quote.USD.percent_change_7d"],
    ]
    return pd.Series(percent_changes).std()


def volatility(df):
    df["volatility"] = df.apply(calculate_volatility, axis=1)
    return df


def metrics(df):
    df["circulating_to_max_supply_ratio"] = df["circulating_supply"] / df["max_supply"]
    df["tvl_ratio"] = df["quote.USD.market_cap"] / df["quote.USD.tvl"]
    return df


def vol_momentum_moving_avg(df):
    df["volume_momentum"] = df["quote.USD.volume_change_24h"]
    df["7d_moving_avg"] = df["quote.USD.price"].rolling(window=7).mean()
    return df


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def profit_potential(df):
    df["price_to_supply_ratio"] = df["quote.USD.price"] / df["circulating_supply"]
    return df


def technical_indicators(df):
    # Convert price to numpy array
    close = df["quote.USD.price"].values

    # Calculate SMA
    df["SMA_50"] = talib.SMA(close, timeperiod=50)
    df["SMA_200"] = talib.SMA(close, timeperiod=200)

    # Calculate EMA
    df["EMA_20"] = talib.EMA(close, timeperiod=20)

    # Calculate MACD
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )

    # Calculate RSI
    df["RSI"] = talib.RSI(close, timeperiod=14)

    # Calculate Bollinger Bands
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]

    price_change = np.abs(close - np.roll(close, 1))
    df["ATR"] = talib.SMA(price_change, timeperiod=14)

    # Stochastic Oscillator using rolling min/max of close prices
    period = 14
    rolling_low = pd.Series(close).rolling(window=period).min()
    rolling_high = pd.Series(close).rolling(window=period).max()

    k = 100 * ((close - rolling_low) / (rolling_high - rolling_low))
    df["STOCH_K"] = k
    df["STOCH_D"] = talib.SMA(k, timeperiod=3)

    return df


def market_dominance(df):
    """
    Calculate market dominance metrics for each cryptocurrency.

    Market dominance considers:
    1. Market cap relative to total market
    2. Volume dominance
    3. Trading pair dominance
    4. Price impact on market
    """
    # Calculate total market cap and volume
    total_market_cap = df["quote.USD.market_cap"].sum()
    total_volume = df["quote.USD.volume_24h"].sum()

    # Market cap dominance
    df["market_cap_dominance"] = df["quote.USD.market_cap"] / total_market_cap

    # Volume dominance
    df["volume_dominance"] = df["quote.USD.volume_24h"] / total_volume

    # Trading pair dominance
    max_pairs = df["num_market_pairs"].max()
    df["pair_dominance"] = df["num_market_pairs"] / max_pairs

    # Calculate market impact score using volatility and volume
    df["market_impact"] = df["volatility"] * df["volume_dominance"]

    # Combined dominance score
    df["market_dominance_score"] = (
        0.4 * df["market_cap_dominance"]
        + 0.3 * df["volume_dominance"]
        + 0.2 * df["pair_dominance"]
        + 0.1 * df["market_impact"]
    )

    return df


def volume_stability(df):
    """
    Calculate volume stability metrics for each cryptocurrency.

    Volume stability considers:
    1. Volume consistency over time
    2. Volume relative to market cap
    3. Volume distribution across trading pairs
    4. Volume correlation with price movements
    """
    # Calculate volume stability metrics
    df["volume_volatility"] = df["quote.USD.volume_change_24h"].abs()

    # Volume to market cap ratio stability
    df["vol_mcap_ratio"] = df["quote.USD.volume_24h"] / df["quote.USD.market_cap"]
    df["vol_mcap_stability"] = 1 / (1 + df["vol_mcap_ratio"])

    # Volume per trading pair
    df["volume_per_pair"] = df["quote.USD.volume_24h"] / df["num_market_pairs"]

    # Volume-price correlation using momentum and volume
    df["vol_price_correlation"] = (
        df["momentum_score"] * df["quote.USD.volume_change_24h"]
    ).abs()

    # Normalize volume stability metrics
    for col in [
        "volume_volatility",
        "vol_mcap_stability",
        "volume_per_pair",
        "vol_price_correlation",
    ]:
        df[f"{col}_normalized"] = (df[col] - df[col].min()) / (
            df[col].max() - df[col].min()
        )

    # Combined volume stability score
    df["volume_stability_score"] = (
        0.35 * (1 - df["volume_volatility_normalized"])
        + 0.30 * df["vol_mcap_stability_normalized"]
        + 0.20 * df["volume_per_pair_normalized"]
        + 0.15 * (1 - df["vol_price_correlation_normalized"])
    )

    return df


def apply_market_metrics(df):
    """
    Apply both market dominance and volume stability calculations to the dataframe.
    """
    df = market_dominance(df)
    df = volume_stability(df)

    # Calculate an overall market stability index
    df["market_stability_index"] = (
        0.6 * df["market_dominance_score"] + 0.4 * df["volume_stability_score"]
    )

    return df


def scoring(df):
    df["combined_score"] = (
        0.2 * df["volume_momentum"]
        + 0.2 * df["quote.USD.percent_change_24h"]
        + 0.2 * df["quote.USD.percent_change_7d"]
        + 0.1 * (50 - df["RSI"].abs()) / 50
        + 0.1 * df["MACD_hist"]
        + 0.1 * (1 - df["BB_width"])
        + 0.1 * (50 - df["STOCH_K"].abs()) / 50
    )
    return df


def process_data(df):
    df = df[pd.to_numeric(df["quote.USD.price"], errors="coerce") > 0]
    df = vol_momentum_moving_avg(df)
    df = liquidity(df)
    df = momentum(df)
    df = volatility(df)
    df = metrics(df)
    df = technical_indicators(df)
    df = apply_market_metrics(df)
    df = scoring(df)
    return df
