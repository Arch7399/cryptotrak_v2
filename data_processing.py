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
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    # Calculate RSI
    df["RSI"] = talib.RSI(close, timeperiod=14)

    # Calculate Bollinger Bands
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]

    # For indicators that typically use High/Low prices, we'll adapt to use just close price:
    
    # Modified ATR calculation using close prices
    # We'll use the daily change in price as a simple volatility measure
    price_change = np.abs(close - np.roll(close, 1))
    df["ATR"] = talib.SMA(price_change, timeperiod=14)

    # Modified Stochastic Oscillator using rolling min/max of close prices
    period = 14
    rolling_low = pd.Series(close).rolling(window=period).min()
    rolling_high = pd.Series(close).rolling(window=period).max()
    
    k = 100 * ((close - rolling_low) / (rolling_high - rolling_low))
    df["STOCH_K"] = k
    df["STOCH_D"] = talib.SMA(k, timeperiod=3)

    return df

def scoring(df):
    df["combined_score"] = (
        0.2 * df["volume_momentum"]
        + 0.2 * df["quote.USD.percent_change_24h"]
        + 0.2 * df["quote.USD.percent_change_7d"]
        + 0.1 * (50 - df["RSI"].abs()) / 50  # RSI closer to 50 is better
        + 0.1 * df["MACD_hist"]  # MACD histogram
        + 0.1 * (1 - df["BB_width"])  # Lower Bollinger Band width is better
        + 0.1 * (50 - df["STOCH_K"].abs()) / 50  # Stochastic K closer to 50 is better
    )
    return df


def process_data(df):
    df = df[df["quote.USD.price"] > 0]
    df = vol_momentum_moving_avg(df)
    df = liquidity(df)
    df = momentum(df)
    df = volatility(df)
    df = metrics(df)
    df = technical_indicators(df)
    df = scoring(df)
    return df