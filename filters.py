import talib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def rsi_filter(df, rsi_high=70, rsi_low=30):
    df["rsi_low_score"] = 50 - df["RSI"]
    df["rsi_high_score"] = df["RSI"] - 50
    df_low_rsi = df[df["RSI"] < rsi_low].sort_values(
        by="rsi_low_score", ascending=False
    )
    df_high_rsi = df[df["RSI"] > rsi_high].sort_values(
        by="rsi_high_score", ascending=False
    )
    return df_low_rsi, df_high_rsi


def price_spikes_crashes(
    df: pd.DataFrame,
    spike_threshold: float = 20,
    crash_threshold: float = -20,
    rsi_period: int = 14,
    bb_period: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Spike and crash detection using multiple technical indicators
    """
    prices = df["quote.USD.price"].values

    # Calculate technical indicators
    df["RSI"] = talib.RSI(prices, timeperiod=rsi_period)

    upper, middle, lower = talib.BBANDS(
        prices, timeperiod=bb_period, nbdevup=2, nbdevdn=2, matype=0
    )
    df["BB_upper"] = upper
    df["BB_lower"] = lower
    df["BB_%B"] = (prices - lower) / (upper - lower)

    df["MACD"], df["MACD_signal"], df["MACD_hist"] = talib.MACD(prices)

    # scoring system
    df["spike_score"] = 0
    df["crash_score"] = 0

    # Price change contribution
    for period in ["1h", "24h", "7d"]:
        col = f"quote.USD.percent_change_{period}"
        if col in df.columns:
            df["spike_score"] += np.where(df[col] > 0, df[col], 0)
            df["crash_score"] += np.where(df[col] < 0, abs(df[col]), 0)

    # RSI contribution
    df["spike_score"] += np.where(df["RSI"] > 70, (df["RSI"] - 70) * 2, 0)
    df["crash_score"] += np.where(df["RSI"] < 30, (30 - df["RSI"]) * 2, 0)

    # Bollinger Bands contribution
    df["spike_score"] += np.where(df["BB_%B"] > 1, (df["BB_%B"] - 1) * 50, 0)
    df["crash_score"] += np.where(df["BB_%B"] < 0, abs(df["BB_%B"]) * 50, 0)

    # MACD contribution
    df["spike_score"] += np.where(df["MACD_hist"] > 0, df["MACD_hist"], 0)
    df["crash_score"] += np.where(df["MACD_hist"] < 0, abs(df["MACD_hist"]), 0)

    # spikes and crashes
    df_spikes = df[df["spike_score"] > spike_threshold].sort_values(
        "spike_score", ascending=False
    )
    df_crashes = df[df["crash_score"] > abs(crash_threshold)].sort_values(
        "crash_score", ascending=False
    )

    return df_spikes, df_crashes


def sudden_volume_surges(
    df: pd.DataFrame, threshold: float = 50, obv_threshold: float = 1000000
) -> pd.DataFrame:
    """
    Volume surge detection using On Balance Volume (OBV) and other volume indicators
    """
    prices = df["quote.USD.price"].values
    volume = df["quote.USD.volume_24h"].values

    # Calculate volume indicators
    df["OBV"] = talib.OBV(prices, volume)
    df["AD"] = talib.AD(high=prices, low=prices * 0.9999, close=prices, volume=volume)
    df["CMF"] = talib.ADOSC(
        high=prices, low=prices * 0.9999, close=prices, volume=volume
    )

    # Volume surge scoring
    df["volume_surge_score"] = 0

    # Standard volume change contribution
    df["volume_surge_score"] += df["quote.USD.volume_change_24h"].abs()

    # OBV contribution
    df["OBV_change"] = df["OBV"].pct_change()
    df["volume_surge_score"] += np.where(
        df["OBV_change"] > 0, df["OBV_change"] * 100, 0
    )

    # Chaikin Money Flow contribution
    df["volume_surge_score"] += np.where(df["CMF"] > 0, df["CMF"] * 50, 0)

    # Filter and sort results
    df_volume_surges = df[
        (df["quote.USD.volume_change_24h"] > threshold) & (df["OBV"] > obv_threshold)
    ].sort_values(by="volume_surge_score", ascending=False)

    return df_volume_surges


def low_liquidity_coins(
    df: pd.DataFrame,
    volume_market_cap_threshold: float = 0.02,
    min_market_pairs: int = 5,
    atr_period: int = 14,
) -> pd.DataFrame:
    """
    Liquidity analysis using ATR and other volatility indicators
    """
    prices = df["quote.USD.price"].values

    # Calculate ATR
    df["ATR"] = talib.ATR(
        high=prices, low=prices * 0.9999, close=prices, timeperiod=atr_period
    )

    # Liquidity scoring
    df["liquidity_score"] = 1 / df["volume_to_market_cap_ratio"]

    # ATR contribution
    df["liquidity_score"] *= 1 + df["ATR"] / prices

    # Market pairs contribution
    df["liquidity_score"] *= 10 / (df["num_market_pairs"] + 1)

    # Filter and sort results
    df_low_liquidity = df[
        (df["volume_to_market_cap_ratio"] < volume_market_cap_threshold)
        & (df["num_market_pairs"] < min_market_pairs)
    ].sort_values(by="liquidity_score", ascending=False)

    return df_low_liquidity


def pump_and_dump_detection(
    df: pd.DataFrame,
    spike_threshold: float = 50,
    drop_threshold: float = -30,
    ema_short: int = 9,
    ema_long: int = 26,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enhanced pump and dump detection using multiple technical indicators
    """
    prices = df["quote.USD.price"].values

    # Calculate EMAs
    df["EMA_short"] = talib.EMA(prices, timeperiod=ema_short)
    df["EMA_long"] = talib.EMA(prices, timeperiod=ema_long)

    # Calculate momentum indicators
    df["MOM"] = talib.MOM(prices, timeperiod=14)
    df["ROC"] = talib.ROC(prices, timeperiod=10)

    # Pump and dump scoring
    df["pump_score"] = 0
    df["dump_score"] = 0

    # Price change contribution
    df["pump_score"] += df["quote.USD.percent_change_24h"].clip(lower=0)
    df["dump_score"] += abs(df["quote.USD.percent_change_1h"].clip(upper=0))

    # EMA crossover contribution
    df["pump_score"] += np.where(
        df["EMA_short"] > df["EMA_long"],
        (df["EMA_short"] / df["EMA_long"] - 1) * 100,
        0,
    )
    df["dump_score"] += np.where(
        df["EMA_short"] < df["EMA_long"],
        (df["EMA_long"] / df["EMA_short"] - 1) * 100,
        0,
    )

    # Momentum contribution
    df["pump_score"] += np.where(df["MOM"] > 0, df["MOM"], 0)
    df["dump_score"] += np.where(df["MOM"] < 0, abs(df["MOM"]), 0)

    # ROC contribution
    df["pump_score"] += np.where(df["ROC"] > 0, df["ROC"], 0)
    df["dump_score"] += np.where(df["ROC"] < 0, abs(df["ROC"]), 0)

    # Filter and sort results
    df_pump = df[df["pump_score"] > spike_threshold].sort_values(
        by="pump_score", ascending=False
    )
    df_dump = df[df["dump_score"] > abs(drop_threshold)].sort_values(
        by="dump_score", ascending=False
    )

    return df_pump, df_dump


def market_cap_volume_discrepancy(
    df: pd.DataFrame,
    min_market_cap: float = 1e8,
    max_volume_ratio: float = 0.01,
    vwap_period: int = 14,
) -> pd.DataFrame:
    """
    Enhanced market cap and volume analysis using VWAP and other indicators
    """
    prices = df["quote.USD.price"].values
    volume = df["quote.USD.volume_24h"].values

    # Calculate VWAP
    df["VWAP"] = talib.SMA(prices * volume, timeperiod=vwap_period) / talib.SMA(
        volume, timeperiod=vwap_period
    )

    # DEnhanced discrepancy scoring
    df["discrepancy_score"] = df["quote.USD.market_cap"] / df["quote.USD.volume_24h"]

    # VWAP contribution
    df["price_to_vwap"] = prices / df["VWAP"]
    df["discrepancy_score"] *= df["price_to_vwap"]

    # Filter and sort results
    df_discrepancy = df[
        (df["quote.USD.market_cap"] > min_market_cap)
        & (df["volume_to_market_cap_ratio"] < max_volume_ratio)
    ].sort_values(by="discrepancy_score", ascending=False)

    return df_discrepancy


def apply_filters(df):
    df_spikes, df_crashes = price_spikes_crashes(df)
    df_volume_surges = sudden_volume_surges(df)
    df_low_liquidity = low_liquidity_coins(df)
    df_pump, df_dump = pump_and_dump_detection(df)
    df_low_rsi, df_high_rsi = rsi_filter(df)
    df_discrepancy = market_cap_volume_discrepancy(df)

    return {
        "PriceSpikes": df_spikes,
        "PriceCrashes": df_crashes,
        "VolumeSurges": df_volume_surges,
        "LowLiquidity": df_low_liquidity,
        "Pump": df_pump,
        "Dump": df_dump,
        "LowRSI": df_low_rsi,
        "HighRSI": df_high_rsi,
        "MarketCapVolumeDiscrepancy": df_discrepancy,
    }
