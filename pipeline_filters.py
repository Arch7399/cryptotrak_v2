import talib
import numpy as np
import pandas as pd


def price_spikes_crashes(
    df, spike_threshold=20, crash_threshold=-20, rsi_period=14, bb_period=20
):
    prices = df["quote.USD.price"].values

    df["RSI"] = talib.RSI(prices, timeperiod=rsi_period)

    upper, middle, lower = talib.BBANDS(
        prices, timeperiod=bb_period, nbdevup=2, nbdevdn=2, matype=0
    )
    df["BB_upper"] = upper
    df["BB_lower"] = lower
    df["BB_%B"] = (prices - lower) / (upper - lower)

    df["MACD"], df["MACD_signal"], df["MACD_hist"] = talib.MACD(prices)

    df["spike_score"] = df[
        [
            "quote.USD.percent_change_1h",
            "quote.USD.percent_change_24h",
            "quote.USD.percent_change_7d",
        ]
    ].max(axis=1)

    df["crash_score"] = df[
        [
            "quote.USD.percent_change_1h",
            "quote.USD.percent_change_24h",
            "quote.USD.percent_change_7d",
        ]
    ].min(axis=1)

    df["ta_spike_score"] = (
        np.where(df["RSI"] > 70, (df["RSI"] - 70) * 2, 0)
        + np.where(df["BB_%B"] > 1, (df["BB_%B"] - 1) * 50, 0)
        + np.where(df["MACD_hist"] > 0, df["MACD_hist"], 0)
    )

    df["ta_crash_score"] = (
        np.where(df["RSI"] < 30, (30 - df["RSI"]) * 2, 0)
        + np.where(df["BB_%B"] < 0, abs(df["BB_%B"]) * 50, 0)
        + np.where(df["MACD_hist"] < 0, abs(df["MACD_hist"]), 0)
    )

    df["final_spike_score"] = df["spike_score"] + df["ta_spike_score"]
    df["final_crash_score"] = abs(df["crash_score"]) + df["ta_crash_score"]

    df["price_spike_flag"] = (df["final_spike_score"] > spike_threshold).astype(int)
    df["price_crash_flag"] = (df["final_crash_score"] > abs(crash_threshold)).astype(
        int
    )

    return df


def sudden_volume_surges(df, threshold=50):
    prices = df["quote.USD.price"].values
    volume = df["quote.USD.volume_24h"].values

    df["OBV"] = talib.OBV(prices, volume)
    df["AD"] = talib.AD(high=prices, low=prices * 0.9999, close=prices, volume=volume)
    df["CMF"] = talib.ADOSC(
        high=prices, low=prices * 0.9999, close=prices, volume=volume
    )

    df["volume_surge_score"] = df["quote.USD.volume_change_24h"].abs()

    df["OBV_change"] = df["OBV"].pct_change()
    df["AD_change"] = df["AD"].pct_change()
    df["ta_volume_score"] = (
        np.where(df["OBV_change"] > 0, df["OBV_change"] * 100, 0)
        + np.where(df["AD_change"] > 0, df["AD_change"] * 100, 0)
        + np.where(df["CMF"] > 0, df["CMF"] * 50, 0)
    )

    # Combine scores
    df["final_volume_surge_score"] = df["volume_surge_score"] + df["ta_volume_score"]

    df["volume_surge_flag"] = (df["final_volume_surge_score"] > threshold).astype(int)
    return df


def low_liquidity_coins(
    df, volume_market_cap_threshold=0.02, min_market_pairs=5, atr_period=14
):
    prices = df["quote.USD.price"].values

    df["ATR"] = talib.ATR(
        high=prices, low=prices * 0.9999, close=prices, timeperiod=atr_period
    )

    df["liquidity_score"] = 1 / df["volume_to_market_cap_ratio"]

    df["volatility_factor"] = df["ATR"] / prices
    df["market_pairs_factor"] = 10 / (df["num_market_pairs"] + 1)

    df["final_liquidity_score"] = (
        df["liquidity_score"]
        * (1 + df["volatility_factor"])
        * df["market_pairs_factor"]
    )

    df["low_liquidity_flag"] = (
        (df["volume_to_market_cap_ratio"] < volume_market_cap_threshold)
        & (df["num_market_pairs"] < min_market_pairs)
    ).astype(int)
    return df


def pump_and_dump_detection(
    df, spike_threshold=50, drop_threshold=-30, ema_short=9, ema_long=26
):
    prices = df["quote.USD.price"].values

    df["EMA_short"] = talib.EMA(prices, timeperiod=ema_short)
    df["EMA_long"] = talib.EMA(prices, timeperiod=ema_long)
    df["MOM"] = talib.MOM(prices, timeperiod=14)
    df["ROC"] = talib.ROC(prices, timeperiod=10)

    df["pump_score"] = df["quote.USD.percent_change_24h"].abs()
    df["dump_score"] = df["quote.USD.percent_change_1h"].abs()

    df["ema_ratio"] = df["EMA_short"] / df["EMA_long"]
    df["ta_pump_score"] = (
        np.where(df["ema_ratio"] > 1, (df["ema_ratio"] - 1) * 100, 0)
        + np.where(df["MOM"] > 0, df["MOM"], 0)
        + np.where(df["ROC"] > 0, df["ROC"], 0)
    )

    df["ta_dump_score"] = (
        np.where(df["ema_ratio"] < 1, (1 - df["ema_ratio"]) * 100, 0)
        + np.where(df["MOM"] < 0, abs(df["MOM"]), 0)
        + np.where(df["ROC"] < 0, abs(df["ROC"]), 0)
    )

    df["final_pump_score"] = df["pump_score"] + df["ta_pump_score"]
    df["final_dump_score"] = df["dump_score"] + df["ta_dump_score"]

    df["pump_flag"] = (df["final_pump_score"] > spike_threshold).astype(int)
    df["dump_flag"] = (df["final_dump_score"] > abs(drop_threshold)).astype(int)
    return df


def rsi_filter(df, rsi_high=70, rsi_low=30):

    if "RSI" not in df.columns:
        prices = df["quote.USD.price"].values
        df["RSI"] = talib.RSI(prices, timeperiod=14)

    df["rsi_low_score"] = 50 - df["RSI"]
    df["rsi_high_score"] = df["RSI"] - 50
    df["rsi_low_flag"] = (df["RSI"] < rsi_low).astype(int)
    df["rsi_high_flag"] = (df["RSI"] > rsi_high).astype(int)
    return df


def market_cap_volume_discrepancy(
    df, min_market_cap=1e8, max_volume_ratio=0.01, vwap_period=14
):
    prices = df["quote.USD.price"].values
    volume = df["quote.USD.volume_24h"].values

    # Calculate VWAP
    df["VWAP"] = talib.SMA(prices * volume, timeperiod=vwap_period) / talib.SMA(
        volume, timeperiod=vwap_period
    )

    df["discrepancy_score"] = df["quote.USD.market_cap"] / df["quote.USD.volume_24h"]

    df["price_to_vwap"] = prices / df["VWAP"]
    df["final_discrepancy_score"] = df["discrepancy_score"] * df["price_to_vwap"]

    df["market_cap_volume_discrepancy_flag"] = (
        (df["quote.USD.market_cap"] > min_market_cap)
        & (df["volume_to_market_cap_ratio"] < max_volume_ratio)
    ).astype(int)
    return df


def apply_filters(df):
    df = price_spikes_crashes(df)
    df = sudden_volume_surges(df)
    df = low_liquidity_coins(df)
    df = pump_and_dump_detection(df)
    df = rsi_filter(df)
    df = market_cap_volume_discrepancy(df)
    return df
