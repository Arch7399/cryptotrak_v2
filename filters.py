import pandas as pd


def price_spikes_crashes(df, spike_threshold=20, crash_threshold=-20):
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

    df_spikes = df[df["spike_score"] > spike_threshold].sort_values(
        by="spike_score", ascending=False
    )
    df_crashes = df[df["crash_score"] < crash_threshold].sort_values(
        by="crash_score", ascending=True
    )

    return df_spikes, df_crashes


def sudden_volume_surges(df, threshold=50):
    df["volume_surge_score"] = df["quote.USD.volume_change_24h"].abs()
    df_volume_surges = df[df["quote.USD.volume_change_24h"] > threshold].sort_values(
        by="volume_surge_score", ascending=False
    )
    return df_volume_surges


def low_liquidity_coins(df, volume_market_cap_threshold=0.02, min_market_pairs=5):
    df["liquidity_score"] = 1 / df["volume_to_market_cap_ratio"]
    df_low_liquidity = df[
        (df["volume_to_market_cap_ratio"] < volume_market_cap_threshold)
        & (df["num_market_pairs"] < min_market_pairs)
    ].sort_values(by="liquidity_score", ascending=False)
    return df_low_liquidity


def pump_and_dump_detection(df, spike_threshold=50, drop_threshold=-30):
    df["pump_score"] = df["quote.USD.percent_change_24h"].abs()
    df["dump_score"] = df["quote.USD.percent_change_1h"].abs()
    df_pump = df[df["quote.USD.percent_change_24h"] > spike_threshold].sort_values(
        by="pump_score", ascending=False
    )
    df_dump = df[df["quote.USD.percent_change_1h"] < drop_threshold].sort_values(
        by="dump_score", ascending=False
    )
    return df_pump, df_dump


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


def market_cap_volume_discrepancy(df, min_market_cap=1e8, max_volume_ratio=0.01):
    df["discrepancy_score"] = df["quote.USD.market_cap"] / df["quote.USD.volume_24h"]
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
        "Price Spikes": df_spikes,
        "Price Crashes": df_crashes,
        "Volume Surges": df_volume_surges,
        "Low Liquidity": df_low_liquidity,
        "Pump": df_pump,
        "Dump": df_dump,
        "Low RSI": df_low_rsi,
        "High RSI": df_high_rsi,
        "Market Cap_Volume Discrepancy": df_discrepancy,
    }
