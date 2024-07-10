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

    df["price_spike_flag"] = (df["spike_score"] > spike_threshold).astype(int)
    df["price_crash_flag"] = (df["crash_score"] < crash_threshold).astype(int)

    return df


def sudden_volume_surges(df, threshold=50):
    df["volume_surge_score"] = df["quote.USD.volume_change_24h"].abs()
    df["volume_surge_flag"] = (df["quote.USD.volume_change_24h"] > threshold).astype(
        int
    )
    return df


def low_liquidity_coins(df, volume_market_cap_threshold=0.02, min_market_pairs=5):
    df["liquidity_score"] = 1 / df["volume_to_market_cap_ratio"]
    df["low_liquidity_flag"] = (
        (df["volume_to_market_cap_ratio"] < volume_market_cap_threshold)
        & (df["num_market_pairs"] < min_market_pairs)
    ).astype(int)
    return df


def pump_and_dump_detection(df, spike_threshold=50, drop_threshold=-30):
    df["pump_score"] = df["quote.USD.percent_change_24h"].abs()
    df["dump_score"] = df["quote.USD.percent_change_1h"].abs()
    df["pump_flag"] = (df["quote.USD.percent_change_24h"] > spike_threshold).astype(int)
    df["dump_flag"] = (df["quote.USD.percent_change_1h"] < drop_threshold).astype(int)
    return df


def rsi_filter(df, rsi_high=70, rsi_low=30):
    df["rsi_low_score"] = 50 - df["RSI"]
    df["rsi_high_score"] = df["RSI"] - 50
    df["rsi_low_flag"] = (df["RSI"] < rsi_low).astype(int)
    df["rsi_high_flag"] = (df["RSI"] > rsi_high).astype(int)
    return df


def market_cap_volume_discrepancy(df, min_market_cap=1e8, max_volume_ratio=0.01):
    df["discrepancy_score"] = df["quote.USD.market_cap"] / df["quote.USD.volume_24h"]
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
