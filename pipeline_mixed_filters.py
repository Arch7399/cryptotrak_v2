from config import Config


def bullish_momentum_breakouts(df, p_s_t, v_s_t, p_w, v_w):
    df["price_spike"] = df["quote.USD.percent_change_24h"] >= p_s_t
    df["volume_surge"] = df["quote.USD.volume_change_24h"] >= v_s_t

    df["spike_surge_score"] = (
        p_w * df["quote.USD.percent_change_24h"]
        + v_w * df["quote.USD.volume_change_24h"]
    )

    df["bullish_momentum_breakout_flag"] = (
        df["price_spike"] & df["volume_surge"]
    ).astype(int)

    return df


def overbought_conditions(df, price_spike_threshold):
    df["price_spike"] = df["quote.USD.percent_change_24h"] > price_spike_threshold
    df["high_rsi"] = df["RSI"] > 70

    df["overbought_score"] = df["quote.USD.percent_change_24h"] * (df["RSI"] - 70)

    df["overbought_flag"] = (df["price_spike"] & df["high_rsi"]).astype(int)

    return df


def pump_and_dump_schemes(df, pump_threshold, dump_threshold, volume_surge_threshold):
    df["pump"] = df["quote.USD.percent_change_24h"] > pump_threshold
    df["dump"] = df["quote.USD.percent_change_1h"] < dump_threshold
    df["volume_surge"] = df["quote.USD.volume_change_24h"] > volume_surge_threshold

    df["pump_dump_score"] = (
        df["quote.USD.percent_change_24h"].abs()
        + df["quote.USD.percent_change_1h"].abs()
        + df["quote.USD.volume_change_24h"] / 100
    )

    df["pump_and_dump_flag"] = (df["pump"] & df["dump"] & df["volume_surge"]).astype(
        int
    )

    return df


def reversal_opportunities(df, crash_threshold, low_rsi_threshold):
    df["price_crash"] = df["quote.USD.percent_change_24h"] < crash_threshold
    df["low_rsi"] = df["RSI"] < low_rsi_threshold

    df["reversal_score"] = df["quote.USD.percent_change_24h"].abs() * (
        low_rsi_threshold - df["RSI"]
    )

    df["reversal_opportunity_flag"] = (df["price_crash"] & df["low_rsi"]).astype(int)

    return df


def false_valuation_market_manipulation(df, lower_threshold, upper_threshold):
    df["volume_to_market_cap_ratio"] = (
        df["quote.USD.volume_24h"] / df["quote.USD.market_cap"]
    )
    df["liquidity_score"] = 1 / df["volume_to_market_cap_ratio"]
    df["discrepancy_score"] = df["quote.USD.market_cap"] / df["quote.USD.volume_24h"]

    df["false_valuation_score"] = (0.7 * df["liquidity_score"]) + (
        0.3 * df["discrepancy_score"]
    )

    df["false_valuation_flag"] = (
        (df["false_valuation_score"] > lower_threshold)
        & (df["false_valuation_score"] < upper_threshold)
    ).astype(int)

    return df


def apply_tandem_filters(df):
    df = bullish_momentum_breakouts(
        df, Config.p_s_t, Config.v_s_t, Config.p_w, Config.v_w
    )
    df = overbought_conditions(df, Config.o_p_s_t)
    df = pump_and_dump_schemes(df, Config.p_t, Config.d_t, Config.p_d_v_s_t)
    df = reversal_opportunities(df, Config.c_t, Config.l_rsi_t)
    df = false_valuation_market_manipulation(df, Config.l_t, Config.u_t)

    return df
