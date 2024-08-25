from config import Config
import pandas as pd
import talib


def add_technical_indicators(df):
    """Add common technical indicators to the dataframe"""
    try:
        prices = df["quote.USD.price"].values
        volume = df["quote.USD.volume_24h"].values

        indicators = {}

        indicators["EMA_9"] = talib.EMA(prices, timeperiod=9)
        indicators["EMA_20"] = talib.EMA(prices, timeperiod=20)
        macd, signal, hist = talib.MACD(prices)
        indicators["MACD"] = macd
        indicators["MACD_signal"] = signal
        indicators["MACD_hist"] = hist
        indicators["RSI"] = talib.RSI(prices, timeperiod=14)

        indicators["OBV"] = talib.OBV(prices, volume)
        indicators["AD"] = talib.AD(
            high=prices, low=prices * 0.9999, close=prices, volume=volume
        )
        indicators["CMF"] = talib.ADOSC(
            high=prices, low=prices * 0.9999, close=prices, volume=volume
        )

        indicators["MOM"] = talib.MOM(prices, timeperiod=10)
        indicators["ROC"] = talib.ROC(prices, timeperiod=10)

        indicators["ATR"] = talib.ATR(
            high=prices, low=prices * 0.9999, close=prices, timeperiod=14
        )
        upper, middle, lower = talib.BBANDS(prices, timeperiod=20)
        indicators["BB_%B"] = (prices - lower) / (upper - lower)

        for name, values in indicators.items():
            df[name] = values

        return df
    except Exception as e:
        print(f"Error in add_technical_indicators: {str(e)}")
        return df


def bullish_momentum_breakouts(df, p_s_t, v_s_t, p_w, v_w):

    df_result = df.copy()

    new_columns = {}

    new_columns["price_spike"] = (
        df_result["quote.USD.percent_change_24h"] >= p_s_t
    ) | (
        (df_result["EMA_9"] > df_result["EMA_20"])
        & (df_result["MACD_hist"] > 0)
        & (df_result["ROC"] > 0)
    )

    new_columns["volume_surge"] = (
        df_result["quote.USD.volume_change_24h"] >= v_s_t
    ) | ((df_result["OBV"].pct_change() > 0.1) & (df_result["CMF"] > 0))

    new_columns["ta_price_score"] = (
        df_result["MACD_hist"] / df_result["MACD_hist"].abs().mean()
        + df_result["ROC"] / df_result["ROC"].abs().mean()
    )

    new_columns["ta_volume_score"] = (
        df_result["CMF"] * 100 + df_result["OBV"].pct_change() * 100
    )

    new_columns["spike_surge_score"] = p_w * (
        df_result["quote.USD.percent_change_24h"] + new_columns["ta_price_score"]
    ) + v_w * (
        df_result["quote.USD.volume_change_24h"] + new_columns["ta_volume_score"]
    )

    new_columns["bullish_momentum_breakout_flag"] = (
        new_columns["price_spike"] & new_columns["volume_surge"]
    ).astype(int)

    return pd.concat(
        [df_result, pd.DataFrame(new_columns, index=df_result.index)], axis=1
    )


def overbought_conditions(df, price_spike_threshold):

    df_result = df.copy()

    df_result["price_spike"] = (
        df_result["quote.USD.percent_change_24h"] > price_spike_threshold
    )
    df_result["high_rsi"] = (
        (df_result["RSI"] > 70)
        | (df_result["BB_%B"] > 1)
        | ((df_result["EMA_9"] / df_result["EMA_20"]) > 1.05)
    )

    df_result["overbought_score"] = (
        df_result["quote.USD.percent_change_24h"]
        * (df_result["RSI"] - 70)
        * (1 + df_result["BB_%B"])
        * (df_result["EMA_9"] / df_result["EMA_20"])
    )

    df_result["overbought_flag"] = (
        df_result["price_spike"] & df_result["high_rsi"]
    ).astype(int)

    return df_result


def pump_and_dump_schemes(df, pump_threshold, dump_threshold, volume_surge_threshold):
    df_result = df.copy()
    new_columns = {}

    new_columns["pump"] = (
        df_result["quote.USD.percent_change_24h"] > pump_threshold
    ) & ((df_result["ROC"] > 0) | (df_result["MACD_hist"] > 0))
    new_columns["dump"] = (
        df_result["quote.USD.percent_change_1h"] < dump_threshold
    ) & ((df_result["ROC"] < 0) | (df_result["MACD_hist"] < 0))
    new_columns["volume_surge"] = (
        df_result["quote.USD.volume_change_24h"] > volume_surge_threshold
    ) & (df_result["CMF"].abs() > df_result["CMF"].abs().mean())

    new_columns["pump_dump_score"] = (
        df_result["quote.USD.percent_change_24h"].abs()
        + df_result["quote.USD.percent_change_1h"].abs()
        + df_result["quote.USD.volume_change_24h"] / 100
        + df_result["ROC"].abs()
        + df_result["MACD_hist"].abs()
        + df_result["CMF"].abs()
    )

    new_columns["pump_and_dump_flag"] = (
        new_columns["pump"] & new_columns["dump"] & new_columns["volume_surge"]
    ).astype(int)

    return pd.concat(
        [df_result, pd.DataFrame(new_columns, index=df_result.index)], axis=1
    )


def reversal_opportunities(df, crash_threshold, low_rsi_threshold):
    df_result = df.copy()

    df_result["price_crash"] = (
        df_result["quote.USD.percent_change_24h"] < crash_threshold
    )
    df_result["low_rsi"] = (
        (df_result["RSI"] < low_rsi_threshold)
        | (df_result["BB_%B"] < 0)
        | (df_result["MACD_hist"] > df_result["MACD_hist"].shift(1))
    )

    df_result["reversal_score"] = (
        df_result["quote.USD.percent_change_24h"].abs()
        * (low_rsi_threshold - df_result["RSI"])
        * (1 + abs(df_result["BB_%B"]))
        * (1 + df_result["MOM"].abs() / df_result["MOM"].abs().mean())
    )

    df_result["reversal_opportunity_flag"] = (
        df_result["price_crash"] & df_result["low_rsi"]
    ).astype(int)

    return df_result


def false_valuation_market_manipulation(df, lower_threshold, upper_threshold):
    df_result = df.copy()
    new_columns = {}

    new_columns["volume_to_market_cap_ratio"] = (
        df_result["quote.USD.volume_24h"] / df_result["quote.USD.market_cap"]
    )
    new_columns["price_to_ATR_ratio"] = df_result["quote.USD.price"] / df_result["ATR"]

    new_columns["liquidity_score"] = 1 / new_columns["volume_to_market_cap_ratio"]
    new_columns["volatility_score"] = df_result["ATR"] / df_result["quote.USD.price"]
    new_columns["momentum_divergence"] = abs(df_result["MOM"] - df_result["ROC"])

    new_columns["false_valuation_score"] = (
        0.4 * new_columns["liquidity_score"]
        + 0.3 * new_columns["volatility_score"]
        + 0.3 * new_columns["momentum_divergence"]
    )

    new_columns["false_valuation_flag"] = (
        (new_columns["false_valuation_score"] > lower_threshold)
        & (new_columns["false_valuation_score"] < upper_threshold)
    ).astype(int)

    return pd.concat(
        [df_result, pd.DataFrame(new_columns, index=df_result.index)], axis=1
    )


def apply_tandem_filters(df):
    df = add_technical_indicators(df)
    df = bullish_momentum_breakouts(
        df, Config.p_s_t, Config.v_s_t, Config.p_w, Config.v_w
    )
    df = overbought_conditions(df, Config.o_p_s_t)
    df = pump_and_dump_schemes(df, Config.p_t, Config.d_t, Config.p_d_v_s_t)
    df = reversal_opportunities(df, Config.c_t, Config.l_rsi_t)
    df = false_valuation_market_manipulation(df, Config.l_t, Config.u_t)

    return df
