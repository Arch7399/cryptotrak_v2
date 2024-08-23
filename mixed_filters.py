import pandas as pd
import talib
import numpy as np
from config import Config


def add_technical_indicators(df):
    """Add common technical indicators to the dataframe"""
    try:
        prices = df["quote.USD.price"].values
        volume = df["quote.USD.volume_24h"].values

        # Create a dictionary for new columns
        indicators = {}

        # Price-based indicators
        indicators["EMA_9"] = talib.EMA(prices, timeperiod=9)
        indicators["EMA_20"] = talib.EMA(prices, timeperiod=20)
        macd, signal, hist = talib.MACD(prices)
        indicators["MACD"] = macd
        indicators["MACD_signal"] = signal
        indicators["MACD_hist"] = hist
        indicators["RSI"] = talib.RSI(prices, timeperiod=14)

        # Volume-based indicators
        indicators["OBV"] = talib.OBV(prices, volume)
        indicators["AD"] = talib.AD(
            high=prices, low=prices * 0.9999, close=prices, volume=volume
        )
        indicators["CMF"] = talib.ADOSC(
            high=prices, low=prices * 0.9999, close=prices, volume=volume
        )

        # Momentum indicators
        indicators["MOM"] = talib.MOM(prices, timeperiod=10)
        indicators["ROC"] = talib.ROC(prices, timeperiod=10)

        # Volatility indicators
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


def safe_condition_check(df, conditions):
    """Safely check multiple conditions with proper alignment"""
    try:
        result = conditions[0]
        for condition in conditions[1:]:
            condition = condition.reindex(result.index)
            result = result & condition
        return result
    except Exception as e:
        print(f"Error in safe_condition_check: {str(e)}")
        return pd.Series(False, index=df.index)


def price_spikes_and_volume_surges(df, price_spike_threshold, volume_surge_threshold):
    try:
        required_columns = ["EMA_9", "EMA_20", "MACD_hist", "ROC", "OBV", "CMF"]
        for col in required_columns:
            if col not in df.columns:
                print(f"Missing column: {col}")
                return pd.DataFrame()

        price_conditions = [
            df["quote.USD.percent_change_24h"] >= price_spike_threshold,
            df["EMA_9"] > df["EMA_20"],
            df["MACD_hist"] > 0,
            df["ROC"] > 0,
        ]
        price_condition = price_conditions[0] | safe_condition_check(
            df, price_conditions[1:]
        )

        volume_conditions = [
            df["quote.USD.volume_change_24h"] >= volume_surge_threshold,
            df["OBV"].pct_change() > 0.1,
            df["CMF"] > 0,
        ]
        volume_condition = volume_conditions[0] | safe_condition_check(
            df, volume_conditions[1:]
        )

        df_spikes = df[price_condition].copy()
        df_volume_surges = df[volume_condition].copy()

        columns_spikes = [
            "symbol",
            "slug",
            "name",
            "quote.USD.percent_change_24h",
            "quote.USD.price",
            "MACD_hist",
            "ROC",
            "timestamp",
        ]
        columns_volume = ["symbol", "quote.USD.volume_change_24h", "OBV", "CMF"]

        df_combined = pd.merge(
            df_spikes[columns_spikes],
            df_volume_surges[columns_volume],
            on="symbol",
            how="inner",
        )

        return df_combined
    except Exception as e:
        print(f"Error in price_spikes_and_volume_surges: {str(e)}")
        return pd.DataFrame()


def calculate_efficiency_metric(df_combined, price_weight, volume_weight):
    try:
        if df_combined.empty:
            return df_combined

        df_combined["ta_price_score"] = df_combined["MACD_hist"] / (
            df_combined["MACD_hist"].abs().mean() or 1
        ) + df_combined["ROC"] / (df_combined["ROC"].abs().mean() or 1)

        df_combined["ta_volume_score"] = (
            df_combined["CMF"] * 100 + df_combined["OBV"].pct_change().fillna(0) * 100
        )

        # Calculate final score
        df_combined["spike_surge_score"] = price_weight * (
            df_combined["quote.USD.percent_change_24h"] + df_combined["ta_price_score"]
        ) + volume_weight * (
            df_combined["quote.USD.volume_change_24h"] + df_combined["ta_volume_score"]
        )

        return df_combined.sort_values(by="spike_surge_score", ascending=False)
    except Exception as e:
        print(f"Error in calculate_efficiency_metric: {str(e)}")
        return df_combined


def overbought_conditions(df, price_spike_threshold):
    condition = (df["quote.USD.percent_change_24h"] > price_spike_threshold) & (
        (df["RSI"] > 70) | (df["BB_%B"] > 1) | ((df["EMA_9"] / df["EMA_20"]) > 1.05)
    )

    df_overbought = df[condition].copy()

    df_overbought["overbought_score"] = (
        df_overbought["quote.USD.percent_change_24h"]
        * (df_overbought["RSI"] - 70)
        * (1 + df_overbought["BB_%B"])
        * (df_overbought["EMA_9"] / df_overbought["EMA_20"])
    )

    return df_overbought.sort_values(by="overbought_score", ascending=False)


def pump_and_dump_schemes(df, pump_threshold, dump_threshold, volume_surge_threshold):
    pump_condition = (df["quote.USD.percent_change_24h"] > pump_threshold) & (
        (df["ROC"] > 0) | (df["MACD_hist"] > 0)
    )

    dump_condition = (df["quote.USD.percent_change_1h"] < dump_threshold) & (
        (df["ROC"] < 0) | (df["MACD_hist"] < 0)
    )

    volume_condition = (df["quote.USD.volume_change_24h"] > volume_surge_threshold) & (
        df["CMF"].abs() > df["CMF"].abs().mean()
    )

    df_pump = df[pump_condition]
    df_dump = df[dump_condition]
    df_volume = df[volume_condition]

    df_combined = df_pump.merge(df_dump, on="symbol", suffixes=("_pump", "_dump"))
    df_combined = df_combined.merge(df_volume, on="symbol")

    df_combined["pump_dump_score"] = (
        df_combined["quote.USD.percent_change_24h_pump"].abs()
        + df_combined["quote.USD.percent_change_1h_dump"].abs()
        + df_combined["quote.USD.volume_change_24h"] / 100
        + df_combined["ROC_pump"].abs()
        + abs(df_combined["MACD_hist_pump"])
        + df_combined["CMF"].abs()
    )

    return df_combined.sort_values(by="pump_dump_score", ascending=False)


def reversal_opportunities(df, crash_threshold, low_rsi_threshold):
    condition = (df["quote.USD.percent_change_24h"] < crash_threshold) & (
        (df["RSI"] < low_rsi_threshold)
        | (df["BB_%B"] < 0)
        | (df["MACD_hist"] > df["MACD_hist"].shift(1))
    )

    df_reversal = df[condition].copy()

    df_reversal.loc[:, "reversal_score"] = (
        df_reversal["quote.USD.percent_change_24h"].abs()
        * (low_rsi_threshold - df_reversal["RSI"])
        * (1 + abs(df_reversal["BB_%B"]))
        * (1 + df_reversal["MOM"].abs() / df_reversal["MOM"].abs().mean())
    )

    return df_reversal.sort_values(by="reversal_score", ascending=False)


def false_valuation_market_manipulation(df, lower_threshold, upper_threshold):
    new_columns = {}
    new_columns["volume_to_market_cap_ratio"] = (
        df["quote.USD.volume_24h"] / df["quote.USD.market_cap"]
    )
    new_columns["price_to_ATR_ratio"] = df["quote.USD.price"] / df["ATR"]
    pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    df["liquidity_score"] = 1 / df["volume_to_market_cap_ratio"]
    df["volatility_score"] = df["ATR"] / df["quote.USD.price"]
    df["momentum_divergence"] = abs(df["MOM"] - df["ROC"])

    df["false_valuation_score"] = (
        0.4 * df["liquidity_score"]
        + 0.3 * df["volatility_score"]
        + 0.3 * df["momentum_divergence"]
    )

    df_sorted = df.sort_values(by="false_valuation_score", ascending=False)

    df_sorted["efficiency_flag"] = df_sorted["false_valuation_score"].apply(
        lambda x: 1 if lower_threshold < x < upper_threshold else 0
    )

    return df_sorted


def bullish_momentum_breakouts(df, p_s_t, v_s_t, p_w, v_w):
    try:
        df_combined = price_spikes_and_volume_surges(df, p_s_t, v_s_t)
        if df_combined.empty:
            return pd.DataFrame()

        df_sorted = calculate_efficiency_metric(df_combined, p_w, v_w)

        columns = [
            "name",
            "slug",
            "symbol",
            "spike_surge_score",
            "quote.USD.price",
            "quote.USD.percent_change_24h",
            "quote.USD.volume_change_24h",
            "MACD_hist",
            "ROC",
            "CMF",
            "timestamp",
        ]

        return df_sorted[columns]
    except Exception as e:
        print(f"Error in bullish_momentum_breakouts: {str(e)}")
        return pd.DataFrame()


def apply_tandem_filters(df):
    df = add_technical_indicators(df)

    return {
        "Bullish": bullish_momentum_breakouts(
            df, Config.p_s_t, Config.v_s_t, Config.p_w, Config.v_w
        ),
        "Overbought": overbought_conditions(df, Config.o_p_s_t),
        "PumpAndDump": pump_and_dump_schemes(
            df, Config.p_t, Config.d_t, Config.p_d_v_s_t
        ),
        "ReversalOpportunities": reversal_opportunities(df, Config.c_t, Config.l_rsi_t),
        "FalseValuation": false_valuation_market_manipulation(
            df, Config.l_t, Config.u_t
        ),
    }
