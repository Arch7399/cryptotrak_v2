import pandas as pd
from config import Config


def price_spikes_and_volume_surges(df, price_spike_threshold, volume_surge_threshold):
    df_spikes = df[df["quote.USD.percent_change_24h"] >= price_spike_threshold]
    df_volume_surges = df[df["quote.USD.volume_change_24h"] >= volume_surge_threshold]

    df_combined = pd.merge(
        df_spikes[
            [
                "symbol",
                "slug",
                "name",
                "quote.USD.percent_change_24h",
                "quote.USD.price",
                "timestamp",
            ]
        ],
        df_volume_surges[["symbol", "quote.USD.volume_change_24h"]],
        on="symbol",
    )

    return df_combined


def calculate_efficiency_metric(df_combined, price_weight, volume_weight):
    df_combined["spike_surge_score"] = (
        price_weight * df_combined["quote.USD.percent_change_24h"]
        + volume_weight * df_combined["quote.USD.volume_change_24h"]
    )

    df_combined = df_combined.sort_values(by="spike_surge_score", ascending=False)

    return df_combined


def bullish_momentum_breakouts(df, p_s_t, v_s_t, p_w, v_w):
    df_combined = price_spikes_and_volume_surges(df, p_s_t, v_s_t)
    df_sorted = calculate_efficiency_metric(df_combined, p_w, v_w)

    return df_sorted[
        [
            "name",
            "slug",
            "symbol",
            "spike_surge_score",
            "quote.USD.price",
            "quote.USD.percent_change_24h",
            "quote.USD.volume_change_24h",
            "timestamp",
        ]
    ]


def overbought_conditions(df, price_spike_threshold):

    df_spikes = df[df["quote.USD.percent_change_24h"] > price_spike_threshold]
    df_high_rsi = df[df["RSI"] > 70]

    df_spikes = df_spikes[
        [
            "symbol",
            "slug",
            "name",
            "quote.USD.percent_change_24h",
            "quote.USD.price",
            "timestamp",
        ]
    ]
    df_high_rsi = df_high_rsi[["symbol", "RSI"]]

    df_combined = df_spikes.merge(df_high_rsi, how="inner", on="symbol")
    df_combined["overbought_score"] = df_combined["quote.USD.percent_change_24h"] * (
        df_combined["RSI"] - 70
    )
    df_combined = df_combined.sort_values(by="overbought_score", ascending=False)

    return df_combined[
        [
            "name",
            "slug",
            "symbol",
            "overbought_score",
            "quote.USD.percent_change_24h",
            "quote.USD.price",
            "RSI",
            "timestamp",
        ]
    ]


def pump_and_dump_schemes(df, pump_threshold, dump_threshold, volume_surge_threshold):
    df_pump = df[df["quote.USD.percent_change_24h"] > pump_threshold]
    df_dump = df[df["quote.USD.percent_change_1h"] < dump_threshold]
    df_volume_surges = df[df["quote.USD.volume_change_24h"] > volume_surge_threshold]

    df_combined = df_pump.merge(df_dump, on="symbol", suffixes=("_pump", "_dump"))
    df_combined = df_combined.merge(df_volume_surges, on="symbol")

    df_combined["pump_dump_score"] = (
        df_combined["quote.USD.percent_change_24h_pump"].abs()
        + df_combined["quote.USD.percent_change_1h_dump"].abs()
        + df_combined["quote.USD.volume_change_24h"] / 100
    )

    df_combined = df_combined.sort_values(by="pump_dump_score", ascending=False)

    return df_combined[
        [
            "name",
            "slug",
            "symbol",
            "pump_dump_score",
            "quote.USD.price",
            "quote.USD.percent_change_24h_pump",
            "quote.USD.percent_change_1h_dump",
            "quote.USD.volume_change_24h",
            "timestamp",
        ]
    ]


def reversal_opportunities(df, crash_threshold, low_rsi_threshold):
    df_crashes = df[df["quote.USD.percent_change_24h"] < crash_threshold]
    df_low_rsi = df[df["RSI"] < low_rsi_threshold]

    df_crashes = df_crashes[
        [
            "symbol",
            "slug",
            "name",
            "quote.USD.price",
            "quote.USD.percent_change_24h",
            "timestamp",
        ]
    ]
    df_low_rsi = df_low_rsi[["symbol", "RSI"]]

    df_combined = df_crashes.merge(df_low_rsi, on="symbol")

    df_combined["reversal_score"] = df_combined[
        "quote.USD.percent_change_24h"
    ].abs() * (low_rsi_threshold - df_combined["RSI"])

    df_combined = df_combined.sort_values(by="reversal_score", ascending=False)

    return df_combined[
        [
            "name",
            "slug",
            "symbol",
            "reversal_score",
            "quote.USD.price",
            "quote.USD.percent_change_24h",
            "RSI",
            "timestamp",
        ]
    ]


def false_valuation_market_manipulation(df, lower_threshold, upper_threshold):

    df["quote.USD.volume_24h"] = pd.to_numeric(
        df["quote.USD.volume_24h"], errors="coerce"
    )
    df["quote.USD.market_cap"] = pd.to_numeric(
        df["quote.USD.market_cap"], errors="coerce"
    )

    df["volume_to_market_cap_ratio"] = (
        df["quote.USD.volume_24h"] / df["quote.USD.market_cap"]
    )

    df["liquidity_score"] = 1 / df["volume_to_market_cap_ratio"]

    df["discrepancy_score"] = df["quote.USD.market_cap"] / df["quote.USD.volume_24h"]

    df["false_valuation_score"] = (0.7 * df["liquidity_score"]) + (
        0.3 * df["discrepancy_score"]
    )

    df_sorted = df.sort_values(by="false_valuation_score", ascending=False)

    df_sorted["efficiency_flag"] = df_sorted["false_valuation_score"].apply(
        lambda x: 1 if lower_threshold < x < upper_threshold else 0
    )

    return df_sorted[
        [
            "name",
            "slug",
            "symbol",
            "quote.USD.price",
            "false_valuation_score",
            "efficiency_flag",
            "timestamp",
        ]
    ]


def apply_tandem_filters(df):
    return {
        "Bullish": bullish_momentum_breakouts(
            df, Config.p_s_t, Config.v_s_t, Config.p_w, Config.v_w
        ),
        "Overbought": overbought_conditions(df, Config.o_p_s_t),
        "PumpAndDump": pump_and_dump_schemes(
            df, Config.p_t, Config.d_t, Config.p_d_v_s_t
        ),
        "ReversalOpportunities": reversal_opportunities(
            df, Config.c_t, Config.l_rsi_t
        ),
        "FalseValuation": false_valuation_market_manipulation(
            df, Config.l_t, Config.u_t
        ),
    }
