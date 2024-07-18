import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from data_processing import process_data
from pipeline_filters import apply_filters
from pipeline_mixed_filters import apply_tandem_filters
from datetime import datetime, timedelta
from latest_data import filter_latest


def prepare_data(raw_data_path):
    df = pd.read_csv(raw_data_path)
    df = filter_latest(df)
    df = df.sort_values("timestamp")
    return df


def calculate_z_scores(df, columns):
    for col in columns:
        df[f"{col}_zscore"] = stats.zscore(df[col])
    return df


def calculate_time_decay(df, half_life=timedelta(hours=6)):
    now = df["timestamp"].max()
    df["time_decay"] = np.exp(-np.log(2) * (now - df["timestamp"]) / half_life)
    return df


def calculate_flag_severity(df):
    positive_flags = [
        "price_spike_flag",
        "volume_surge_flag",
        "bullish_momentum_breakout_flag",
        "reversal_opportunity_flag",
    ]
    negative_flags = [
        "price_crash_flag",
        "low_liquidity_flag",
        "pump_flag",
        "dump_flag",
        "market_cap_volume_discrepancy_flag",
        "false_valuation_flag",
    ]

    df["positive_flags"] = df[positive_flags].sum(axis=1)
    df["negative_flags"] = df[negative_flags].sum(axis=1)

    df["flag_severity"] = df["positive_flags"] - df["negative_flags"] * 1.5
    return df
    pass


def improve_promising_currency_code(df):
    # Add more relevant features
    df["market_dominance"] = (
        df["quote.USD.market_cap"] / df["quote.USD.market_cap"].sum()
    )
    df["volume_stability"] = (
        df["quote.USD.volume_24h"].rolling(window=7).std()
        / df["quote.USD.volume_24h"].rolling(window=7).mean()
    )
    df["price_stability"] = (
        df["quote.USD.price"].rolling(window=7).std()
        / df["quote.USD.price"].rolling(window=7).mean()
    )

    # Calculate technical indicators
    df["SMA_50"] = df["quote.USD.price"].rolling(window=50).mean()
    df["SMA_200"] = df["quote.USD.price"].rolling(window=200).mean()
    df["EMA_20"] = df["quote.USD.price"].ewm(span=20, adjust=False).mean()

    # Create trend indicators
    df["uptrend"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
    df["golden_cross"] = (
        (df["SMA_50"] > df["SMA_200"])
        & (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1))
    ).astype(int)

    # Normalize features
    scaler = MinMaxScaler()
    features_to_normalize = [
        "market_dominance",
        "volume_stability",
        "price_stability",
        "quote.USD.percent_change_24h",
        "quote.USD.percent_change_7d",
        "RSI",
    ]
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

    return df


def score_currencies(df):
    # Calculate z-scores for relevant metrics
    z_score_columns = [
        "quote.USD.percent_change_24h",
        "quote.USD.percent_change_7d",
        "quote.USD.volume_change_24h",
        "volume_to_market_cap_ratio",
        "volatility",
    ]
    df = calculate_z_scores(df, z_score_columns)

    # Calculate time decay factor
    df = calculate_time_decay(df)

    # Calculate flag severity
    df = calculate_flag_severity(df)

    # Apply improvements
    df = improve_promising_currency_code(df)

    # Calculate composite score
    df["promise_score"] = (
        df["market_dominance"] * 0.2
        + (1 - df["volume_stability"]) * 0.15
        + (1 - df["price_stability"]) * 0.15
        + df["quote.USD.percent_change_24h"] * 0.1
        + df["quote.USD.percent_change_7d"] * 0.1
        + (df["RSI"] - 50).abs() / 50 * 0.1
        + df["uptrend"] * 0.1
        + df["golden_cross"] * 0.1
        + df["flag_severity"] * 0.1  # Include flag severity in the score
    )

    # Penalize currencies with too many negative flags
    df["promise_score"] = np.where(
        df["negative_flags"] > 3, df["promise_score"] * 0.5, df["promise_score"]
    )

    # Bonus for currencies with no negative flags
    df["promise_score"] = np.where(
        df["negative_flags"] == 0, df["promise_score"] * 1.2, df["promise_score"]
    )

    return df


def identify_promising_currencies(raw_data_path):
    # Load and process the raw data
    df = prepare_data(raw_data_path)
    df_processed = process_data(df)

    # Apply individual filters
    df_filtered = apply_filters(df_processed)

    # Apply tandem filters
    df_tandem_filtered = apply_tandem_filters(df_filtered)

    # Score currencies
    df_scored = score_currencies(df_tandem_filtered)

    # Identify promising currencies
    promising_currencies = df_scored.nlargest(20, "promise_score")

    # Select relevant columns
    columns_to_keep = [
        "name",
        "slug",
        "symbol",
        "promise_score",
        "market_dominance",
        "volume_stability",
        "price_stability",
        "uptrend",
        "golden_cross",
        "RSI",
        "quote.USD.price",
        "quote.USD.market_cap",
        "quote.USD.volume_24h",
        "quote.USD.percent_change_24h",
        "quote.USD.percent_change_7d",
        "circulating_supply",
        "max_supply",
        "circulating_to_max_supply_ratio",
        "positive_flags",
        "negative_flags",
        "flag_severity",
        "time_decay",
        "timestamp",
    ] + [col for col in promising_currencies.columns if col.endswith("_flag")]

    return promising_currencies[columns_to_keep].sort_values(
        "promise_score", ascending=False
    )


def performers():
    raw_data_path = f"C:/Users/{os.getenv('USER')}/Desktop/CryptoAPI.csv"

    # Identify promising currencies
    promising_currencies = identify_promising_currencies(raw_data_path)

    # Save the full results to CSV
    if not os.path.isfile(
        rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/PromisingCurrencies.csv"
    ):
        promising_currencies.to_csv(
            rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/PromisingCurrencies.csv",
            header="column_names",
        )
    else:
        promising_currencies.to_csv(
            rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/PromisingCurrencies.csv",
            mode="a",
            header=False,
        )

    # Get the names of the top 5 performing currencies
    top_5_performers = promising_currencies["name"].head(5).tolist()

    return top_5_performers


# This part is for testing the function independently
if __name__ == "__main__":
    top_performers = performers()
    print("Top 5 performing currencies:", top_performers)
