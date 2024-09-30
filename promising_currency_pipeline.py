import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from data_processing import process_data
from pipeline_filters import apply_filters
from pipeline_mixed_filters import apply_tandem_filters
from datetime import datetime, timedelta
from latest_data import filter_latest
from config import Config
import talib


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

    # Create trend indicators
    df["uptrend"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
    df["golden_cross"] = (
        (df["SMA_50"] > df["SMA_200"])
        & (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1))
    ).astype(int)

    # ADX calculation
    close = df["quote.USD.price"].values

    df["ADX"] = talib.ADX(close, close, close, timeperiod=14)

    # Add OBV (On-Balance Volume)
    df["OBV"] = talib.OBV(close, df["quote.USD.volume_24h"].values)

    return df


def detect_anomalies(df):
    features = [
        "market_dominance",
        "volume_stability",
        "price_stability",
        "quote.USD.percent_change_24h",
        "quote.USD.percent_change_7d",
        "RSI",
        "MACD_hist",
        "BB_width",
    ]

    available_features = [f for f in features if f in df.columns]
    missing_features = set(features) - set(available_features)

    if missing_features:
        print(
            f"Warning: The following features are missing and will be excluded: {missing_features}"
        )

    features = available_features

    # Create a SimpleImputer
    imputer = SimpleImputer(strategy="mean")

    # Fit and transform the data
    X = imputer.fit_transform(df[features])

    # Initialize and fit the Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df["anomaly"] = iso_forest.fit_predict(X)

    return df


def score_currencies(df):
    z_score_columns = [
        "quote.USD.percent_change_24h",
        "quote.USD.percent_change_7d",
        "quote.USD.volume_change_24h",
        "volatility",
    ]
    df = calculate_z_scores(df, z_score_columns)

    df = calculate_time_decay(df)
    df = calculate_flag_severity(df)

    df = improve_promising_currency_code(df)

    df = detect_anomalies(df)

    # Initial promise score calculation
    df["promise_score"] = (
        df["market_dominance"] * 0.08
        + (1 - df["volume_stability"]) * 0.04
        + (1 - df["price_stability"]) * 0.04
        + df["quote.USD.percent_change_24h"] * 0.08
        + df["quote.USD.percent_change_7d"] * 0.08
        + (df["RSI"] - 50).abs() / 50 * 0.08
        + df["uptrend"] * 0.04
        + df["golden_cross"] * 0.04
        + df["MACD_hist"] * 0.08
        + df["BB_width"] * 0.04
        + df["ADX"] / 100 * 0.08
        + df["OBV"].pct_change() * 0.04
        + df["flag_severity"] * 0.12  # Increased weight for flag severity
        + df["CMF"] * 0.02
        + df["market_stability_index"] * 0.02
        + df["vol_price_correlation_normalized"] * 0.02
        + df["price_to_ATR_ratio"].clip(lower=0, upper=1) * 0.02
    )

    # Apply initial penalties
    volatility_penalty = np.where(
        df["volatility_score"] > 0.5,
        df["promise_score"] * (1 - df["volatility_score"] * 0.5),
        df["promise_score"],
    )
    df["promise_score"] = volatility_penalty

    volume_adjustment = np.where(
        (df["volume_stability"] < 0.3) & (df["volume_dominance"] > 0.01),
        df["promise_score"] * 1.1,
        df["promise_score"] * (0.9 - df["volume_stability"] * 0.2),
    )
    df["promise_score"] = volume_adjustment

    mcap_volume_adjustment = np.where(
        (df["vol_mcap_ratio"] > 0.1) & (df["vol_mcap_ratio"] < 0.5),
        df["promise_score"] * 1.05,
        df["promise_score"] * 0.85,
    )
    df["promise_score"] = mcap_volume_adjustment

    # Handle any negative scores
    df["promise_score"] = df["promise_score"].clip(lower=0)

    # # First scaling to 0-100
    # scaler = MinMaxScaler(feature_range=(0, 100))
    # df["promise_score"] = scaler.fit_transform(df[["promise_score"]])

    # Enhanced flag-based penalties AFTER scaling
    def calculate_flag_based_penalty(row):
        base_score = row["promise_score"]

        # Calculate severity ratio (positive to negative flags)
        severity_ratio = (row["positive_flags"] + 1) / (row["negative_flags"] + 1)

        # Define penalty based on negative flags count and severity
        if row["negative_flags"] == 0:
            # Bonus for positive flags when no negative flags
            bonus = min(row["positive_flags"] * 5, 20)  # Up to 20% bonus
            return min(base_score * (1 + bonus / 100), 100)

        elif row["negative_flags"] == 1:
            # Less severe penalty if balanced by positive flags
            penalty_factor = 0.7 if severity_ratio > 1 else 0.6
            return base_score * penalty_factor

        elif row["negative_flags"] == 2:
            # Moderate penalty, slightly reduced if many positive flags
            penalty_factor = 0.4 if severity_ratio > 1.5 else 0.3
            return base_score * penalty_factor

        else:  # 3+ negative flags
            # Severe penalty, very slightly reduced if exceptional positive flags
            penalty_factor = 0.2 if severity_ratio > 2 else 0.1
            return base_score * penalty_factor

    df["promise_score"] = df.apply(calculate_flag_based_penalty, axis=1)

    # Specific flag type penalties
    def apply_specific_flag_penalties(row):
        score = row["promise_score"]

        # Severe negative flags (additional penalties)
        if row["pump_flag"] or row["dump_flag"]:
            score *= 0.7  # 30% reduction for pump/dump flags

        if row["market_cap_volume_discrepancy_flag"]:
            score *= 0.8  # 20% reduction for market cap discrepancy

        if row["false_valuation_flag"]:
            score *= 0.75  # 25% reduction for false valuation

        # Positive flag bonuses (only if negative flags <= 1)
        if row["negative_flags"] <= 1:
            if row["bullish_momentum_breakout_flag"]:
                score = min(score * 1.1, 100)  # 10% bonus

            if row["reversal_opportunity_flag"]:
                score = min(score * 1.05, 100)  # 5% bonus

        return score

    df["promise_score"] = df.apply(apply_specific_flag_penalties, axis=1)

    # Final adjustments
    df["promise_score"] = np.where(
        df["anomaly"] == -1, df["promise_score"] * 0.6, df["promise_score"]
    )

    df["promise_score"] = np.where(
        (df["RSI"] > 80) | (df["RSI"] < 20),
        df["promise_score"] * 0.75,
        df["promise_score"],
    )

    # # Ensure final scores are within 0-100 range
    # df["promise_score"] = df["promise_score"].clip(0, 100).round(2)

    df = df[df["quote.USD.price"].between(Config.min_usd_price, Config.max_usd_price)]
    return df


def identify_promising_currencies(raw_data_path):
    df = prepare_data(raw_data_path)
    df_processed = process_data(df)

    df_filtered = apply_filters(df_processed)

    df_tandem_filtered = apply_tandem_filters(df_filtered)

    df_scored = score_currencies(df_tandem_filtered)

    promising_currencies = df_scored.nlargest(30, "promise_score")

    return promising_currencies.sort_values("promise_score", ascending=False)


def performers():
    raw_data_path = f"C:/Users/{os.getenv('USER')}/Desktop/CryptoAPI.csv"

    promising_currencies = identify_promising_currencies(raw_data_path)

    # promising_currencies["timestamp"] = promising_currencies["timestamp"].dt.strftime(
    #     "%Y-%m-%dT%H:%M:%S"
    # )

    output_path = (
        rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/PromisingCurrencies.csv"
    )
    promising_currencies.to_csv(
        output_path, mode="a", header=not os.path.exists(output_path), index=False
    )

    top_5_performers = promising_currencies["name"].head(5).tolist()

    return top_5_performers


if __name__ == "__main__":
    top_performers = performers()
    print("Top 5 performing currencies:", top_performers)
