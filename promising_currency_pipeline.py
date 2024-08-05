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
    df["market_dominance"] = df["quote.USD.market_cap"] / df["quote.USD.market_cap"].sum()
    df["volume_stability"] = df["quote.USD.volume_24h"].rolling(window=7).std() / df["quote.USD.volume_24h"].rolling(window=7).mean()
    df["price_stability"] = df["quote.USD.price"].rolling(window=7).std() / df["quote.USD.price"].rolling(window=7).mean()

    # Create trend indicators
    df["uptrend"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
    df["golden_cross"] = ((df["SMA_50"] > df["SMA_200"]) & (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1))).astype(int)

    # Modified ADX calculation
    close = df["quote.USD.price"].values
    
    df["ADX"] = talib.ADX(close, close, close, timeperiod=14)  # This isn't ideal but will give a trend strength indication

    # Add OBV (On-Balance Volume)
    df["OBV"] = talib.OBV(close, df["quote.USD.volume_24h"].values)

    return df


def detect_anomalies(df):
    # Select features for anomaly detection
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
    
    # Verify all features exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    missing_features = set(features) - set(available_features)
    
    if missing_features:
        print(f"Warning: The following features are missing and will be excluded: {missing_features}")
    
    # Use only available features
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
    
    # Detect anomalies
    df = detect_anomalies(df)

    # Calculate composite score
    df["promise_score"] = (
        df["market_dominance"] * 0.10
        + (1 - df["volume_stability"]) * 0.05
        + (1 - df["price_stability"]) * 0.05
        + df["quote.USD.percent_change_24h"] * 0.10
        + df["quote.USD.percent_change_7d"] * 0.10
        + (df["RSI"] - 50).abs() / 50 * 0.10
        + df["uptrend"] * 0.05
        + df["golden_cross"] * 0.05
        + df["MACD_hist"] * 0.10
        + df["BB_width"] * 0.05
        + df["ADX"] / 100 * 0.10  # ADX ranges from 0 to 100
        + df["OBV"].pct_change() * 0.05  # OBV momentum
        + df["flag_severity"] * 0.10
    )

    # Penalize currencies with too many negative flags
    df["promise_score"] = np.where(df["negative_flags"] > 3, df["promise_score"] * 0.5, df["promise_score"])

    # Bonus for currencies with no negative flags
    df["promise_score"] = np.where(df["negative_flags"] == 0, df["promise_score"] * 1.2, df["promise_score"])

    # Penalize anomalies
    df["promise_score"] = np.where(df["anomaly"] == -1, df["promise_score"] * 0.8, df["promise_score"])

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
        "MACD_hist",
        "BB_width",
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
        "anomaly",
        "timestamp",
    ] + [col for col in promising_currencies.columns if col.endswith("_flag")]

    return promising_currencies[columns_to_keep].sort_values(
        "promise_score", ascending=False
    )


def performers():
    raw_data_path = f"C:/Users/{os.getenv('USER')}/Desktop/CryptoAPI.csv"

    # Identify promising currencies
    promising_currencies = identify_promising_currencies(raw_data_path)

    promising_currencies["timestamp"] = promising_currencies["timestamp"].dt.strftime('%Y-%m-%dT%H:%M:%S')

    # Save the full results to CSV
    output_path = (
        rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/PromisingCurrencies.csv"
    )
    promising_currencies.to_csv(
        output_path, mode="a", header=not os.path.exists(output_path), index=False
    )

    # Get the names of the top 5 performing currencies
    top_5_performers = promising_currencies["name"].head(5).tolist()

    return top_5_performers


# This part is for testing the function independently
if __name__ == "__main__":
    top_performers = performers()
    print("Top 5 performing currencies:", top_performers)
