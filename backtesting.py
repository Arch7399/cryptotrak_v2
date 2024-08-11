import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import os
from dotenv import load_dotenv
import talib

load_dotenv()

def load_data(
    price_diff_file: str, promising_currencies_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the price difference and promising currencies data.
    Includes improved handling of missing values and ensures data quality.
    """
    # Load price difference data
    price_diff_df = pd.read_csv(price_diff_file, index_col="slug")
    
    # Convert column names to datetime
    price_diff_df.columns = pd.to_datetime(price_diff_df.columns)
    
    # Handle missing values
    price_diff_df = price_diff_df.replace('', np.nan)
    
    # Convert to float and handle any conversion errors
    price_diff_df = price_diff_df.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows where all values are NaN
    price_diff_df = price_diff_df.dropna(how='all')
    
    # Drop columns where all values are NaN
    price_diff_df = price_diff_df.dropna(axis=1, how='all')
    
    # Optionally, fill remaining NaN values with forward fill then backward fill
    price_diff_df = price_diff_df.fillna(method='ffill').fillna(method='bfill')
    
    # Load promising currencies
    promising_currencies_df = pd.read_csv(promising_currencies_file)
    promising_currencies_df["timestamp"] = pd.to_datetime(promising_currencies_df["timestamp"])

    # Print data quality information
    print(f"\nData Quality Report:")
    print(f"Original price_diff shape: {price_diff_df.shape}")
    print(f"Missing values remaining: {price_diff_df.isna().sum().sum()}")
    print(f"Number of currencies with complete data: {(price_diff_df.notna().all(axis=1)).sum()}")
    print(f"Number of timepoints with complete data: {(price_diff_df.notna().all(axis=0)).sum()}")

    return price_diff_df.sort_index(axis=1), promising_currencies_df


def calculate_returns(
    df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.Series:
    """
    Calculate percentage returns for a specific time period, handling missing values.
    """
    if start_date not in df.columns or end_date not in df.columns:
        print(f"Warning: {start_date} or {end_date} not found in price_diff_df")
        return pd.Series()

    start_prices = df[start_date]
    end_prices = df[end_date]
    
    # Handle missing values
    valid_mask = ~(start_prices.isna() | end_prices.isna())
    returns = pd.Series(np.nan, index=df.index)
    
    # Calculate returns only for valid pairs
    returns[valid_mask] = (end_prices[valid_mask] - start_prices[valid_mask]) / start_prices[valid_mask]
    
    return returns.replace([np.inf, -np.inf], np.nan)


def generate_signals(
    promising_currencies: pd.DataFrame, timestamp: pd.Timestamp
) -> pd.Series:
    """
    Generate buy/sell signals based on the promising currencies at a specific timestamp.
    """
    # Find the closest timestamp that's not after the target timestamp
    valid_timestamps = promising_currencies["timestamp"][promising_currencies["timestamp"] <= timestamp]
    if valid_timestamps.empty:
        return pd.Series()
    
    closest_timestamp = valid_timestamps.max()
    
    currencies = promising_currencies[promising_currencies["timestamp"] == closest_timestamp]["slug"]
    return pd.Series(1, index=currencies)

def calculate_technical_score(indicators: Dict[str, Dict[str, float]]) -> float:
    """
    Calculate a composite technical score based on various indicators,
    with improved handling of missing or invalid data.
    """
    if not indicators:
        return np.nan

    scores = []
    for currency, currency_indicators in indicators.items():
        if all(np.isnan(value) for value in currency_indicators.values()):
            continue
        
        score = 0
        weight_sum = 0
        
        # RSI score (closer to 50 is better)
        if "RSI" in currency_indicators and not np.isnan(currency_indicators["RSI"]):
            rsi_score = 1 - abs(currency_indicators["RSI"] - 50) / 50
            score += rsi_score * 0.3
            weight_sum += 0.3
        
        # MACD histogram score
        if "MACD_hist" in currency_indicators and not np.isnan(currency_indicators["MACD_hist"]):
            macd_hist_normalized = np.tanh(currency_indicators["MACD_hist"])
            score += (macd_hist_normalized + 1) / 2 * 0.3
            weight_sum += 0.3
        
        # Bollinger Band width score
        if "BB_width" in currency_indicators and not np.isnan(currency_indicators["BB_width"]):
            bb_score = 1 / (1 + currency_indicators["BB_width"])
            score += bb_score * 0.2
            weight_sum += 0.2
        
        # Trend score
        if ("SMA_50" in currency_indicators and not np.isnan(currency_indicators["SMA_50"]) and
            "EMA_20" in currency_indicators and not np.isnan(currency_indicators["EMA_20"])):
            trend_score = int(currency_indicators["EMA_20"] > currency_indicators["SMA_50"])
            score += trend_score * 0.2
            weight_sum += 0.2
        
        if weight_sum > 0:
            scores.append(score / weight_sum)  # Normalize by weight sum
    
    return np.mean(scores) if scores else np.nan

def calculate_technical_indicators(price_series: pd.Series) -> Dict[str, float]:
    """
    Calculate technical indicators for a given price series with improved error handling
    and minimum data point requirements.
    """
    # Remove NaN values
    valid_prices = price_series.dropna()
    
    # Check if we have enough data points
    if len(valid_prices) < 20:  # Minimum required for most indicators
        return {
            "SMA_50": np.nan,
            "EMA_20": np.nan,
            "MACD_hist": np.nan,
            "RSI": np.nan,
            "BB_width": np.nan
        }
    
    try:
        close = valid_prices.values
        result = {}
        
        # Calculate indicators with appropriate window sizes
        sma_period = min(50, len(close))
        ema_period = min(20, len(close))
        rsi_period = min(14, len(close))
        
        # Calculate basic indicators
        result["SMA_50"] = talib.SMA(close, timeperiod=sma_period)[-1]
        result["EMA_20"] = talib.EMA(close, timeperiod=ema_period)[-1]
        
        # MACD
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        result["MACD_hist"] = hist[-1] if not np.isnan(hist[-1]) else 0
        
        # RSI
        result["RSI"] = talib.RSI(close, timeperiod=rsi_period)[-1]
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=ema_period, nbdevup=2, nbdevdn=2)
        result["BB_width"] = (upper[-1] - lower[-1]) / middle[-1] if not np.isnan(middle[-1]) and middle[-1] != 0 else 0
        
        return result
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return {
            "SMA_50": np.nan,
            "EMA_20": np.nan,
            "MACD_hist": np.nan,
            "RSI": np.nan,
            "BB_width": np.nan
        }


def simulate_trading(signals: pd.Series, returns: pd.Series) -> pd.Series:
    """
    Simulate trading based on signals and calculate portfolio returns.
    """
    return signals * returns


def evaluate_strategy(signals: pd.Series, returns: pd.Series, technical_indicators: Dict[str, Dict[str, float]] = None) -> Dict[str, float]:
    """
    Evaluate the trading strategy using various performance metrics and technical indicators.
    Technical indicators parameter is now optional with a default value of None.
    """
    if signals.empty or returns.empty:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "technical_score": np.nan,
        }

    true_labels = (returns > 0).astype(int)
    predicted_labels = (signals > 0).astype(int)

    # Align indices
    true_labels, predicted_labels = true_labels.align(predicted_labels, join="inner")

    # Remove NaN values
    mask = ~np.isnan(true_labels) & ~np.isnan(predicted_labels)
    true_labels = true_labels[mask]
    predicted_labels = predicted_labels[mask]

    if len(true_labels) == 0:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "technical_score": np.nan,
        }

    # Calculate technical score only if technical indicators are provided
    technical_score = calculate_technical_score(technical_indicators) if technical_indicators else np.nan

    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, average="binary", zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, average="binary", zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, average="binary", zero_division=0),
        "technical_score": technical_score
    }


def analyze_currency_changes(
    prev_currencies: set, current_currencies: set
) -> Tuple[set, set]:
    """
    Analyze newly predicted and dropped currencies.
    """
    newly_predicted = current_currencies - prev_currencies
    dropped_predictions = prev_currencies - current_currencies
    return newly_predicted, dropped_predictions


def calculate_dynamic_windows(timestamps: np.ndarray) -> Tuple[int, int]:
    """
    Dynamically calculate appropriate window sizes based on available data.
    
    Returns:
    - window_size: The main window size for long-term analysis
    - short_term_window: The window size for short-term analysis
    """
    num_timestamps = len(timestamps)
    
    if num_timestamps <= 2:
        return 1, 1
    
    # Convert numpy timestamps to pandas timestamps if necessary
    if isinstance(timestamps[0], np.datetime64):
        timestamps = pd.to_datetime(timestamps)
    
    # Calculate time deltas between timestamps
    time_deltas = pd.Series(timestamps).diff().dropna()
    
    # Calculate median in a way that works with pandas timedelta
    median_delta = time_deltas.median()
    median_hours = median_delta.total_seconds() / 3600
    
    # Rest of the function remains the same
    if median_hours <= 1:  # Hourly or more frequent data
        window_size = min(24, num_timestamps // 3)
        short_term_window = min(4, window_size // 2)
    elif median_hours <= 24:  # Daily data
        window_size = min(7, num_timestamps // 3)
        short_term_window = min(2, window_size // 2)
    else:  # Weekly or less frequent data
        window_size = min(4, num_timestamps // 3)
        short_term_window = min(2, window_size // 2)
    
    window_size = max(2, window_size)
    short_term_window = max(1, short_term_window)
    short_term_window = min(short_term_window, window_size - 1)
    
    return window_size, short_term_window

def backtest(
    price_diff_df: pd.DataFrame,
    promising_currencies_df: pd.DataFrame,
) -> Dict:
    """
    Perform backtesting using dynamically calculated window sizes.
    """
    results = {
        "long_term_returns": [],
        "short_term_returns": [],
        "long_term_metrics": [],
        "short_term_metrics": [],
        "newly_predicted": [],
        "dropped_predictions": [],
        "timestamps": [],
    }

    # Ensure we have valid DataFrames
    if price_diff_df is None or promising_currencies_df is None:
        print("Invalid input data. Please check your data loading.")
        return results

    # Get timestamps and sort them
    # Convert DatetimeArray to numpy array and sort
    timestamps = np.sort(promising_currencies_df["timestamp"].unique())
    
    # Calculate dynamic window sizes
    window_size, short_term_window = calculate_dynamic_windows(timestamps)
    
    print(f"Dynamic window sizes calculated:")
    print(f"Window size: {window_size}")
    print(f"Short-term window: {short_term_window}")
    print(f"Number of timestamps: {len(timestamps)}")
    print(f"Date range: {timestamps[0]} to {timestamps[-1]}")

    if len(timestamps) < window_size + 1:
        print("Not enough data points for the calculated window sizes.")
        return results

    for i in range(len(timestamps) - window_size):
        start_date = timestamps[i]
        end_date = timestamps[i + window_size]
        short_term_start = timestamps[i + window_size - short_term_window]

        print(f"Processing window {i+1}/{len(timestamps) - window_size}:")
        print(f"  Long-term: {start_date} to {end_date}")
        print(f"  Short-term: {short_term_start} to {end_date}")

        try:
            # Calculate technical indicators
            window_indicators = {}
            for slug in price_diff_df.index:
                price_series = price_diff_df.loc[slug, start_date:end_date]
                if len(price_series) >= 50:  # Minimum data points for technical indicators
                    window_indicators[slug] = calculate_technical_indicators(price_series)

            # Long-term evaluation
            long_term_returns = calculate_returns(price_diff_df, start_date, end_date)
            long_term_signals = generate_signals(promising_currencies_df, start_date)
            
            if not long_term_returns.empty and not long_term_signals.empty:
                long_term_portfolio_returns = simulate_trading(long_term_signals, long_term_returns)
                long_term_metrics = evaluate_strategy(long_term_signals, long_term_returns, window_indicators)
            else:
                print(f"  Warning: No data for long-term evaluation in this window")
                long_term_portfolio_returns = pd.Series([np.nan])
                long_term_metrics = {
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1_score": np.nan,
                    "technical_score": np.nan,
                }

            # Short-term evaluation
            short_term_returns = calculate_returns(price_diff_df, short_term_start, end_date)
            short_term_signals = generate_signals(promising_currencies_df, short_term_start)
            
            if not short_term_returns.empty and not short_term_signals.empty:
                short_term_portfolio_returns = simulate_trading(short_term_signals, short_term_returns)
                short_term_metrics = evaluate_strategy(short_term_signals, short_term_returns, window_indicators)
            else:
                print(f"  Warning: No data for short-term evaluation in this window")
                short_term_portfolio_returns = pd.Series([np.nan])
                short_term_metrics = {
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1_score": np.nan,
                    "technical_score": np.nan,
                }

            # Analyze currency changes
            prev_currencies = set(generate_signals(promising_currencies_df, start_date).index)
            current_currencies = set(generate_signals(promising_currencies_df, end_date).index)
            newly_predicted, dropped_predictions = analyze_currency_changes(
                prev_currencies, current_currencies
            )

            # Store results
            results["timestamps"].append(start_date)
            results["long_term_returns"].append(long_term_portfolio_returns.mean())
            results["short_term_returns"].append(short_term_portfolio_returns.mean())
            results["long_term_metrics"].append(long_term_metrics)
            results["short_term_metrics"].append(short_term_metrics)
            results["newly_predicted"].append(newly_predicted)
            results["dropped_predictions"].append(dropped_predictions)

        except Exception as e:
            print(f"  Error processing window: {e}")

    return results

def plot_results(results: Dict):
    """
    Plot the backtesting results, now including technical indicators.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Original plots
    ax1.plot(results["long_term_returns"], label="Long-term returns")
    ax1.plot(results["short_term_returns"], label="Short-term returns")
    ax1.set_title("Portfolio Returns")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Returns")
    ax1.legend()

    # Metrics plot
    long_term_accuracy = [m["accuracy"] for m in results["long_term_metrics"]]
    short_term_accuracy = [m["accuracy"] for m in results["short_term_metrics"]]
    ax2.plot(long_term_accuracy, label="Long-term accuracy")
    ax2.plot(short_term_accuracy, label="Short-term accuracy")
    ax2.set_title("Model Accuracy")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    # Technical indicators plot
    technical_scores = [m["technical_score"] for m in results["long_term_metrics"]]
    ax3.plot(technical_scores, label="Technical Score")
    ax3.set_title("Technical Indicators Score")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Score")
    ax3.legend()

    plt.tight_layout()
    plt.show()

def main():
    price_diff_file = rf"C:/Users/{os.getenv('USER')}/Desktop/price_difference_dump/Price_Diff_PromisingCurrencies.csv"
    promising_currencies_file = rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/PromisingCurrencies.csv"

    try:
        price_diff_df, promising_currencies_df = load_data(
            price_diff_file, promising_currencies_file
        )

        print("\nSample technical indicators:")
        sample_slug = price_diff_df.index[0]
        sample_indicators = calculate_technical_indicators(price_diff_df.loc[sample_slug])
        print(f"Indicators for {sample_slug}: {sample_indicators}")

        sample_score = calculate_technical_score({sample_slug: sample_indicators})
        print(f"Technical score for {sample_slug}: {sample_score}")

        print("\nDataset Information:")
        print(f"Price diff DataFrame shape: {price_diff_df.shape}")
        print(f"Promising currencies DataFrame shape: {promising_currencies_df.shape}")
        
        # Debug timestamp information
        print("\nTimestamp Debug Info:")
        print(f"Price diff columns dtype: {type(price_diff_df.columns[0])}")
        print(f"First few price diff timestamps: {price_diff_df.columns[:3]}")
        print(f"Promising currencies timestamp dtype: {promising_currencies_df['timestamp'].dtype}")
        print(f"First few promising currencies timestamps: {promising_currencies_df['timestamp'].iloc[:3]}")
        
        results = backtest(price_diff_df, promising_currencies_df)

        if results["timestamps"]:  # Only plot if we have results
            plot_results(results)
        else:
            print("No results generated. Check the data and parameters.")

    except Exception as e:
        print(f"An error occurred in main: {e}")
        import traceback
        traceback.print_exc()

    
if __name__ == "__main__":
    main()
