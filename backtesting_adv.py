import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import os
from dotenv import load_dotenv

load_dotenv()


def load_data(
    price_diff_file: str, promising_currencies_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the price difference and promising currencies data.
    """
    price_diff_df = pd.read_csv(price_diff_file, index_col="slug")
    price_diff_df.columns = pd.to_datetime(
        price_diff_df.columns, format="%Y-%m-%d %H:%M:%S.%f"
    )

    promising_currencies_df = pd.read_csv(promising_currencies_file)
    promising_currencies_df["timestamp"] = pd.to_datetime(
        promising_currencies_df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f"
    )

    return price_diff_df.sort_index(axis=1), promising_currencies_df


def calculate_returns(
    df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.Series:
    """
    Calculate percentage returns for a specific time period.
    """
    if start_date not in df.columns or end_date not in df.columns:
        print(f"Warning: {start_date} or {end_date} not found in price_diff_df")
        return pd.Series()

    start_prices = df[start_date]
    end_prices = df[end_date]
    returns = (end_prices - start_prices) / start_prices
    return returns.replace([np.inf, -np.inf], np.nan)


def generate_signals(
    promising_currencies: pd.DataFrame, timestamp: pd.Timestamp
) -> pd.Series:
    """
    Generate buy/sell signals based on the promising currencies at a specific timestamp.
    """
    currencies = promising_currencies[promising_currencies["timestamp"] == timestamp][
        "slug"
    ]
    return pd.Series(1, index=currencies)


def simulate_trading(signals: pd.Series, returns: pd.Series) -> pd.Series:
    """
    Simulate trading based on signals and calculate portfolio returns.
    """
    return signals * returns


def evaluate_strategy(signals: pd.Series, returns: pd.Series) -> Dict[str, float]:
    """
    Evaluate the trading strategy using various performance metrics.
    """
    if signals.empty or returns.empty:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
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
        }

    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(
            true_labels, predicted_labels, average="binary", zero_division=0
        ),
        "recall": recall_score(
            true_labels, predicted_labels, average="binary", zero_division=0
        ),
        "f1_score": f1_score(
            true_labels, predicted_labels, average="binary", zero_division=0
        ),
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


def backtest(
    price_diff_df: pd.DataFrame,
    promising_currencies_df: pd.DataFrame,
    window_size: int = 2,
    short_term_window: int = 1,
) -> Dict:
    """
    Perform backtesting using a sliding window approach.
    """
    results = {
        "long_term_returns": [],
        "short_term_returns": [],
        "long_term_metrics": [],
        "short_term_metrics": [],
        "newly_predicted": [],
        "dropped_predictions": [],
    }

    timestamps = promising_currencies_df["timestamp"].unique()
    timestamps = np.sort(timestamps)

    print(f"Number of timestamps: {len(timestamps)}")
    print(f"Window size: {window_size}, Short-term window: {short_term_window}")

    for i in range(len(timestamps) - window_size):
        start_date = timestamps[i]
        end_date = timestamps[i + window_size]
        short_term_start = timestamps[max(0, i + window_size - short_term_window)]

        print(f"Processing window: {start_date} to {end_date}")

        # Long-term evaluation
        long_term_returns = calculate_returns(price_diff_df, start_date, end_date)
        long_term_signals = generate_signals(promising_currencies_df, start_date)
        long_term_portfolio_returns = simulate_trading(
            long_term_signals, long_term_returns
        )
        long_term_metrics = evaluate_strategy(long_term_signals, long_term_returns)

        # Short-term evaluation
        short_term_returns = calculate_returns(
            price_diff_df, short_term_start, end_date
        )
        short_term_signals = generate_signals(promising_currencies_df, short_term_start)
        short_term_portfolio_returns = simulate_trading(
            short_term_signals, short_term_returns
        )
        short_term_metrics = evaluate_strategy(short_term_signals, short_term_returns)

        # Analyze currency changes
        prev_currencies = set(
            generate_signals(promising_currencies_df, start_date).index
        )
        current_currencies = set(
            generate_signals(promising_currencies_df, end_date).index
        )
        newly_predicted, dropped_predictions = analyze_currency_changes(
            prev_currencies, current_currencies
        )

        # Store results
        results["long_term_returns"].append(long_term_portfolio_returns.mean())
        results["short_term_returns"].append(short_term_portfolio_returns.mean())
        results["long_term_metrics"].append(long_term_metrics)
        results["short_term_metrics"].append(short_term_metrics)
        results["newly_predicted"].append(newly_predicted)
        results["dropped_predictions"].append(dropped_predictions)

    return results


def plot_results(results: Dict):
    """
    Plot the backtesting results.
    """
    if not results["long_term_returns"] and not results["short_term_returns"]:
        print("No data to plot. Skipping visualization.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot returns
    ax1.plot(results["long_term_returns"], label="Long-term returns")
    ax1.plot(results["short_term_returns"], label="Short-term returns")
    ax1.set_title("Portfolio Returns")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Returns")
    ax1.legend()

    # Plot metrics
    long_term_accuracy = [m["accuracy"] for m in results["long_term_metrics"]]
    short_term_accuracy = [m["accuracy"] for m in results["short_term_metrics"]]
    ax2.plot(long_term_accuracy, label="Long-term accuracy")
    ax2.plot(short_term_accuracy, label="Short-term accuracy")
    ax2.set_title("Model Accuracy")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    price_diff_file = rf"C:/Users/{os.getenv('USER')}/Desktop/price_difference_dump/Price_Diff_PromisingCurrencies.csv"
    promising_currencies_file = (
        rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/PromisingCurrencies.csv"
    )

    price_diff_df, promising_currencies_df = load_data(
        price_diff_file, promising_currencies_file
    )

    print("Price diff DataFrame shape:", price_diff_df.shape)
    print("Promising currencies DataFrame shape:", promising_currencies_df.shape)
    print("Total timestamps:", len(promising_currencies_df["timestamp"].unique()))
    print(
        "First timestamp:",
        promising_currencies_df["timestamp"].min(),
        "Last timestamp:",
        promising_currencies_df["timestamp"].max(),
    )

    # Adjust window sizes based on the number of available timestamps
    total_timestamps = len(promising_currencies_df["timestamp"].unique())
    window_size = min(total_timestamps - 1, 3)  # Need atleast 2 windows for comparison
    short_term_window = min(window_size - 1, 2)

    results = backtest(
        price_diff_df,
        promising_currencies_df,
        window_size=window_size,
        short_term_window=short_term_window,
    )

    plot_results(results)

    print("Average long-term return:", np.nanmean(results["long_term_returns"]))
    print("Average short-term return:", np.nanmean(results["short_term_returns"]))
    print(
        "Average long-term accuracy:",
        np.nanmean([m["accuracy"] for m in results["long_term_metrics"]]),
    )
    print(
        "Average short-term accuracy:",
        np.nanmean([m["accuracy"] for m in results["short_term_metrics"]]),
    )

    # Analyze currency prediction changes
    all_newly_predicted = set().union(*results["newly_predicted"])
    all_dropped_predictions = set().union(*results["dropped_predictions"])
    print("Total newly predicted currencies:", len(all_newly_predicted))
    print("Total dropped predictions:", len(all_dropped_predictions))


if __name__ == "__main__":
    main()
