import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()


def load_data(file_path):
    """Load the Price_Diff_PromisingCurrencies.csv file"""
    df = pd.read_csv(file_path, index_col="slug")
    df.columns = pd.to_datetime(df.columns, format="%Y-%m-%d %H:%M:%S.%f")
    return df.sort_index(axis=1)  # Sort columns by date


def calculate_returns(df):
    """Calculate percentage returns from price differences"""
    # Replace empty strings with NaN
    df = df.replace("", np.nan).astype(float)

    # Calculate cumulative prices
    cumulative_prices = df.cumsum()

    # Calculate returns, handling division by zero
    returns = cumulative_prices.pct_change(axis=1)
    returns = returns.replace([np.inf, -np.inf], np.nan)

    return returns


def generate_signals(returns, threshold=0.01):
    """Generate buy/sell signals based on return threshold"""
    return (returns > threshold).astype(int) - (returns < -threshold).astype(int)


def simulate_trading(signals, returns):
    """Simulate trading based on signals"""
    portfolio_returns = (signals.shift(1, axis=1) * returns).sum(axis=0)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    return cumulative_returns


def evaluate_strategy(signals, returns):
    """Evaluate the trading strategy"""
    true_labels = (returns > 0).astype(int)
    predicted_labels = (signals > 0).astype(int)

    # Flatten and remove NaN values
    true_flat = true_labels.values.flatten()
    pred_flat = predicted_labels.values.flatten()
    mask = ~np.isnan(true_flat) & ~np.isnan(pred_flat)
    true_flat = true_flat[mask]
    pred_flat = pred_flat[mask]

    if len(true_flat) == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}

    accuracy = accuracy_score(true_flat, pred_flat)
    precision = precision_score(
        true_flat, pred_flat, average="weighted", zero_division=0
    )
    recall = recall_score(true_flat, pred_flat, average="weighted", zero_division=0)
    f1 = f1_score(true_flat, pred_flat, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def plot_cumulative_returns(cumulative_returns):
    """Plot cumulative returns"""
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns.index, cumulative_returns.values)
    plt.title("Cumulative Returns of Trading Strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def backtest(file_path, threshold=0.01):
    """Main backtesting function"""
    df = load_data(file_path)
    returns = calculate_returns(df)
    signals = generate_signals(returns, threshold)
    cumulative_returns = simulate_trading(signals, returns)
    performance_metrics = evaluate_strategy(signals, returns)

    print("Performance Metrics:")
    for metric, value in performance_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    plot_cumulative_returns(cumulative_returns)

    return cumulative_returns, performance_metrics, returns, signals


# Usage
file_path = rf"C:/Users/{os.getenv('USER')}/Desktop/price_difference_dump/Price_Diff_PromisingCurrencies.csv"
cumulative_returns, performance_metrics, returns, signals = backtest(file_path)

# Additional analysis
print("\nTop 5 performing currencies:")
total_returns = returns.sum(axis=1).sort_values(ascending=False)
print(total_returns.head())

print("\nBottom 5 performing currencies:")
print(total_returns.tail())

print("\nNumber of buy signals per currency:")
buy_signals = (signals > 0).sum(axis=1).sort_values(ascending=False)
print(buy_signals.head())

print("\nNumber of sell signals per currency:")
sell_signals = (signals < 0).sum(axis=1).sort_values(ascending=False)
print(sell_signals.head())

# print or export cumulative_returns for further testing/analysis.
print(cumulative_returns)
