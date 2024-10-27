import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
import pandas as pd
import os
from tests.backtester.backtester import PredictionBacktester
from tests.visualization.backtest_plots import BacktestVisualizer


def main():
    crypto_df = pd.read_csv(rf"C:/Users/{os.getenv('USER')}/Desktop/CryptoAPI.csv")
    promising_currencies_df = pd.read_csv(
        rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/PromisingCurrencies.csv"
    )

    crypto_df["timestamp"] = pd.to_datetime(crypto_df["timestamp"])
    promising_currencies_df["timestamp"] = pd.to_datetime(
        promising_currencies_df["timestamp"]
    )

    initial_equity = Config.initial_equity  # change initial money amount in config.py
    backtester = PredictionBacktester(
        crypto_df=crypto_df,
        promising_currencies_df=promising_currencies_df,
        initial_equity=initial_equity,
        verbose=True,
    )
    results = backtester.backtest()

    visualizer = BacktestVisualizer(results, initial_equity)
    summary = visualizer.plot_metrics()
    backtester.plot_results()

    print("\nBacktest Summary Statistics:")
    for metric, value in summary.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")

    print("\nPosition Summary:")
    print(f"Total signals evaluated: {len(results['position_creation_log'])}")
    positions_created = sum(
        1
        for log in results["position_creation_log"]
        if log.get("position_created", False)
    )
    print(f"Positions created: {positions_created}")
    print(f"Positions closed: {len(results['trades_history'])}")
    print(f"Positions still open: {positions_created - len(results['trades_history'])}")

    rejection_reasons = {}
    for log in results["position_creation_log"]:
        if "reason_rejected" in log:
            reason = log["reason_rejected"]
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    if rejection_reasons:
        print("\nRejection reasons:")
        for reason, count in rejection_reasons.items():
            print(f"{reason}: {count}")


if __name__ == "__main__":
    main()
