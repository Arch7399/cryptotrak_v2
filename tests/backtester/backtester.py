from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from models.position import EnhancedPosition
from risk_management.risk_manager import EnhancedRiskManagement
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
from utils.metrics import MetricsCalculator
from utils.volatility import VolatilityCalculator
from tests.visualization.backtest_plots import BacktestVisualizer
from trading.position_manager import PositionManager


class PredictionBacktester:
    def __init__(
        self,
        crypto_df: pd.DataFrame,
        promising_currencies_df: pd.DataFrame,
        initial_equity: float = 1000.0,
        verbose: bool = False,
    ):
        self.crypto_df = crypto_df
        self.promising_currencies_df = promising_currencies_df
        self.initial_equity = initial_equity
        self.verbose = verbose
        self.risk_manager = EnhancedRiskManagement(
            initial_equity=initial_equity,
            max_risk_per_trade=0.05,
            max_account_risk=0.15,
            max_volume_percent=0.03,
            fee_rate=0.001,
            max_positions=100,
        )
        self.metrics = MetricsCalculator
        self.visualizer = BacktestVisualizer
        self.volatility = VolatilityCalculator
        self.position = PositionManager
        self.fee_rate = 0.001
        self.results = self.initialize_results()
        self.timestamps = sorted(crypto_df["timestamp"].unique())
        self.timestamp_map = {ts: i for i, ts in enumerate(self.timestamps)}
        self.equity_used_per_timestamp = {}

    def calculate_trading_fees(self, position_size: float, price: float) -> float:
        return position_size * price * self.fee_rate

    def get_next_timestamp(
        self, current_timestamp: pd.Timestamp
    ) -> Optional[pd.Timestamp]:
        current_index = self.timestamp_map.get(current_timestamp)
        if current_index is not None and current_index < len(self.timestamps) - 1:
            return self.timestamps[current_index + 1]
        return None

    def get_predictions_for_timestamp(self, timestamp: pd.Timestamp) -> List[str]:
        exact_predictions = self.get_exact_predictions(timestamp)
        return [currency for currency, _ in exact_predictions]

    def initialize_results(self) -> Dict:
        return {
            "timestamps": [],
            "returns": [],
            "equity_curve": [self.initial_equity],
            "positions": {},
            "trades_history": [],
            "metrics": {},
            "signals_generated": [],
            "position_creation_log": [],
        }

    def get_recent_predictions(
        self, current_timestamp: pd.Timestamp, hours: int = 24
    ) -> List[str]:
        time_threshold = current_timestamp - pd.Timedelta(hours=hours)
        recent_predictions = self.promising_currencies_df[
            (self.promising_currencies_df["timestamp"] > time_threshold)
            & (self.promising_currencies_df["timestamp"] <= current_timestamp)
        ]
        return recent_predictions["slug"].unique().tolist()

    def get_exact_predictions(
        self, current_timestamp: pd.Timestamp
    ) -> List[Tuple[str, pd.Timestamp]]:
        exact_predictions = self.promising_currencies_df[
            self.promising_currencies_df["timestamp"] == current_timestamp
        ]
        return [
            (row["slug"], row["timestamp"]) for _, row in exact_predictions.iterrows()
        ]

    def backtest(self):
        total_timestamps = len(self.timestamps)

        for i, timestamp in enumerate(self.timestamps):
            if self.verbose and i % 100 == 0:
                print(f"Processing timestamp {i+1}/{total_timestamps}: {timestamp}")

            # Reset equity used for new timestamp
            timestamp_str = str(timestamp)
            self.equity_used_per_timestamp[timestamp_str] = 0

            current_data = self.crypto_df[self.crypto_df["timestamp"] == timestamp]
            next_timestamp = self.get_next_timestamp(timestamp)

            if self.verbose and self.results["positions"]:
                print(f"\nOpen positions at {timestamp}:")
                for currency, pos in self.results["positions"].items():
                    print(
                        f"{currency}: Entry {pos.entry_price}, SL {pos.stop_loss}, TP {pos.take_profit}, PV {pos.position_value}"
                    )

            for currency in list(self.results["positions"].keys()):
                if currency not in current_data["slug"].values:
                    missing_data = self.crypto_df[
                        (self.crypto_df["slug"] == currency)
                        & (self.crypto_df["timestamp"] == timestamp)
                    ]
                    if not missing_data.empty:
                        current_data = pd.concat([current_data, missing_data])

            closed_positions = self.position.process_existing_positions(
                self, current_data, timestamp
            )
            if closed_positions and self.verbose:
                print(f"Closed positions: {closed_positions}")

            if next_timestamp:
                predicted_currencies = self.get_predictions_for_timestamp(timestamp)
                for currency in predicted_currencies:
                    if currency not in self.results["positions"]:
                        currency_data = current_data[current_data["slug"] == currency]
                        if not currency_data.empty:
                            new_position = self.position.create_position(
                                self, currency_data.iloc[0], timestamp, next_timestamp
                            )
                            if new_position:
                                self.results["positions"][currency] = new_position
                                if self.verbose:
                                    print(f"Created new position for {currency}")

            current_equity = self.risk_manager.current_equity
            self.results["timestamps"].append(timestamp)
            self.results["equity_curve"].append(current_equity)

        if self.verbose:
            print("\nBacktest completed:")
            print(f"Total trades: {len(self.results['trades_history'])}")
            print(f"Final equity: {self.results['equity_curve'][-1]}")

        self.metrics.calculate_metrics(self)
        return self.results

    def plot_results(self):
        self.visualizer.plot_results(self)
