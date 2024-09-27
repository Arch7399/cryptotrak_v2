import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime, timedelta
import math

@dataclass
class EnhancedPosition:
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    position_value: float
    entry_fee: float
    currency: str
    entry_timestamp: pd.Timestamp
    prediction_timestamp: pd.Timestamp
    highest_price: float = field(default=0.0)  
    current_tp_level: int = field(default=0)  
    original_position_size: float = field(default=0.0)  

    def __post_init__(self):
        self.highest_price = self.entry_price
        self.original_position_size = self.position_size


class EnhancedRiskManagement:
    def __init__(
        self,
        initial_equity: float,
        max_risk_per_trade: float = 0.05,
        max_account_risk: float = 0.15,
        max_volume_percent: float = 0.03,
        fee_rate: float = 0.001,
        max_positions: int = 100,
        trailing_stop_percentage: float = 0.02,  # modify trailing stops here
        take_profit_levels: List[float] = [1.5, 2.0, 2.5],  # modify take profit levels here
        position_size_reduction: float = 0.5,  # Reduce position by 50% at each TP
    ):
        self.current_equity = initial_equity
        self.initial_equity = initial_equity
        self.max_risk_per_trade = max_risk_per_trade
        self.max_account_risk = max_account_risk
        self.max_volume_percent = max_volume_percent
        self.fee_rate = fee_rate
        self.max_position_value = initial_equity * max_account_risk
        self.max_positions = max_positions
        self.trailing_stop_percentage = trailing_stop_percentage
        self.take_profit_levels = take_profit_levels
        self.position_size_reduction = position_size_reduction

        self.min_market_cap = 1000000
        self.min_daily_volume = 100000
        self.max_volatility = 0.1 # adjust volatility max here
        self.max_price_impact = 0.05

    def update_equity(self, pnl: float):
        self.current_equity += pnl

    def set_trailing_stop_loss(
        self, 
        entry_price: float, 
        current_price: float, 
        current_stop_loss: float,
        highest_price: float
    ) -> float:
        """
        Updates the trailing stop loss based on price movement.
        Returns the new stop loss price.
        """
        trailing_distance = entry_price * self.trailing_stop_percentage
        
        # If price has moved up, move the stop loss up
        if current_price > highest_price:
            highest_price = current_price  
        new_stop_loss = highest_price - trailing_distance
        return max(new_stop_loss, current_stop_loss)

    def get_next_take_profit(
        self, 
        entry_price: float, 
        current_tp_level: int
    ) -> Tuple[float, int]:
        """
        Returns the next take profit price and level index.
        If all levels are exhausted, returns (None, -1).
        """
        if current_tp_level >= len(self.take_profit_levels):
            return None, -1
            
        tp_multiplier = self.take_profit_levels[current_tp_level]
        risk = entry_price * self.trailing_stop_percentage
        next_tp = entry_price + (risk * tp_multiplier)
        
        return next_tp, current_tp_level + 1

    def calculate_partial_exit_size(
        self, 
        current_position_size: float, 
        tp_level: int
    ) -> float:
        """
        Calculates how much of the position to exit at current take profit level.
        """
        exit_size = current_position_size * (self.position_size_reduction / (tp_level + 1))
        return math.floor(exit_size)

    def calculate_position_size(
        self,
        price: float,
        stop_loss: float,
        volume_24h: float,
        market_cap: float = None,
    ) -> float:
        if (
            price <= 0
            or stop_loss <= 0
            or volume_24h < self.min_daily_volume
            or (market_cap and market_cap < self.min_market_cap)
        ):
            return 0

        
        risk_amount = min(
            self.current_equity * self.max_risk_per_trade, self.max_position_value * 0.2
        )
        price_risk = abs(price - stop_loss) / price

        if price_risk <= 0 or price_risk > self.max_volatility:
            return 0

        risk_based_size = risk_amount / (price_risk * price)
        volume_based_size = volume_24h * self.max_volume_percent / price
        max_position_size = self.max_position_value / price

        position_size = min(risk_based_size, volume_based_size, max_position_size)

        return position_size if position_size * price <= self.current_equity else 0

    def calculate_actual_exit_price(
        self, position_size: float, exit_price: float, volume_24h: float
    ) -> float:
        if volume_24h <= 0:
            return exit_price * 0.9

        impact = min((position_size * exit_price) / volume_24h * 0.1, 0.1)
        actual_exit_price = exit_price * (1 - impact)
        return actual_exit_price

    def calculate_pnl(
        self, position: EnhancedPosition, exit_price: float, volume_24h: float
    ) -> float:
        actual_exit_price = self.calculate_actual_exit_price(
            position.position_size, exit_price, volume_24h
        )

        exit_fee = position.position_size * actual_exit_price * self.fee_rate

        gross_pnl = (actual_exit_price - position.entry_price) * position.position_size
        net_pnl = gross_pnl - position.entry_fee - exit_fee

        max_loss = -(position.position_size * position.entry_price)
        return max(net_pnl, max_loss)

    def can_open_position(self, required_margin: float) -> bool:
        return required_margin <= self.current_equity * self.max_account_risk

    def estimate_price_impact(
        self,
        order_size: float,
        price: float,
        volume_24h: float,
        market_cap: Optional[float] = None,
    ) -> float:
        if volume_24h <= 0:
            return float("inf")

        volume_impact = (order_size * price) / volume_24h

        if market_cap and market_cap > 0:
            market_cap_impact = (order_size * price) / market_cap
            return max(volume_impact * 0.1, market_cap_impact * 0.05)
        else:
            return volume_impact * 0.15

    def set_stop_loss(
        self, entry_price: float, volatility: float, multiplier: float = 1.5
    ) -> float:
        return entry_price * (
            1 - min(volatility * multiplier, 0.15)
        )  # set stop loss here

    def set_take_profit(
        self, entry_price: float, stop_loss: float, risk_reward_ratio: float = 1.5
    ) -> float:
        risk = entry_price - stop_loss
        return entry_price + (risk * risk_reward_ratio)

    def calculate_trading_fees(self, position_size: float, price: float) -> float:
        return position_size * price * self.fee_rate


class BacktestVisualizer:
    def __init__(self, results, initial_equity):
        self.results = results
        self.initial_equity = initial_equity

    def calculate_metrics(self):
        if not self.results["trades_history"]:
            return pd.DataFrame(), pd.Series()

        trades_df = pd.DataFrame(self.results["trades_history"])
        trades_df["exit_timestamp"] = pd.to_datetime(trades_df["exit_timestamp"])
        trades_df = trades_df.sort_values("exit_timestamp")

        # Calculate actual equity changes
        equity_changes = [self.initial_equity]
        current_equity = self.initial_equity

        for _, trade in trades_df.iterrows():
            current_equity += trade["return"]
            equity_changes.append(current_equity)

        trades_df["equity"] = equity_changes[1:]

        
        trades_df["is_win"] = trades_df["return"] > 0
        trades_df["rolling_wins"] = (
            trades_df["is_win"].rolling(window=10, min_periods=1).sum()
        )
        trades_df["rolling_total"] = trades_df.index.map(lambda x: min(x + 1, 10))
        trades_df["rolling_win_rate"] = (
            trades_df["rolling_wins"] / trades_df["rolling_total"]
        )
        trades_df["rolling_avg_return"] = (
            trades_df["return"].rolling(window=10, min_periods=1).mean()
        )

        final_equity = equity_changes[-1]
        total_return_pct = ((final_equity / self.initial_equity) - 1) * 100

        summary = pd.Series(
            {
                "Total Trades": len(trades_df),
                "Win Rate": len(trades_df[trades_df["return"] > 0]) / len(trades_df),
                "Average Return ($)": trades_df["return"].mean(),
                "Return Std Dev ($)": trades_df["return"].std(),
                "Max Return ($)": trades_df["return"].max(),
                "Min Return ($)": trades_df["return"].min(),
                "Initial Equity ($)": self.initial_equity,
                "Final Equity ($)": final_equity,
                "Total Return (%)": total_return_pct,
                "Profit Factor": (
                    abs(
                        trades_df[trades_df["return"] > 0]["return"].sum()
                        / trades_df[trades_df["return"] < 0]["return"].sum()
                    )
                    if len(trades_df[trades_df["return"] < 0]) > 0
                    else float("inf")
                ),
            }
        )

        return trades_df, summary

    def plot_metrics(self):
        trades_df, summary = self.calculate_metrics()
        if trades_df.empty:
            print("No trades to visualize")
            return summary

        # Set style and colors
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = {
            'equity': '#1f77b4',      # Blue
            'win_rate': '#2ca02c',    # Green
            'hist': '#1f77b4',        # Blue
            'returns': '#ff7f0e',     # Orange
            'zero_line': '#d62728'    # Red
        }

        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.patch.set_facecolor('white')

        # Plot 1: Actual Equity Changes
        full_equity_timeline = [self.initial_equity] + trades_df["equity"].tolist()
        timestamps = [
            trades_df["exit_timestamp"].iloc[0] - pd.Timedelta(hours=5)
        ] + trades_df["exit_timestamp"].tolist()
        axes[0, 0].plot(timestamps, full_equity_timeline, color=colors['equity'], linewidth=2)
        axes[0, 0].set_title("Equity Changes Over Time", pad=20, fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel("Equity ($)", fontsize=10)
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        axes[0, 0].tick_params(axis='both', labelsize=9)

        # Plot 2: Rolling Win Rate
        axes[0, 1].plot(trades_df["exit_timestamp"], trades_df["rolling_win_rate"], 
                        color=colors['win_rate'], linewidth=2)
        axes[0, 1].set_title("Rolling Win Rate (10 trades)", pad=20, fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel("Win Rate", fontsize=10)
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        axes[0, 1].tick_params(axis='both', labelsize=9)
        axes[0, 1].set_ylim([0, max(trades_df["rolling_win_rate"]) * 1.1])

        # Plot 3: Return Distribution
        trades_df["return"].hist(ax=axes[1, 0], bins=20, color=colors['hist'], 
                            alpha=0.7, edgecolor='white')
        axes[1, 0].set_title("Return Distribution ($)", pad=20, fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel("Return ($)", fontsize=10)
        axes[1, 0].set_ylabel("Frequency", fontsize=10)
        axes[1, 0].tick_params(axis='both', labelsize=9)
        axes[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Plot 4: Trade Returns Over Time
        axes[1, 1].plot(trades_df["exit_timestamp"], trades_df["return"], 
                        color=colors['returns'], linewidth=2)
        axes[1, 1].axhline(y=0, color=colors['zero_line'], linestyle='-', alpha=0.3)
        axes[1, 1].set_title("Trade Returns Over Time", pad=20, fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel("Return ($)", fontsize=10)
        axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        axes[1, 1].tick_params(axis='both', labelsize=9)

        for ax in axes.flat:
            if ax is not axes[1,0]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.margins(x=0.02)

        plt.tight_layout(pad=5.0)
        plt.show()
        return summary

    def generate_summary_statistics(self):
        trades_df = pd.DataFrame(self.results["trades_history"])
        if trades_df.empty:
            return "No trades to analyze"

        summary = {
            "Total Trades": len(trades_df),
            "Win Rate": len(trades_df[trades_df["return"] > 0]) / len(trades_df),
            "Average Return": trades_df["return"].mean(),
            "Return Std Dev": trades_df["return"].std(),
            "Max Return": trades_df["return"].max(),
            "Min Return": trades_df["return"].min(),
            "Profit Factor": (
                abs(
                    trades_df[trades_df["return"] > 0]["return"].sum()
                    / trades_df[trades_df["return"] < 0]["return"].sum()
                )
                if len(trades_df[trades_df["return"] < 0]) > 0
                else float("inf")
            ),
        }

        return pd.Series(summary)


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

    def process_positions(self, current_data: pd.DataFrame, timestamp: pd.Timestamp) -> Dict[str, float]:
        closed_positions = {}
        current_data_dict = current_data.set_index("slug").to_dict(orient="index")

        for currency, position in list(self.results["positions"].items()):
            if currency not in current_data_dict:
                if self.verbose:
                    print(f"No data found for {currency} at {timestamp}")
                continue

            current_price = current_data_dict[currency]["quote.USD.price"]
            
            # Update highest price and trailing stop loss
            if current_price > position.highest_price:
                position.highest_price = current_price
                position.stop_loss = self.risk_manager.set_trailing_stop_loss(
                    position.entry_price,
                    current_price,
                    position.stop_loss,
                    position.highest_price
                )

            # Check stop loss
            if current_price <= position.stop_loss:
                exit_fee = self.calculate_trading_fees(position.position_size, current_price)
                pnl = self.risk_manager.calculate_pnl(position, current_price, current_data_dict[currency]["quote.USD.volume_24h"])
                self.risk_manager.update_equity(pnl)
                
                closed_positions[currency] = pnl
                del self.results["positions"][currency]
                
                # Record trade
                self.record_trade(currency, position, current_price, timestamp, "stop_loss", exit_fee, pnl)
                
            # Check take profit
            elif current_price >= position.take_profit:
                # Calculate partial position size to exit
                exit_size = self.risk_manager.calculate_partial_exit_size(
                    position.position_size,
                    position.current_tp_level
                )
                
                exit_fee = self.calculate_trading_fees(exit_size, current_price)
                
                # Calculate PnL for partial exit
                partial_pnl = (current_price - position.entry_price) * exit_size - exit_fee
                self.risk_manager.update_equity(partial_pnl)
                
                # Update position size
                position.position_size -= exit_size
                
                # Get next take profit level
                next_tp, next_tp_level = self.risk_manager.get_next_take_profit(
                    position.entry_price,
                    position.current_tp_level
                )
                
                if next_tp is None or position.position_size <= 0:
                    if position.position_size > 0:
                        final_exit_fee = self.calculate_trading_fees(position.position_size, current_price)
                        final_pnl = (current_price - position.entry_price) * position.position_size - final_exit_fee
                        self.risk_manager.update_equity(final_pnl)

                        partial_pnl += final_pnl
                    
                    closed_positions[currency] = partial_pnl
                    del self.results["positions"][currency]
                    
                    # Record final trade
                    self.record_trade(currency, position, current_price, timestamp, "take_profit_final", exit_fee + final_exit_fee, partial_pnl)
                else:
                    # Update position for next take profit level
                    position.take_profit = next_tp
                    position.current_tp_level = next_tp_level
                    position.position_value = position.position_size * current_price
                    
                    # Record partial trade
                    self.record_trade(currency, position, current_price, timestamp, "take_profit_partial", exit_fee, partial_pnl)

        return closed_positions

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

    def calculate_volatility(self, row: pd.Series) -> float:
        try:
            changes = [
                row["quote.USD.percent_change_1h"],
                row["quote.USD.percent_change_24h"],
            ]
            valid_changes = [x / 100 for x in changes if pd.notna(x)]
            return max(
                np.std(valid_changes) if valid_changes else 0.02, 0.02
            )  
        except Exception as e:
            print(f"Error calculating volatility: {str(e)}")
            return 0.02  # Default volatility

    def get_recent_predictions(
        self, current_timestamp: pd.Timestamp, hours: int = 24
    ) -> List[str]:
        time_threshold = current_timestamp - pd.Timedelta(hours=hours)
        recent_predictions = self.promising_currencies_df[
            (self.promising_currencies_df["timestamp"] > time_threshold)
            & (self.promising_currencies_df["timestamp"] <= current_timestamp)
        ]
        return recent_predictions["slug"].unique().tolist()

    def create_position(
        self, row: pd.Series, timestamp: pd.Timestamp, next_timestamp: pd.Timestamp
    ) -> Optional[EnhancedPosition]:
        try:
            price = row["quote.USD.price"]
            volume_24h = row["quote.USD.volume_24h"]
            market_cap = (
                row["quote.USD.market_cap"] if "quote.USD.market_cap" in row else None
            )
            volatility = self.calculate_volatility(row)

            sentiment_score = row.get("sentiment_score", 0)
            trend_strength = row.get("trend_strength", 0)
            volume_change = row.get("volume_change", 0)

            log_entry = {
                "timestamp": timestamp,
                "currency": row["slug"],
                "price": price,
                "volume_24h": volume_24h,
                "market_cap": market_cap,
                "volatility": volatility,
                "sentiment_score": sentiment_score,
                "trend_strength": trend_strength,
                "volume_change": volume_change,
                "current_equity": self.risk_manager.current_equity,
            }

            # Check maximum number of positions
            if len(self.results["positions"]) >= self.risk_manager.max_positions:
                log_entry["reason_rejected"] = "Maximum positions reached"
                self.results["position_creation_log"].append(log_entry)
                return None

            if volume_24h < self.risk_manager.min_daily_volume:
                log_entry["reason_rejected"] = "Insufficient volume"
                self.results["position_creation_log"].append(log_entry)
                return None

            if market_cap and market_cap < self.risk_manager.min_market_cap:
                log_entry["reason_rejected"] = "Market cap too low"
                self.results["position_creation_log"].append(log_entry)
                return None

            if volatility > self.risk_manager.max_volatility:
                log_entry["reason_rejected"] = "Volatility too high"
                self.results["position_creation_log"].append(log_entry)
                return None

            stop_loss = price * (1 - min(volatility * 2.0, 0.25))
            position_size = self.risk_manager.calculate_position_size(
                price=price,
                stop_loss=stop_loss,
                volume_24h=volume_24h,
                market_cap=market_cap,
            )

            # Round down to whole number and reject if less than 1
            position_size = math.floor(position_size)
            if position_size < 1:
                log_entry["reason_rejected"] = "Position size less than 1"
                self.results["position_creation_log"].append(log_entry)
                return None

            # Limit position value to 0.5% of initial equity
            max_position_value = self.initial_equity * 0.005  # 0.5%
            position_value = position_size * price
            
            if position_value > max_position_value:
                # Reduce position size to meet the 0.5% limit
                position_size = math.floor(max_position_value / price)
                position_value = position_size * price
                
                # Check if the reduced position size is still valid
                if position_size < 1:
                    log_entry["reason_rejected"] = "Position size too small after value limit"
                    self.results["position_creation_log"].append(log_entry)
                    return None

            # timestamp equity limit
            timestamp_str = str(timestamp)
            current_timestamp_usage = self.equity_used_per_timestamp.get(
                timestamp_str, 0
            )
            max_equity_per_timestamp = self.initial_equity * 0.05

            if current_timestamp_usage + position_value > max_equity_per_timestamp:
                log_entry["reason_rejected"] = "Exceeded timestamp equity limit"
                self.results["position_creation_log"].append(log_entry)
                return None

            # risk/reward
            risk = price - stop_loss
            take_profit = price + (risk * 1.5)

            entry_fee = self.calculate_trading_fees(position_size, price)

            log_entry["position_created"] = True
            log_entry["entry_price"] = price
            log_entry["stop_loss"] = stop_loss
            log_entry["take_profit"] = take_profit
            log_entry["position_size"] = position_size
            log_entry["position_value"] = position_value
            self.results["position_creation_log"].append(log_entry)

            # Update equity used for this timestamp
            self.equity_used_per_timestamp[timestamp_str] = (
                current_timestamp_usage + position_value
            )

            return EnhancedPosition(
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                position_value=position_value,
                entry_fee=entry_fee,
                currency=row["slug"],
                entry_timestamp=timestamp,
                prediction_timestamp=timestamp,
            )
        except Exception as e:
            print(f"Error creating position for {row['slug']}: {str(e)}")
            return None

    def process_existing_positions(
        self, current_data: pd.DataFrame, timestamp: pd.Timestamp
    ):
        closed_positions = {}

        for currency, position in list(self.results["positions"].items()):
            current_row = current_data[current_data["slug"] == currency]

            if current_row.empty:
                if self.verbose:
                    print(f"No data found for {currency} at {timestamp}")
                continue

            current_price = current_row["quote.USD.price"].iloc[0]

            # Check if position has been open for more than 5 hours
            if (timestamp - position.entry_timestamp) > pd.Timedelta(hours=5): # change hours here
                
                exit_price = current_price
                volume_24h = current_row["quote.USD.volume_24h"].iloc[0]
                pnl = self.risk_manager.calculate_pnl(
                    position, current_price, volume_24h
                )
                self.risk_manager.update_equity(pnl)

                
                self.results["trades_history"].append(
                    {
                        "currency": currency,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "entry_timestamp": position.entry_timestamp,
                        "exit_timestamp": timestamp,
                        "return": pnl,
                        "position_size": position.position_size,
                        "position_value": position.position_value,
                        "fees": position.entry_fee
                        + self.risk_manager.calculate_trading_fees(
                            position.position_size, exit_price
                        ),
                        "outcome": "time_limit_reached",
                    }
                )

                closed_positions[currency] = pnl
                del self.results["positions"][currency]
            else:
                # Check stop loss and take profit
                if (
                    current_price <= position.stop_loss
                    or current_price >= position.take_profit
                ):
                    exit_price = current_price
                    volume_24h = current_row["quote.USD.volume_24h"].iloc[0]
                    pnl = self.risk_manager.calculate_pnl(
                        position, current_price, volume_24h
                    )
                    self.risk_manager.update_equity(pnl)

                  
                    self.results["trades_history"].append(
                        {
                            "currency": currency,
                            "entry_price": position.entry_price,
                            "exit_price": exit_price,
                            "entry_timestamp": position.entry_timestamp,
                            "exit_timestamp": timestamp,
                            "return": pnl,
                            "position_size": position.position_size,
                            "position_value": position.position_value,
                            "fees": position.entry_fee
                            + self.risk_manager.calculate_trading_fees(
                                position.position_size, exit_price
                            ),
                            "outcome": (
                                "stop_loss"
                                if current_price <= position.stop_loss
                                else "take_profit"
                            ),
                        }
                    )

                    closed_positions[currency] = pnl
                    del self.results["positions"][currency]
                else:
                    # Update unrealized PnL
                    unrealized_pnl = 0  
                    for pos in self.results["positions"].values():
                        unrealized_pnl += (current_price - pos.entry_price) * pos.position_size

                    self.results["equity_curve"][-1] += unrealized_pnl

        return closed_positions

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

            closed_positions = self.process_existing_positions(current_data, timestamp)
            if closed_positions and self.verbose:
                print(f"Closed positions: {closed_positions}")

            if next_timestamp:
                predicted_currencies = self.get_predictions_for_timestamp(timestamp)
                for currency in predicted_currencies:
                    if currency not in self.results["positions"]:
                        currency_data = current_data[current_data["slug"] == currency]
                        if not currency_data.empty:
                            new_position = self.create_position(
                                currency_data.iloc[0], timestamp, next_timestamp
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

        self.calculate_metrics()
        return self.results

    def calculate_metrics(self):
        if not self.results["returns"]:
            self.results["metrics"] = {}
            return

        returns_series = pd.Series(self.results["returns"])
        equity_series = pd.Series(self.results["equity_curve"])

        # Sharpe Ratio
        sharpe_ratio = (
            np.sqrt(252) * returns_series.mean() / returns_series.std()
            if returns_series.std() != 0
            else 0
        )

        # Maximum Drawdown
        rolling_max = equity_series.expanding().max()
        drawdowns = equity_series / rolling_max - 1
        max_drawdown = drawdowns.min()

        # Win Rate and Profit Factor
        if self.results["trades_history"]:
            winning_trades = [
                t for t in self.results["trades_history"] if t["return"] > 0
            ]
            losing_trades = [
                t for t in self.results["trades_history"] if t["return"] <= 0
            ]

            win_rate = len(winning_trades) / len(self.results["trades_history"])

            gross_profit = (
                sum(t["return"] for t in winning_trades) if winning_trades else 0
            )
            gross_loss = (
                abs(sum(t["return"] for t in losing_trades)) if losing_trades else 0
            )
            profit_factor = (
                gross_profit / gross_loss if gross_loss != 0 else float("inf")
            )

            avg_win = (
                np.mean([t["return"] for t in winning_trades]) if winning_trades else 0
            )
            avg_loss = (
                np.mean([t["return"] for t in losing_trades]) if losing_trades else 0
            )
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0

        self.results["metrics"] = {
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.results["trades_history"]),
            "average_win": avg_win,
            "average_loss": avg_loss,
            "final_equity": self.risk_manager.current_equity,
            "return_on_initial_equity": (
                self.risk_manager.current_equity / self.initial_equity - 1
            ),
        }

    def plot_results(self):
        plt.style.use('seaborn-v0_8-whitegrid')
    
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
        fig.set_facecolor('white')

        # Classic professional colors
        equity_colors = ('#1f77b4', '#ff7f0e')  # Classic blue and orange
        drawdown_colors = ('#d62728', '#ff9896')  # Deep and light red
        sharpe_color = '#2ca02c'  # Forest green
        returns_color = '#17becf'  # Turquoise
        
        timestamps = pd.to_datetime(self.results["timestamps"])

        realized_equity = [self.initial_equity]
        unrealized_equity = [self.initial_equity]
        for i, timestamp in enumerate(timestamps):
            realized_pnl = sum(
                trade["return"]
                for trade in self.results["trades_history"]
                if trade["exit_timestamp"] <= timestamp
            )
            unrealized_pnl = sum(
                (
                    (
                        self.crypto_df.loc[
                            (self.crypto_df["timestamp"] == timestamp) & 
                            (self.crypto_df["slug"] == pos.currency), 
                            "quote.USD.price"
                        ].iloc[0]
                        if not self.crypto_df.loc[
                            (self.crypto_df["timestamp"] == timestamp) & 
                            (self.crypto_df["slug"] == pos.currency)
                        ].empty
                        else pos.entry_price
                    )
                    - pos.entry_price
                ) * pos.position_size
                for pos in self.results["positions"].values()
                if pos.entry_timestamp <= timestamp
            )

            realized_equity.append(self.initial_equity + realized_pnl)
            unrealized_equity.append(self.initial_equity + realized_pnl + unrealized_pnl)

        # Plot 1: Equity Curves 
        ax1.plot(timestamps, realized_equity[1:], label="Realized Equity", 
         color=equity_colors[0], linewidth=2)
        ax1.plot(timestamps, unrealized_equity[1:], label="Unrealized Equity", 
         color=equity_colors[1], linewidth=2)
        ax1.set_title("Equity Curves", pad=20, fontsize=14, fontweight='bold')
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Plot 2: Drawdown 
        equity_series = pd.Series(unrealized_equity[1:], index=timestamps)
        rolling_max = equity_series.expanding().max()
        drawdowns = equity_series / rolling_max - 1
        ax2.fill_between(timestamps, drawdowns, 0, color=drawdown_colors[0], alpha=0.3)
        ax2.plot(timestamps, drawdowns, color=drawdown_colors[1], linewidth=1, alpha=0.5)
        ax2.set_title(f"Drawdown (Max: {drawdowns.min():.2%})", pad=20, fontsize=14, fontweight='bold')
        ax2.set_ylabel("Drawdown", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

        # Plot 3: Sharpe Ratio 
        returns = pd.Series(unrealized_equity).pct_change().dropna()
        rolling_sharpe = returns.rolling(window=10).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0
        )
        ax3.plot(timestamps[0:], rolling_sharpe, color=sharpe_color, linewidth=2)
        ax3.set_title("Rolling Sharpe Ratio (30-day window)", pad=20, fontsize=14, fontweight='bold')
        ax3.set_ylabel("Sharpe Ratio", fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

        # Plot 4: Cumulative Returns 
        cumulative_returns = (equity_series / self.initial_equity - 1) * 100
        ax4.plot(timestamps, cumulative_returns, color=returns_color, linewidth=2)
        ax4.set_title("Cumulative Returns (%)", pad=20, fontsize=14, fontweight='bold')
        ax4.set_ylabel("Cumulative Returns (%)", fontsize=12)
        ax4.set_xlabel("Time", fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

        # styling for all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(timestamps.min(), timestamps.max())
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x')
            ax.margins(x=0.02)

        plt.tight_layout(pad=7.0)
        plt.show()

def main():
    crypto_df = pd.read_csv(rf"C:/Users/{os.getenv('USER')}/Desktop/CryptoAPI.csv")
    promising_currencies_df = pd.read_csv(
        rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/PromisingCurrencies.csv"
    )

    crypto_df["timestamp"] = pd.to_datetime(crypto_df["timestamp"])
    promising_currencies_df["timestamp"] = pd.to_datetime(
        promising_currencies_df["timestamp"]
    )

    initial_equity = 1000.0 # change initial money amount here
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
