from typing import Dict, Optional
from models.position import EnhancedPosition
from risk_management.risk_manager import EnhancedRiskManagement
from utils.volatility import VolatilityCalculator
from config import Config
import pandas as pd
import numpy as np
import math


class PositionManager:
    def __init__(self, risk_manager: EnhancedRiskManagement, fee_rate: float):
        self.risk_manager = risk_manager
        self.fee_rate = fee_rate
        self.volatility = VolatilityCalculator

    def calculate_trading_fees(self, position_size: float, price: float) -> float:
        return position_size * price * self.fee_rate

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
            if (timestamp - position.entry_timestamp) > pd.Timedelta(
                hours=Config.position_hold_time_limit
            ):  # change hours here

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
                        unrealized_pnl += (
                            current_price - pos.entry_price
                        ) * pos.position_size

                    self.results["equity_curve"][-1] += unrealized_pnl

        return closed_positions

    def process_positions(
        self,
        current_data: pd.DataFrame,
        positions: Dict[str, EnhancedPosition],
        timestamp: pd.Timestamp,
    ) -> Dict[str, float]:
        closed_positions = {}
        current_data_dict = current_data.set_index("slug").to_dict(orient="index")

        for currency, position in list(positions.items()):
            if currency not in current_data_dict:
                continue

            current_price = current_data_dict[currency]["quote.USD.price"]
            # Update highest price and trailing stop loss
            if current_price > position.highest_price:
                position.highest_price = current_price
                position.stop_loss = self.risk_manager.set_trailing_stop_loss(
                    position.entry_price,
                    current_price,
                    position.stop_loss,
                    position.highest_price,
                )

            # Check stop loss
            if current_price <= position.stop_loss:
                exit_fee = self.calculate_trading_fees(
                    position.position_size, current_price
                )
                pnl = self.risk_manager.calculate_pnl(
                    position,
                    current_price,
                    current_data_dict[currency]["quote.USD.volume_24h"],
                )
                self.risk_manager.update_equity(pnl)

                closed_positions[currency] = pnl
                del self.results["positions"][currency]

                # Record trade
                self.record_trade(
                    currency,
                    position,
                    current_price,
                    timestamp,
                    "stop_loss",
                    exit_fee,
                    pnl,
                )

            # Check take profit
            elif current_price >= position.take_profit:
                # Calculate partial position size to exit
                exit_size = self.risk_manager.calculate_partial_exit_size(
                    position.position_size, position.current_tp_level
                )

                exit_fee = self.calculate_trading_fees(exit_size, current_price)

                # Calculate PnL for partial exit
                partial_pnl = (
                    current_price - position.entry_price
                ) * exit_size - exit_fee
                self.risk_manager.update_equity(partial_pnl)

                # Update position size
                position.position_size -= exit_size

                # Get next take profit level
                next_tp, next_tp_level = self.risk_manager.get_next_take_profit(
                    position.entry_price, position.current_tp_level
                )

                if next_tp is None or position.position_size <= 0:
                    if position.position_size > 0:
                        final_exit_fee = self.calculate_trading_fees(
                            position.position_size, current_price
                        )
                        final_pnl = (
                            current_price - position.entry_price
                        ) * position.position_size - final_exit_fee
                        self.risk_manager.update_equity(final_pnl)

                        partial_pnl += final_pnl

                    closed_positions[currency] = partial_pnl
                    del self.results["positions"][currency]

                    # Record final trade
                    self.record_trade(
                        currency,
                        position,
                        current_price,
                        timestamp,
                        "take_profit_final",
                        exit_fee + final_exit_fee,
                        partial_pnl,
                    )
                else:
                    # Update position for next take profit level
                    position.take_profit = next_tp
                    position.current_tp_level = next_tp_level
                    position.position_value = position.position_size * current_price

                    # Record partial trade
                    self.record_trade(
                        currency,
                        position,
                        current_price,
                        timestamp,
                        "take_profit_partial",
                        exit_fee,
                        partial_pnl,
                    )

        return closed_positions

    def create_position(
        self, row: pd.Series, timestamp: pd.Timestamp, next_timestamp: pd.Timestamp
    ) -> Optional[EnhancedPosition]:
        try:
            price = row["quote.USD.price"]
            volume_24h = row["quote.USD.volume_24h"]
            market_cap = (
                row["quote.USD.market_cap"] if "quote.USD.market_cap" in row else None
            )
            volatility = self.volatility.calculate_volatility(self, row)

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
                    log_entry["reason_rejected"] = (
                        "Position size too small after value limit"
                    )
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
