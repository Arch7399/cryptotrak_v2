from typing import List, Optional, Tuple
import math

from tests.models.position import EnhancedPosition


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
        take_profit_levels: List[float] = [
            1.5,
            2.0,
            2.5,
        ],  # modify take profit levels here
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
        self.max_volatility = 0.1  # adjust volatility max here
        self.max_price_impact = 0.05

    def update_equity(self, pnl: float):
        self.current_equity += pnl

    def set_trailing_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        current_stop_loss: float,
        highest_price: float,
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
        self, entry_price: float, current_tp_level: int
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
        self, current_position_size: float, tp_level: int
    ) -> float:
        """
        Calculates how much of the position to exit at current take profit level.
        """
        exit_size = current_position_size * (
            self.position_size_reduction / (tp_level + 1)
        )
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
