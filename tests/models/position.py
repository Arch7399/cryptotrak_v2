from dataclasses import dataclass, field
import pandas as pd


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
