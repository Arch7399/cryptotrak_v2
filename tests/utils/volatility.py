import pandas as pd
import numpy as np


class VolatilityCalculator:
    def calculate_volatility(self, row: pd.Series) -> float:
        try:
            changes = [
                row["quote.USD.percent_change_1h"],
                row["quote.USD.percent_change_24h"],
            ]
            valid_changes = [x / 100 for x in changes if pd.notna(x)]
            return max(np.std(valid_changes) if valid_changes else 0.02, 0.02)
        except Exception as e:
            print(f"Error calculating volatility: {str(e)}")
            return 0.02
