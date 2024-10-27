import numpy as np
import pandas as pd


class MetricsCalculator:
    @staticmethod
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
