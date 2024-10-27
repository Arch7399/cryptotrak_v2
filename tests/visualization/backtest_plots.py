import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
        plt.style.use("seaborn-v0_8-whitegrid")
        colors = {
            "equity": "#1f77b4",  # Blue
            "win_rate": "#2ca02c",  # Green
            "hist": "#1f77b4",  # Blue
            "returns": "#ff7f0e",  # Orange
            "zero_line": "#d62728",  # Red
        }

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.patch.set_facecolor("white")

        # Plot 1: Actual Equity Changes
        full_equity_timeline = [self.initial_equity] + trades_df["equity"].tolist()
        timestamps = [
            trades_df["exit_timestamp"].iloc[0] - pd.Timedelta(hours=5)
        ] + trades_df["exit_timestamp"].tolist()
        axes[0, 0].plot(
            timestamps, full_equity_timeline, color=colors["equity"], linewidth=2
        )
        axes[0, 0].set_title(
            "Equity Changes Over Time", pad=20, fontsize=12, fontweight="bold"
        )
        axes[0, 0].set_ylabel("Equity ($)", fontsize=10)
        axes[0, 0].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
        )
        axes[0, 0].tick_params(axis="both", labelsize=9)

        # Plot 2: Rolling Win Rate
        axes[0, 1].plot(
            trades_df["exit_timestamp"],
            trades_df["rolling_win_rate"],
            color=colors["win_rate"],
            linewidth=2,
        )
        axes[0, 1].set_title(
            "Rolling Win Rate (10 trades)", pad=20, fontsize=12, fontweight="bold"
        )
        axes[0, 1].set_ylabel("Win Rate", fontsize=10)
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))
        axes[0, 1].tick_params(axis="both", labelsize=9)
        axes[0, 1].set_ylim([0, max(trades_df["rolling_win_rate"]) * 1.1])

        # Plot 3: Return Distribution
        trades_df["return"].hist(
            ax=axes[1, 0], bins=20, color=colors["hist"], alpha=0.7, edgecolor="white"
        )
        axes[1, 0].set_title(
            "Return Distribution ($)", pad=20, fontsize=12, fontweight="bold"
        )
        axes[1, 0].set_xlabel("Return ($)", fontsize=10)
        axes[1, 0].set_ylabel("Frequency", fontsize=10)
        axes[1, 0].tick_params(axis="both", labelsize=9)
        axes[1, 0].xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
        )

        # Plot 4: Trade Returns Over Time
        axes[1, 1].plot(
            trades_df["exit_timestamp"],
            trades_df["return"],
            color=colors["returns"],
            linewidth=2,
        )
        axes[1, 1].axhline(y=0, color=colors["zero_line"], linestyle="-", alpha=0.3)
        axes[1, 1].set_title(
            "Trade Returns Over Time", pad=20, fontsize=12, fontweight="bold"
        )
        axes[1, 1].set_ylabel("Return ($)", fontsize=10)
        axes[1, 1].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
        )
        axes[1, 1].tick_params(axis="both", labelsize=9)

        for ax in axes.flat:
            if ax is not axes[1, 0]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.tick_params(axis="x", rotation=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
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

    def plot_results(self):
        plt.style.use("seaborn-v0_8-whitegrid")

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
        fig.set_facecolor("white")

        # Classic professional colors
        equity_colors = ("#1f77b4", "#ff7f0e")  # Classic blue and orange
        drawdown_colors = ("#d62728", "#ff9896")  # Deep and light red
        sharpe_color = "#2ca02c"  # Forest green
        returns_color = "#17becf"  # Turquoise

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
                            (self.crypto_df["timestamp"] == timestamp)
                            & (self.crypto_df["slug"] == pos.currency),
                            "quote.USD.price",
                        ].iloc[0]
                        if not self.crypto_df.loc[
                            (self.crypto_df["timestamp"] == timestamp)
                            & (self.crypto_df["slug"] == pos.currency)
                        ].empty
                        else pos.entry_price
                    )
                    - pos.entry_price
                )
                * pos.position_size
                for pos in self.results["positions"].values()
                if pos.entry_timestamp <= timestamp
            )

            realized_equity.append(self.initial_equity + realized_pnl)
            unrealized_equity.append(
                self.initial_equity + realized_pnl + unrealized_pnl
            )

        # Plot 1: Equity Curves
        ax1.plot(
            timestamps,
            realized_equity[1:],
            label="Realized Equity",
            color=equity_colors[0],
            linewidth=2,
        )
        ax1.plot(
            timestamps,
            unrealized_equity[1:],
            label="Unrealized Equity",
            color=equity_colors[1],
            linewidth=2,
        )
        ax1.set_title("Equity Curves", pad=20, fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Plot 2: Drawdown
        equity_series = pd.Series(unrealized_equity[1:], index=timestamps)
        rolling_max = equity_series.expanding().max()
        drawdowns = equity_series / rolling_max - 1
        ax2.fill_between(timestamps, drawdowns, 0, color=drawdown_colors[0], alpha=0.3)
        ax2.plot(
            timestamps, drawdowns, color=drawdown_colors[1], linewidth=1, alpha=0.5
        )
        ax2.set_title(
            f"Drawdown (Max: {drawdowns.min():.2%})",
            pad=20,
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_ylabel("Drawdown", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))

        # Plot 3: Sharpe Ratio
        returns = pd.Series(unrealized_equity).pct_change().dropna()
        rolling_sharpe = returns.rolling(window=10).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0
        )
        ax3.plot(timestamps[0:], rolling_sharpe, color=sharpe_color, linewidth=2)
        ax3.set_title(
            "Rolling Sharpe Ratio (30-day window)",
            pad=20,
            fontsize=14,
            fontweight="bold",
        )
        ax3.set_ylabel("Sharpe Ratio", fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))

        # Plot 4: Cumulative Returns
        cumulative_returns = (equity_series / self.initial_equity - 1) * 100
        ax4.plot(timestamps, cumulative_returns, color=returns_color, linewidth=2)
        ax4.set_title("Cumulative Returns (%)", pad=20, fontsize=14, fontweight="bold")
        ax4.set_ylabel("Cumulative Returns (%)", fontsize=12)
        ax4.set_xlabel("Time", fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1f}%"))

        # styling for all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(timestamps.min(), timestamps.max())
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.tick_params(axis="x")
            ax.margins(x=0.02)

        plt.tight_layout(pad=7.0)
        plt.show()
