import pandas as pd
import numpy as np
import talib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class MetricWeights:
    """Data class to store weights for various scoring calculations."""

    market_dominance: Dict[str, float] = None
    volume_stability: Dict[str, float] = None
    combined_score: Dict[str, float] = None
    market_stability: Dict[str, float] = None

    def __post_init__(self):
        self.market_dominance = {
            "market_cap": 0.4,
            "volume": 0.3,
            "pairs": 0.2,
            "impact": 0.1,
        }

        self.volume_stability = {
            "volatility": 0.35,
            "mcap_stability": 0.30,
            "pair_volume": 0.20,
            "price_correlation": 0.15,
        }

        self.combined_score = {
            "volume_momentum": 0.2,
            "percent_change_24h": 0.2,
            "percent_change_7d": 0.2,
            "rsi": 0.1,
            "macd_hist": 0.1,
            "bb_width": 0.1,
            "stoch_k": 0.1,
        }

        self.market_stability = {
            "dominance": 0.6,
            "volume_stability": 0.4,
        }


class CryptoMetricsCalculator:
    """Main class for calculating various cryptocurrency metrics."""

    def __init__(self, weights: Optional[MetricWeights] = None):
        self.weights = weights or MetricWeights()
        self.required_columns = {
            "quote.USD.price",
            "quote.USD.volume_24h",
            "quote.USD.market_cap",
            "quote.USD.volume_change_24h",
            "quote.USD.percent_change_1h",
            "quote.USD.percent_change_24h",
            "quote.USD.percent_change_7d",
            "quote.USD.percent_change_30d",
            "quote.USD.tvl",
            "circulating_supply",
            "max_supply",
            "num_market_pairs",
        }

    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate that required columns are present in the DataFrame."""
        missing_columns = self.required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    @staticmethod
    def _safe_numeric_conversion(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Safely convert multiple columns to numeric values."""
        df = df.copy()
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def calculate_liquidity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity-related metrics."""
        numeric_columns = [
            "quote.USD.volume_24h",
            "quote.USD.market_cap",
            "num_market_pairs",
        ]
        df = self._safe_numeric_conversion(df, numeric_columns)

        # Vectorized calculation
        df["volume_to_market_cap_ratio"] = (
            df["quote.USD.volume_24h"] / df["quote.USD.market_cap"]
        )
        return df

    def calculate_momentum_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-related metrics using vectorized operations."""
        momentum_columns = [
            "quote.USD.percent_change_1h",
            "quote.USD.percent_change_24h",
            "quote.USD.percent_change_7d",
            "quote.USD.percent_change_30d",
        ]
        df["momentum_score"] = df[momentum_columns].mean(axis=1)
        return df

    def calculate_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility metrics using vectorized operations."""
        percent_changes = df[
            [
                "quote.USD.percent_change_1h",
                "quote.USD.percent_change_24h",
                "quote.USD.percent_change_7d",
            ]
        ]
        df["volatility"] = percent_changes.std(axis=1)
        return df

    def calculate_basic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic market metrics."""
        df["circulating_to_max_supply_ratio"] = (
            df["circulating_supply"] / df["max_supply"]
        )
        df["tvl_ratio"] = df["quote.USD.market_cap"] / df["quote.USD.tvl"]
        df["price_to_supply_ratio"] = df["quote.USD.price"] / df["circulating_supply"]
        return df

    def calculate_volume_momentum_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume momentum and moving averages."""
        df["volume_momentum"] = df["quote.USD.volume_change_24h"]
        df["7d_moving_avg"] = df["quote.USD.price"].rolling(window=7).mean()
        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using vectorized operations where possible."""
        close = df["quote.USD.price"].values

        # Calculate indicators using TA-Lib
        indicators = {
            "SMA_50": partial(talib.SMA, timeperiod=50),
            "SMA_200": partial(talib.SMA, timeperiod=200),
            "EMA_20": partial(talib.EMA, timeperiod=20),
            "RSI": partial(talib.RSI, timeperiod=14),
        }

        for name, func in indicators.items():
            df[name] = func(close)

        # MACD calculation
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # Bollinger Bands
        df["BB_upper"], df["BB_middle"], df["BB_lower"] = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]

        # ATR calculation
        price_change = np.abs(close - np.roll(close, 1))
        df["ATR"] = talib.SMA(price_change, timeperiod=14)

        # Stochastic Oscillator
        rolling_window = 14
        rolling_low = pd.Series(close).rolling(window=rolling_window).min()
        rolling_high = pd.Series(close).rolling(window=rolling_window).max()

        k = 100 * ((close - rolling_low) / (rolling_high - rolling_low))
        df["STOCH_K"] = k
        df["STOCH_D"] = talib.SMA(k, timeperiod=3)

        return df

    def calculate_market_dominance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market dominance metrics using vectorized operations."""
        # Calculate totals
        total_market_cap = df["quote.USD.market_cap"].sum()
        total_volume = df["quote.USD.volume_24h"].sum()
        max_pairs = df["num_market_pairs"].max()

        # Calculate dominance metrics
        df["market_cap_dominance"] = df["quote.USD.market_cap"] / total_market_cap
        df["volume_dominance"] = df["quote.USD.volume_24h"] / total_volume
        df["pair_dominance"] = df["num_market_pairs"] / max_pairs
        df["market_impact"] = df["volatility"] * df["volume_dominance"]

        # Calculate weighted score
        weights = self.weights.market_dominance
        df["market_dominance_score"] = (
            weights["market_cap"] * df["market_cap_dominance"]
            + weights["volume"] * df["volume_dominance"]
            + weights["pairs"] * df["pair_dominance"]
            + weights["impact"] * df["market_impact"]
        )

        return df

    def calculate_volume_stability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume stability metrics using vectorized operations."""
        # Calculate basic metrics
        df["volume_volatility"] = df["quote.USD.volume_change_24h"].abs()
        df["vol_mcap_ratio"] = df["quote.USD.volume_24h"] / df["quote.USD.market_cap"]
        df["vol_mcap_stability"] = 1 / (1 + df["vol_mcap_ratio"])
        df["volume_per_pair"] = df["quote.USD.volume_24h"] / df["num_market_pairs"]
        df["vol_price_correlation"] = (
            df["momentum_score"] * df["quote.USD.volume_change_24h"]
        ).abs()

        # Normalize metrics
        metrics_to_normalize = [
            "volume_volatility",
            "vol_mcap_stability",
            "volume_per_pair",
            "vol_price_correlation",
        ]

        for metric in metrics_to_normalize:
            df[f"{metric}_normalized"] = (df[metric] - df[metric].min()) / (
                df[metric].max() - df[metric].min()
            )

        # Calculate weighted score
        weights = self.weights.volume_stability
        df["volume_stability_score"] = (
            weights["volatility"] * (1 - df["volume_volatility_normalized"])
            + weights["mcap_stability"] * df["vol_mcap_stability_normalized"]
            + weights["pair_volume"] * df["volume_per_pair_normalized"]
            + weights["price_correlation"]
            * (1 - df["vol_price_correlation_normalized"])
        )

        return df

    def calculate_market_stability_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the overall market stability index."""
        weights = self.weights.market_stability
        df["market_stability_index"] = (
            weights["dominance"] * df["market_dominance_score"]
            + weights["volume_stability"] * df["volume_stability_score"]
        )
        return df

    def calculate_combined_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate final combined score using weighted metrics."""
        weights = self.weights.combined_score
        df["combined_score"] = (
            weights["volume_momentum"] * df["volume_momentum"]
            + weights["percent_change_24h"] * df["quote.USD.percent_change_24h"]
            + weights["percent_change_7d"] * df["quote.USD.percent_change_7d"]
            + weights["rsi"] * (50 - df["RSI"].abs()) / 50
            + weights["macd_hist"] * df["MACD_hist"]
            + weights["bb_width"] * (1 - df["BB_width"])
            + weights["stoch_k"] * (50 - df["STOCH_K"].abs()) / 50
        )
        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main method to process cryptocurrency data and calculate all metrics."""
        try:
            # Validate input data
            self.validate_data(df)

            # Filter valid prices
            df = df[pd.to_numeric(df["quote.USD.price"], errors="coerce") > 0].copy()

            # Calculate all metrics in sequence
            calculation_steps = [
                self.calculate_volume_momentum_metrics,
                self.calculate_liquidity_metrics,
                self.calculate_momentum_metrics,
                self.calculate_volatility_metrics,
                self.calculate_basic_metrics,
                self.calculate_technical_indicators,
                self.calculate_market_dominance,
                self.calculate_volume_stability,
                self.calculate_market_stability_index,
                self.calculate_combined_score,
            ]

            for step in calculation_steps:
                df = step(df)

            logger.info("Successfully processed cryptocurrency metrics")
            return df

        except Exception as e:
            logger.error(f"Error processing cryptocurrency metrics: {str(e)}")
            raise


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to maintain backward compatibility with the original interface.
    """
    calculator = CryptoMetricsCalculator()
    return calculator.process_data(df)
