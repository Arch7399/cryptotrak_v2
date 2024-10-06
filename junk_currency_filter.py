import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class FilterCriteria:
    """
    Data class to store filter criteria for cryptocurrency filtering.
    Using a dataclass makes the criteria more maintainable and reusable.
    """

    min_market_cap: float = 1_000_000  # $1 million
    min_volume_24h: float = 10_000  # $10,000
    min_age_days: int = 30
    max_circulating_supply_ratio: float = 0.9  # 90% of max supply
    required_columns: tuple = (
        "quote.USD.market_cap",
        "quote.USD.volume_24h",
        "circulating_supply",
        "max_supply",
        "last_updated",
        "date_added",
    )


class CurrencyFilterStats:
    """
    Class to track and report filtering statistics.
    Separates the statistics tracking logic from the main filtering logic.
    """

    def __init__(self, total_currencies: int):
        self.total_currencies = total_currencies
        self.stats: Dict[str, int] = {
            "zero_market_cap": 0,
            "low_market_cap": 0,
            "low_volume": 0,
            "young_currencies": 0,
        }

    def update_stats(self, filter_mask: pd.Series, category: str) -> None:
        """Update statistics for a specific filtering category."""
        self.stats[category] = int(filter_mask.sum())

    def log_results(self, filtered_count: int, criteria: FilterCriteria) -> None:
        """Log the filtering results with detailed statistics."""
        junk_currencies = self.total_currencies - filtered_count

        logger.info(f"Total currencies: {self.total_currencies}")
        logger.info(f"Currencies after filtering: {filtered_count}")
        logger.info(
            f"Filtered out {junk_currencies} junk currencies "
            f"({junk_currencies/self.total_currencies:.2%})"
        )

        logger.info("\nDetailed filtering results:")
        logger.info(f"Currencies with zero market cap: {self.stats['zero_market_cap']}")
        logger.info(
            f"Currencies with market cap below ${criteria.min_market_cap:,}: "
            f"{self.stats['low_market_cap']}"
        )
        logger.info(
            f"Currencies with 24h volume below ${criteria.min_volume_24h:,}: "
            f"{self.stats['low_volume']}"
        )
        logger.info(
            f"Currencies younger than {criteria.min_age_days} days: "
            f"{self.stats['young_currencies']}"
        )


class CurrencyFilter:
    """
    Main class for filtering cryptocurrency data.
    Implements the filtering logic with optimized performance and clear structure.
    """

    def __init__(self, criteria: Optional[FilterCriteria] = None):
        self.criteria = criteria or FilterCriteria()

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that the input DataFrame has all required columns."""
        missing_columns = set(self.criteria.required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _prepare_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare numeric columns using vectorized operations."""
        numeric_columns = [
            "quote.USD.market_cap",
            "quote.USD.volume_24h",
            "circulating_supply",
            "max_supply",
        ]

        # Create a copy to avoid modifying the input DataFrame
        df = df.copy()

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _calculate_age(self, df: pd.DataFrame) -> pd.Series:
        """Calculate cryptocurrency age using vectorized operations."""
        return (
            pd.to_datetime(df["last_updated"]) - pd.to_datetime(df["date_added"])
        ).dt.days

    def _create_filter_masks(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create boolean masks for each filtering criterion."""
        return {
            "zero_market_cap": df["quote.USD.market_cap"] == 0,
            "low_market_cap": (
                (df["quote.USD.market_cap"] > 0)
                & (df["quote.USD.market_cap"] < self.criteria.min_market_cap)
            ),
            "low_volume": df["quote.USD.volume_24h"] < self.criteria.min_volume_24h,
            "young_currencies": df["age_days"] < self.criteria.min_age_days,
        }

    def filter_currencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out junk cryptocurrencies based on defined criteria.

        Args:
            df: DataFrame containing cryptocurrency data

        Returns:
            DataFrame containing only the currencies that meet the filtering criteria

        Raises:
            ValueError: If required columns are missing from the DataFrame
        """
        # Validate input
        self._validate_dataframe(df)

        # Initialize statistics tracker
        stats = CurrencyFilterStats(len(df))

        # Prepare data
        df = self._prepare_numeric_columns(df)
        df["age_days"] = self._calculate_age(df)

        # Create filter masks and update statistics
        filter_masks = self._create_filter_masks(df)
        for category, mask in filter_masks.items():
            stats.update_stats(mask, category)

        # Apply main filters using vectorized operations
        supply_filter = (
            (
                (df["max_supply"] > 0)
                & (
                    (
                        df["circulating_supply"] / df["max_supply"]
                        <= self.criteria.max_circulating_supply_ratio
                    )
                    | (df["circulating_supply"] / df["max_supply"] >= 0.9)
                )
            )
            | (df["max_supply"].isnull())
            | (df["max_supply"] == 0)
        )

        df_filtered = df[
            (df["quote.USD.market_cap"] > 0)
            & (df["quote.USD.market_cap"] >= self.criteria.min_market_cap)
            & (df["quote.USD.volume_24h"] >= self.criteria.min_volume_24h)
            & (df["age_days"] >= self.criteria.min_age_days)
            & supply_filter
        ]

        # Log results
        stats.log_results(len(df_filtered), self.criteria)

        return df_filtered


def filter_junk_currencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to maintain backward compatibility with the original interface.

    Args:
        df: DataFrame containing cryptocurrency data

    Returns:
        DataFrame containing only the currencies that meet the filtering criteria
    """
    return CurrencyFilter().filter_currencies(df)
