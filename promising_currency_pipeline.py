import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from functools import lru_cache
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import talib
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from data_processing import process_data
from pipeline_filters import apply_filters
from pipeline_mixed_filters import apply_tandem_filters
from latest_data import filter_latest
from config import Config


@dataclass
class FeatureWeights:
    """Stores weights for different features in promise score calculation"""

    MARKET_DOMINANCE: float = 0.08
    VOLUME_STABILITY: float = 0.04
    PRICE_STABILITY: float = 0.04
    PERCENT_CHANGE_24H: float = 0.08
    PERCENT_CHANGE_7D: float = 0.08
    RSI: float = 0.08
    UPTREND: float = 0.04
    GOLDEN_CROSS: float = 0.04
    MACD_HIST: float = 0.08
    BB_WIDTH: float = 0.04
    ADX: float = 0.08
    OBV: float = 0.04
    FLAG_SEVERITY: float = 0.12
    CMF: float = 0.02
    MARKET_STABILITY: float = 0.02
    VOL_PRICE_CORR: float = 0.02
    PRICE_ATR_RATIO: float = 0.02


class DataPreprocessor:
    """Handles data preprocessing operations"""

    @staticmethod
    def prepare_data(raw_data_path: str) -> pd.DataFrame:
        """Load and prepare initial dataset"""
        df = pd.read_csv(raw_data_path)
        return filter_latest(df).sort_values("timestamp")

    @staticmethod
    def calculate_z_scores(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Vectorized z-score calculation"""
        z_scores = pd.DataFrame()
        for col in columns:
            z_scores[f"{col}_zscore"] = stats.zscore(df[col])
        return pd.concat([df, z_scores], axis=1)

    @staticmethod
    def calculate_time_decay(
        df: pd.DataFrame, half_life: timedelta = timedelta(hours=6)
    ) -> pd.DataFrame:
        """Vectorized time decay calculation"""
        now = df["timestamp"].max()
        df["time_decay"] = np.exp(
            -np.log(2)
            * (pd.to_datetime(now) - pd.to_datetime(df["timestamp"]))
            / half_life
        )
        return df


class FlagProcessor:
    """Handles flag-related calculations"""

    POSITIVE_FLAGS = [
        "price_spike_flag",
        "volume_surge_flag",
        "bullish_momentum_breakout_flag",
        "reversal_opportunity_flag",
    ]

    NEGATIVE_FLAGS = [
        "price_crash_flag",
        "low_liquidity_flag",
        "pump_flag",
        "dump_flag",
        "market_cap_volume_discrepancy_flag",
        "false_valuation_flag",
    ]

    @classmethod
    def calculate_flag_severity(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized flag severity calculation"""
        df["positive_flags"] = df[cls.POSITIVE_FLAGS].sum(axis=1)
        df["negative_flags"] = df[cls.NEGATIVE_FLAGS].sum(axis=1)
        df["flag_severity"] = df["positive_flags"] - df["negative_flags"] * 1.5
        return df

    @staticmethod
    @lru_cache(maxsize=128)
    def calculate_flag_based_penalty(positive_flags: int, negative_flags: int) -> float:
        """Cached calculation of flag-based penalties"""
        severity_ratio = (positive_flags + 1) / (negative_flags + 1)

        if negative_flags == 0:
            bonus = min(positive_flags * 5, 20)
            return min(1 + bonus / 100, 1)
        elif negative_flags == 1:
            return 0.7 if severity_ratio > 1 else 0.6
        elif negative_flags == 2:
            return 0.4 if severity_ratio > 1.5 else 0.3
        else:
            return 0.2 if severity_ratio > 2 else 0.1


class TechnicalAnalyzer:
    """Handles technical analysis calculations"""

    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized technical indicator calculations"""
        close = df["quote.USD.price"].values

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    lambda: pd.Series(
                        talib.ADX(close, close, close, timeperiod=14), name="ADX"
                    )
                ),
                executor.submit(
                    lambda: pd.Series(
                        talib.OBV(close, df["quote.USD.volume_24h"].values), name="OBV"
                    )
                ),
            ]

            results = [future.result() for future in futures]

        for result in results:
            df[result.name] = result

        return df

    @staticmethod
    def calculate_market_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market-related metrics"""
        df["market_dominance"] = (
            df["quote.USD.market_cap"] / df["quote.USD.market_cap"].sum()
        )

        # Vectorized rolling calculations
        rolling_volume = df["quote.USD.volume_24h"].rolling(window=7)
        rolling_price = df["quote.USD.price"].rolling(window=7)

        df["volume_stability"] = rolling_volume.std() / rolling_volume.mean()
        df["price_stability"] = rolling_price.std() / rolling_price.mean()

        # Trend indicators
        df["uptrend"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
        df["golden_cross"] = (
            (df["SMA_50"] > df["SMA_200"])
            & (df["SMA_50"].shift(1) <= df["SMA_200"].shift(1))
        ).astype(int)

        return df


class AnomalyDetector:
    """Handles anomaly detection"""

    def __init__(self, contamination: float = 0.1):
        self.features = [
            "market_dominance",
            "volume_stability",
            "price_stability",
            "quote.USD.percent_change_24h",
            "quote.USD.percent_change_7d",
            "RSI",
            "MACD_hist",
            "BB_width",
        ]
        self.imputer = SimpleImputer(strategy="mean")
        self.iso_forest = IsolationForest(contamination=contamination, random_state=42)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform anomaly detection"""
        available_features = [f for f in self.features if f in df.columns]

        if missing := set(self.features) - set(available_features):
            print(f"Warning: Missing features: {missing}")

        X = self.imputer.fit_transform(df[available_features])
        df["anomaly"] = self.iso_forest.fit_predict(X)
        return df


class ScoreCalculator:
    """Handles promise score calculations"""

    def __init__(self, weights: Optional[FeatureWeights] = None):
        self.weights = weights or FeatureWeights()

    def calculate_base_score(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized base score calculation"""
        components = {
            "market_dominance": df["market_dominance"] * self.weights.MARKET_DOMINANCE,
            "volume_stability": (1 - df["volume_stability"])
            * self.weights.VOLUME_STABILITY,
            "price_stability": (1 - df["price_stability"])
            * self.weights.PRICE_STABILITY,
            "percent_change_24h": df["quote.USD.percent_change_24h"]
            * self.weights.PERCENT_CHANGE_24H,
            "percent_change_7d": df["quote.USD.percent_change_7d"]
            * self.weights.PERCENT_CHANGE_7D,
            "rsi": (df["RSI"] - 50).abs() / 50 * self.weights.RSI,
            "uptrend": df["uptrend"] * self.weights.UPTREND,
            "golden_cross": df["golden_cross"] * self.weights.GOLDEN_CROSS,
            "macd": df["MACD_hist"] * self.weights.MACD_HIST,
            "bb_width": df["BB_width"] * self.weights.BB_WIDTH,
            "adx": df["ADX"] / 100 * self.weights.ADX,
            "obv": df["OBV"].pct_change(fill_method=None) * self.weights.OBV,
            "flag_severity": df["flag_severity"] * self.weights.FLAG_SEVERITY,
            "cmf": df["CMF"] * self.weights.CMF,
            "market_stability": df["market_stability_index"]
            * self.weights.MARKET_STABILITY,
            "vol_price_corr": df["vol_price_correlation_normalized"]
            * self.weights.VOL_PRICE_CORR,
            "price_atr": df["price_to_ATR_ratio"].clip(lower=0, upper=1)
            * self.weights.PRICE_ATR_RATIO,
        }

        return pd.DataFrame(components).sum(axis=1)

    def apply_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply score adjustments"""
        df["promise_score"] = self.calculate_base_score(df)

        # Vectorized adjustments
        conditions = [
            (df["volatility_score"] > 0.5),
            ((df["volume_stability"] < 0.3) & (df["volume_dominance"] > 0.01)),
            ((df["vol_mcap_ratio"] > 0.1) & (df["vol_mcap_ratio"] < 0.5)),
            (df["anomaly"] == -1),
            ((df["RSI"] > 80) | (df["RSI"] < 20)),
        ]

        choices = [
            df["promise_score"] * (1 - df["volatility_score"] * 0.5),
            df["promise_score"] * 1.1,
            df["promise_score"] * 1.05,
            df["promise_score"] * 0.6,
            df["promise_score"] * 0.75,
        ]

        df["promise_score"] = np.select(
            conditions, choices, default=df["promise_score"]
        )

        return df


class CurrencyAnalyzer:
    """Main class for currency analysis"""

    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.flag_processor = FlagProcessor()
        self.technical_analyzer = TechnicalAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.score_calculator = ScoreCalculator()

    def analyze(self, raw_data_path: str) -> pd.DataFrame:
        """Perform complete analysis pipeline"""
        df = self.preprocessor.prepare_data(raw_data_path)
        df = process_data(df)
        df = apply_filters(df)
        df = apply_tandem_filters(df)

        # Apply all analysis steps
        df = self.technical_analyzer.calculate_market_metrics(df)
        df = self.technical_analyzer.calculate_technical_indicators(df)
        df = self.flag_processor.calculate_flag_severity(df)
        df = self.anomaly_detector.detect(df)
        df = self.score_calculator.apply_adjustments(df)

        # Apply price filter
        mask = df["quote.USD.price"].between(Config.min_usd_price, Config.max_usd_price)
        return df[mask].nlargest(30, "promise_score")


def performers() -> List[str]:
    """Get top performing currencies"""
    raw_data_path = f"C:/Users/{os.getenv('USER')}/Desktop/CryptoAPI.csv"
    analyzer = CurrencyAnalyzer()

    promising_currencies = analyzer.analyze(raw_data_path)

    output_path = (
        f"C:/Users/{os.getenv('USER')}/Desktop/Analysis/PromisingCurrencies.csv"
    )
    promising_currencies.to_csv(
        output_path, mode="a", header=not os.path.exists(output_path), index=False
    )

    return promising_currencies["name"].head(5).tolist()


if __name__ == "__main__":
    top_performers = performers()
    print("Top 5 performing currencies:", top_performers)
