import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()


def prepare_data(df):
    """
    Prepare the dataset with enhanced features focusing on lower price range prediction
    """
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()

    # Add time-based features
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek

    # Convert hour to cyclical features to capture its circular nature
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Enhanced momentum indicators
    momentum_indicators = [
        "RSI",
        "MACD",
        "MACD_hist",
        "MACD_signal",  # Added MACD signal line
        "EMA_short",
        "EMA_long",
        "MOM",
        "ROC",
        "momentum_score",  # Added overall momentum score
        "momentum_divergence",  # Added momentum divergence
        "EMA_9",  # Added shorter-term EMA
        "uptrend",  # Added uptrend indicator
    ]

    # Enhanced volume indicators
    volume_indicators = [
        "OBV",
        "CMF",
        "volume_surge_score",
        "volume_momentum",
        "volume_to_market_cap_ratio",
        "volume_stability",  # Added volume stability
        "volume_stability_score",
        "volume_per_pair",  # Added volume per trading pair
        "vol_price_correlation",  # Added volume-price correlation
    ]

    # Enhanced volatility indicators
    volatility_indicators = [
        "ATR",
        "BB_width",
        "BB_%B",
        "volatility",
        "volatility_score",  # Added volatility score
        "volatility_factor",  # Added volatility factor
        "price_stability",  # Added price stability
    ]

    # Enhanced price action indicators
    price_action = [
        "quote.USD.price",
        "quote.USD.percent_change_1h",
        "quote.USD.percent_change_24h",
        "quote.USD.percent_change_7d",  # Added more timeframe changes
        "VWAP",
        "price_to_vwap",
        "discrepancy_score",
        "SMA_50",  # Added moving averages
        "SMA_200",
        "golden_cross",  # Added golden cross indicator
    ]

    # Enhanced market context indicators
    market_context = [
        "market_dominance_score",
        "liquidity_score",
        "market_impact",
        "volume_stability_score",
        "market_stability_index",  # Added market stability
        "combined_score",  # Added combined market score
        "ADX",  # Added Average Directional Index
    ]

    # Technical pattern indicators
    pattern_indicators = [
        "STOCH_K",  # Added stochastic oscillator
        "STOCH_D",
        "ta_price_score",
        "ta_volume_score",
        "promise_score",  # Added overall promise score
    ]

    # Add time features to the feature list
    time_features = ["hour_sin", "hour_cos", "day_of_week"]

    # Combine all feature groups
    feature_columns = (
        momentum_indicators
        + volume_indicators
        + volatility_indicators
        + price_action
        + market_context
        + pattern_indicators
        + time_features
    )

    # Remove any columns that don't exist in the dataframe
    existing_features = [col for col in feature_columns if col in df.columns]

    # Create feature matrix and target vector
    X = df[existing_features].copy()
    y = df["target"].copy()

    # Handle missing values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().bfill()
    X = X.fillna(0)

    return X, y


def create_model():
    """
    Create a model optimized for short-term predictions with increased complexity
    to capture time-based patterns
    """
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_depth=5,
        max_leaf_nodes=50,
        min_samples_leaf=15,
        l2_regularization=0.1,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )


def plot_predictions(timestamps, y_true, y_pred, fold_number):
    """
    Plot actual vs predicted values for a given fold
    """
    plt.figure(figsize=(15, 6))
    plt.plot(timestamps, y_true, label="Actual", color="blue", alpha=0.5)
    plt.plot(timestamps, y_pred, label="Predicted", color="red", alpha=0.5)
    plt.title(f"Actual vs Predicted Values - Fold {fold_number}")
    plt.xlabel("Time")
    plt.ylabel("Price Change")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()


def plot_prediction_improvement(scores, y_true_list, y_pred_list):
    """
    Plot model prediction improvement over folds and prediction accuracy comparison
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Plot 1: R² Score Improvement
    ax1.plot(range(1, len(scores) + 1), scores, marker="o")
    ax1.set_title("Model Performance Improvement (R² Score) Over Folds")
    ax1.set_xlabel("Fold Number")
    ax1.set_ylabel("R² Score")
    ax1.grid(True)

    # Plot 2: Prediction Accuracy Comparison
    colors = plt.cm.rainbow(np.linspace(0, 1, len(y_true_list)))
    for i, (y_true, y_pred, color) in enumerate(zip(y_true_list, y_pred_list, colors)):
        ax2.scatter(y_true, y_pred, alpha=0.5, color=color, label=f"Fold {i+1}")

    # Add perfect prediction line
    min_val = min([min(y) for y in y_true_list])
    max_val = max([max(y) for y in y_true_list])
    ax2.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.5,
        label="Perfect Prediction",
    )

    ax2.set_title("Prediction Accuracy Comparison")
    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Predicted Values")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    return fig


def evaluate_feature_importance(model, X, y):
    """
    Analyze and visualize feature importance using permutation importance
    """
    X_array = X.values

    result = permutation_importance(
        model, X_array, y, n_repeats=10, random_state=42, n_jobs=-1
    )

    importance_scores = result.importances_mean

    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": importance_scores}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance_scores)), importance_scores)
    plt.xticks(range(len(importance_scores)), X.columns, rotation=45, ha="right")
    plt.title("Feature Importance for Short-term Price Prediction")
    plt.tight_layout()

    return feature_importance


def train_and_evaluate():
    """
    Train the model using time series cross-validation
    """
    # Load data
    df = pd.read_csv(rf"C:/Users/{os.getenv('USER')}/Desktop/ml_training_data.csv")
    X, y = prepare_data(df.iloc[:-30])
    timestamps = pd.to_datetime(df.iloc[:-30]["timestamp"])

    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    model = create_model()

    scores = []
    y_true_list = []
    y_pred_list = []
    final_X_test = None
    final_y_test = None

    # Perform time series cross-validation
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        timestamps_test = timestamps.iloc[test_idx]

        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        model.fit(X_train_scaled.values, y_train)
        y_pred = model.predict(X_test_scaled.values)

        score = r2_score(y_test, y_pred)
        scores.append(score)

        # Store predictions and actual values for plotting
        y_true_list.append(y_test)
        y_pred_list.append(y_pred)

        # Create and save individual fold prediction plot
        fold_fig = plot_predictions(timestamps_test, y_test, y_pred, fold)
        fold_fig.savefig(f"plots/fold_{fold}_predictions.png")
        plt.close(fold_fig)

        final_X_test = X_test_scaled
        final_y_test = y_test

    # Create and save improvement visualization
    improvement_fig = plot_prediction_improvement(scores, y_true_list, y_pred_list)
    improvement_fig.savefig(f"plots/model_improvement.png")
    plt.close(improvement_fig)

    # Get feature importance using the last fold
    feature_importance = evaluate_feature_importance(model, final_X_test, final_y_test)

    print("Cross-validation scores:", scores)
    print("Average R² score:", np.mean(scores))
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    return model, scaler, feature_importance


if __name__ == "__main__":
    model, scaler, importance = train_and_evaluate()
