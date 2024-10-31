import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from dotenv import load_dotenv

load_dotenv()


def handle_infinite_values(data):
    """
    Convert infinite values to NaN in numpy arrays or lists
    """
    if isinstance(data, list):
        return [np.nan_to_num(np.array(d), nan=np.nan) for d in data]
    return np.nan_to_num(np.array(data), nan=np.nan)


def prepare_data(df):
    """
    Prepare the dataset with enhanced features focusing on lower price range prediction
    """
    df = df.copy()

    # Add time-based features
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek

    # Convert hour to cyclical features to capture its circular nature
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # momentum indicators
    momentum_indicators = [
        "RSI",
        "MACD",
        "MACD_hist",
        "MACD_signal",
        "EMA_short",
        "EMA_long",
        "MOM",
        "ROC",
        "momentum_score",
        "momentum_divergence",
        "EMA_9",
        "uptrend",
    ]

    # volume indicators
    volume_indicators = [
        "OBV",
        "CMF",
        "volume_surge_score",
        "volume_momentum",
        "volume_to_market_cap_ratio",
        "volume_stability",
        "volume_stability_score",
        "volume_per_pair",
        "vol_price_correlation",
    ]

    # volatility indicators
    volatility_indicators = [
        "ATR",
        "BB_width",
        "BB_%B",
        "volatility",
        "volatility_score",
        "volatility_factor",
        "price_stability",
    ]

    # price action indicators
    price_action = [
        "quote.USD.price",
        "quote.USD.percent_change_1h",
        "quote.USD.percent_change_24h",
        "quote.USD.percent_change_7d",
        "VWAP",
        "price_to_vwap",
        "discrepancy_score",
        "SMA_50",
        "SMA_200",
        "golden_cross",
    ]

    # market context indicators
    market_context = [
        "market_dominance_score",
        "liquidity_score",
        "market_impact",
        "volume_stability_score",
        "market_stability_index",
        "combined_score",
        "ADX",
    ]

    # Technical pattern indicators
    pattern_indicators = [
        "STOCH_K",
        "STOCH_D",
        "ta_price_score",
        "ta_volume_score",
    ]

    time_features = ["hour_sin", "hour_cos", "day_of_week"]

    feature_columns = (
        momentum_indicators
        + volume_indicators
        + volatility_indicators
        + price_action
        + market_context
        + pattern_indicators
        + time_features
    )

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


def plot_direction_accuracy(y_true_list, y_pred_list):
    """
    Create a line chart to show the directional accuracy of predictions for each fold
    """
    # Handle infinite values
    y_true_list = handle_infinite_values(y_true_list)
    y_pred_list = handle_infinite_values(y_pred_list)

    fig, ax = plt.subplots(figsize=(15, 6))

    for fold, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list), start=1):
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))

        # Create direction indicator (+1 for correct, -1 for incorrect)
        direction_accuracy = np.where(true_direction == pred_direction, 1, -1)

        valid_predictions = ~np.isnan(true_direction) & ~np.isnan(pred_direction)
        accuracy = (
            np.mean(
                true_direction[valid_predictions] == pred_direction[valid_predictions]
            )
            * 100
        )

        # Plot correct and incorrect predictions as lines
        ax.plot(
            range(len(direction_accuracy)),
            direction_accuracy,
            label=f"Fold {fold} - {accuracy:.1f}%",
            linewidth=2,
        )

    # Customize plot
    ax.set_title("Direction Accuracy over Time")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Percentage of Correct Direction Predictions")
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Incorrect", "Average", "Correct"])
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig


def plot_prediction_deviations(y_true_list, y_pred_list):
    """
    Create statistical plots showing prediction deviations across folds
    """
    y_true_list = handle_infinite_values(y_true_list)
    y_pred_list = handle_infinite_values(y_pred_list)

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    deviations_by_fold = [
        y_pred - y_true for y_true, y_pred in zip(y_true_list, y_pred_list)
    ]

    deviations_df = pd.DataFrame(
        {
            f"Fold {i+1}": np.nan_to_num(deviations, nan=np.nan)
            for i, deviations in enumerate(deviations_by_fold)
        }
    )

    # Box Plot
    sns.boxplot(data=deviations_df, ax=axes[0, 0])
    axes[0, 0].set_title("Distribution of Prediction Deviations by Fold")
    axes[0, 0].set_xlabel("Fold")
    axes[0, 0].set_ylabel("Deviation")

    # Violin Plot
    sns.violinplot(data=deviations_df, ax=axes[0, 1])
    axes[0, 1].set_title("Violin Plot of Prediction Deviations")
    axes[0, 1].set_xlabel("Fold")
    axes[0, 1].set_ylabel("Deviation")

    # KDE Plot
    for i, deviations in enumerate(deviations_by_fold):
        valid_deviations = deviations[~np.isnan(deviations)]
        if len(valid_deviations) > 0:
            sns.kdeplot(data=valid_deviations, ax=axes[1, 0], label=f"Fold {i+1}")
    axes[1, 0].set_title("Density Distribution of Deviations")
    axes[1, 0].set_xlabel("Deviation")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()

    # Q-Q Plot
    from scipy import stats

    for i, deviations in enumerate(deviations_by_fold):
        valid_deviations = deviations[~np.isnan(deviations)]
        if len(valid_deviations) > 0:
            stats.probplot(valid_deviations, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot of Prediction Deviations")

    plt.tight_layout()
    return fig


def plot_prediction_errors(y_true_list, y_pred_list):
    """
    Create visualization of prediction errors across folds
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Calculate errors for each fold
    errors_by_fold = []
    mean_errors = []
    median_errors = []
    std_errors = []

    for fold, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list), 1):
        errors = y_pred - y_true
        errors_by_fold.append(errors)
        mean_errors.append(np.mean(np.abs(errors)))
        median_errors.append(np.median(np.abs(errors)))
        std_errors.append(np.std(errors))

    # Plot 1: Error Distribution by Fold
    bp = ax1.boxplot(
        errors_by_fold, labels=[f"Fold {i+1}" for i in range(len(errors_by_fold))]
    )
    ax1.set_title("Distribution of Prediction Errors by Fold")
    ax1.set_ylabel("Error (Predicted - Actual)")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error Metrics by Fold
    x = range(1, len(y_true_list) + 1)
    ax2.plot(x, mean_errors, "b-", label="Mean Absolute Error", marker="o")
    ax2.plot(x, median_errors, "g-", label="Median Absolute Error", marker="s")
    ax2.plot(x, std_errors, "r-", label="Standard Deviation", marker="^")
    ax2.set_xlabel("Fold Number")
    ax2.set_ylabel("Error Magnitude")
    ax2.set_title("Error Metrics Across Folds")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    return fig, mean_errors, median_errors, std_errors


def train_and_evaluate():
    """
    Train the model using time series cross-validation with additional analysis plots
    """
    df = pd.read_csv(rf"C:/Users/{os.getenv('USER')}/Desktop/ml_training_data.csv")
    df = df[df["quote.USD.price"] < 1000]
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

        os.makedirs("plots", exist_ok=True)

        # Create and save individual fold prediction plot
        fold_fig = plot_predictions(timestamps_test, y_test, y_pred, fold)
        fold_fig.savefig(os.path.join("plots", f"fold_{fold}_predictions.png"))
        plt.close(fold_fig)

        final_X_test = X_test_scaled
        final_y_test = y_test

    # Create and save improvement visualization
    improvement_fig = plot_prediction_improvement(scores, y_true_list, y_pred_list)
    improvement_fig.savefig(os.path.join("plots", "model_improvement.png"))
    plt.close(improvement_fig)

    # Add direction accuracy plot
    direction_fig = plot_direction_accuracy(y_true_list, y_pred_list)
    direction_fig.savefig(os.path.join("plots", "direction_accuracy.png"))
    plt.close(direction_fig)

    # Add deviation analysis plots
    deviation_fig = plot_prediction_deviations(y_true_list, y_pred_list)
    deviation_fig.savefig(os.path.join("plots", "prediction_deviations.png"))
    plt.close(deviation_fig)

    # Add new prediction error analysis
    error_fig, mean_errors, median_errors, std_errors = plot_prediction_errors(
        y_true_list, y_pred_list
    )
    error_fig.savefig(os.path.join("plots", "prediction_errors.png"))
    plt.close(error_fig)

    # Print detailed error metrics for each fold
    print("\nDetailed Error Analysis by Fold:")
    print("-" * 50)
    for fold in range(len(y_true_list)):
        errors = y_pred_list[fold] - y_true_list[fold]
        print(f"\nFold {fold + 1}:")
        print(f"Mean Absolute Error: {mean_errors[fold]:.4f}")
        print(f"Median Absolute Error: {median_errors[fold]:.4f}")
        print(f"Standard Deviation of Errors: {std_errors[fold]:.4f}")
        print(f"Max Error: {np.max(np.abs(errors)):.4f}")
        print(f"Min Error: {np.min(np.abs(errors)):.4f}")

    # Get feature importance using the last fold
    feature_importance = evaluate_feature_importance(model, final_X_test, final_y_test)

    print("\nCross-validation scores:", scores)
    print("Average R² score:", np.mean(scores))
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    return model, scaler, feature_importance


if __name__ == "__main__":
    model, scaler, importance = train_and_evaluate()
