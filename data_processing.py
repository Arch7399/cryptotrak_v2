import pandas as pd


def liquidity(df):
    df["quote.USD.volume_24h"] = pd.to_numeric(
        df["quote.USD.volume_24h"], errors="coerce"
    )
    df["quote.USD.market_cap"] = pd.to_numeric(
        df["quote.USD.market_cap"], errors="coerce"
    )
    df["num_market_pairs"] = pd.to_numeric(df["num_market_pairs"], errors="coerce")
    df["volume_to_market_cap_ratio"] = (
        df["quote.USD.volume_24h"] / df["quote.USD.market_cap"]
    )
    return df


def momentum(df):
    df["momentum_score"] = df[
        [
            "quote.USD.percent_change_1h",
            "quote.USD.percent_change_24h",
            "quote.USD.percent_change_7d",
            "quote.USD.percent_change_30d",
        ]
    ].mean(axis=1)
    return df


def calculate_volatility(row):
    percent_changes = [
        row["quote.USD.percent_change_1h"],
        row["quote.USD.percent_change_24h"],
        row["quote.USD.percent_change_7d"],
    ]
    return pd.Series(percent_changes).std()


def volatility(df):
    df["volatility"] = df.apply(calculate_volatility, axis=1)
    return df


def metrics(df):
    df["circulating_to_max_supply_ratio"] = df["circulating_supply"] / df["max_supply"]
    df["tvl_ratio"] = df["quote.USD.market_cap"] / df["quote.USD.tvl"]
    return df


def vol_momentum_moving_avg(df):
    df["volume_momentum"] = df["quote.USD.volume_change_24h"]
    df["7d_moving_avg"] = df["quote.USD.price"].rolling(window=7).mean()
    return df


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def profit_potential(df):
    df["price_to_supply_ratio"] = df["quote.USD.price"] / df["circulating_supply"]
    return df


def scoring(df):
    df["combined_score"] = (
        0.5 * df["volume_momentum"]
        + 0.3 * df["quote.USD.percent_change_24h"]
        + 0.2 * df["quote.USD.percent_change_7d"]
    )
    return df


def process_data(df):
    df = df[df["quote.USD.price"] > 0]
    df = liquidity(df)
    df = momentum(df)
    df = volatility(df)
    df = metrics(df)
    df = vol_momentum_moving_avg(df)
    df = profit_potential(df)
    df["RSI"] = calculate_rsi(df["quote.USD.price"])
    df = scoring(df)
    return df
