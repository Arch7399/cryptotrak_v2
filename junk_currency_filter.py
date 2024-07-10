import pandas as pd


def filter_junk_currencies(df):
    # Define junk criteria
    min_market_cap = 1000000  # $1 million
    min_volume_24h = 10000  # $10,000
    min_age_days = 30
    max_circulating_supply_ratio = 0.9  # 90% of max supply

    # Convert columns to numeric, replacing errors with NaN
    numeric_columns = [
        "quote.USD.market_cap",
        "quote.USD.volume_24h",
        "circulating_supply",
        "max_supply",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate age of currencies
    df["age_days"] = (
        pd.to_datetime(df["last_updated"]) - pd.to_datetime(df["date_added"])
    ).dt.days

    # Apply filters
    df_filtered = df[
        (df["quote.USD.market_cap"] > 0)  # Explicitly filter out zero market cap
        & (df["quote.USD.market_cap"] >= min_market_cap)
        & (df["quote.USD.volume_24h"] >= min_volume_24h)
        & (df["age_days"] >= min_age_days)
        & (
            (
                (df["max_supply"] > 0)
                & (
                    (
                        df["circulating_supply"] / df["max_supply"]
                        <= max_circulating_supply_ratio
                    )  # Still room for issuance
                    | (df["circulating_supply"] / df["max_supply"] >= 0.9)
                )
            )  # Near max supply, indicating scarcity
            |
            # If no max supply, ignore this criterion
            (df["max_supply"].isnull())
            | (df["max_supply"] == 0)
        )
    ]

    # Log the filtering process
    total_currencies = len(df)
    filtered_currencies = len(df_filtered)
    junk_currencies = total_currencies - filtered_currencies

    print(f"Total currencies: {total_currencies}")
    print(f"Currencies after filtering: {filtered_currencies}")
    print(
        f"Filtered out {junk_currencies} junk currencies ({junk_currencies/total_currencies:.2%})"
    )

    # Additional logging for specific criteria
    zero_market_cap = (df["quote.USD.market_cap"] == 0).sum()
    low_market_cap = (
        (df["quote.USD.market_cap"] > 0) & (df["quote.USD.market_cap"] < min_market_cap)
    ).sum()
    low_volume = (df["quote.USD.volume_24h"] < min_volume_24h).sum()
    young_currencies = (df["age_days"] < min_age_days).sum()

    print(f"\nDetailed filtering results:")
    print(f"Currencies with zero market cap: {zero_market_cap}")
    print(f"Currencies with market cap below ${min_market_cap:,}: {low_market_cap}")
    print(f"Currencies with 24h volume below ${min_volume_24h:,}: {low_volume}")
    print(f"Currencies younger than {min_age_days} days: {young_currencies}")

    return df_filtered
