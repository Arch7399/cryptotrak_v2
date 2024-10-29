from dotenv import load_dotenv
import os
from time import sleep
import pandas as pd
from api_runner import api_runner
from data_processing import CryptoMetricsCalculator
from email_sender import send_email_alert
from filters import apply_filters
from mixed_filters import apply_tandem_filters
from promising_currency_pipeline import performers
from junk_currency_filter import CurrencyFilter
from prepare_data import prepare_raw_data
from append_price_difference import append_price_changes
from latest_data import filter_latest
from filters_dump import filter_dump
from prepare_ml_dataset import construct_ml_data

load_dotenv()


def main():

    # Run API and save data
    df = api_runner()
    print("API fetch ran successfully")

    filter = CurrencyFilter()
    filtered_df = filter.filter_currencies(df)

    prepare_raw_data(filtered_df)

    df = pd.read_csv(rf"C:/Users/{os.getenv('USER')}/Desktop/CryptoAPI.csv")

    latest_df = filter_latest(df)

    # Read and process data
    calculator = CryptoMetricsCalculator()
    df_processed = calculator.process_data(latest_df)

    # Apply filters
    anomaly_results = apply_filters(df_processed)
    filter_dump(anomaly_results)
    # Apply tandem filters

    tandem_metrics = apply_tandem_filters(df_processed)
    filter_dump(tandem_metrics)

    # Identify performing currencies
    performing_currencies = performers()

    if performing_currencies:
        recipient_emails = os.getenv("RECIPIENTS")
        send_email_alert(performing_currencies, recipient_emails)
        print(f"{performing_currencies} are performing well!")

    try:
        append_price_changes(df)
        construct_ml_data()
        print(
            rf"ML training data constructed at: C:/Users/{os.getenv('USER')}/Desktop/ml_training_data.csv"
        )
    except:
        print(
            f"Not enough data points to construct ML model, run the code atleast twice"
        )

    print("Finished")


if __name__ == "__main__":
    main()
