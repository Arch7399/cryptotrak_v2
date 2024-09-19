from dotenv import load_dotenv
import os
from time import sleep
import pandas as pd
from api_runner import api_runner
from data_processing import process_data
from email_sender import send_email_alert
from filters import apply_filters
from mixed_filters import apply_tandem_filters
from promising_currency_pipeline import performers
from junk_currency_filter import filter_junk_currencies
from prepare_data import prepare_raw_data
from append_price_difference import append_price_changes
from latest_data import filter_latest
from filters_dump import filter_dump
from enhanced_crypto_pipeline import integrate_with_main

load_dotenv()


def main():

    # Run API and save data
    df = api_runner()
    print("API fetch ran successfully")

    df = filter_junk_currencies(df)

    prepare_raw_data(df)

    df = pd.read_csv(rf"C:/Users/{os.getenv('USER')}/Desktop/CryptoAPI.csv")

    latest_df = filter_latest(df)

    # Read and process data
    df_processed = process_data(latest_df)

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

    append_price_changes(df)

    print("Finished")


if __name__ == "__main__":
    main()
