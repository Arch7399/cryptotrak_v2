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

load_dotenv()


def main():
    # Run API and save data
    for i in range(1):
        df = api_runner()
        print("API fetch ran successfully")

    df = filter_junk_currencies(df)

    prepare_raw_data(df)

    df = pd.read_csv(rf"C:/Users/{os.getenv('USER')}/Desktop/CryptoAPI.csv")
    # Read and process data
    df_processed = process_data(df)

    # Apply filters
    anomaly_results = apply_filters(df_processed)

    # Export results
    for metric, df_top in anomaly_results.items():
        if "index" in df_top.columns:
            df_top = df_top.drop(["index"], axis=1)
        df_top.to_csv(
            rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/{metric}.csv",
            mode="w",
            header=True,
            index=False,
        )
        print(f"{metric} exported!")

    # Apply tandem filters
    tandem_metrics = apply_tandem_filters(df_processed)

    # Export tandem filter results
    for metric, df_top in tandem_metrics.items():
        if "index" in df_top.columns:
            df_top = df_top.drop(["index"], axis=1)
        df_top.to_csv(
            f"C:/Users/{os.getenv('USER')}/Desktop/Analysis/Tandem/{metric}.csv",
            mode="w",
            header=True,
            index=False,
        )
        print(f"{metric} exported!")

    # Identify performing currencies
    performing_currencies = performers()

    if performing_currencies:
        recipient_emails = os.getenv("RECIPIENTS")
        send_email_alert(performing_currencies, recipient_emails)
        print(f"{performing_currencies} are performing well!")

    print("Finished")


if __name__ == "__main__":
    main()
