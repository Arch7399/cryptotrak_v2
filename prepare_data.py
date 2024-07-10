import pandas as pd
import os


def prepare_raw_data(df):
    df["timestamp"] = pd.to_datetime("now")

    if not os.path.isfile(rf"C:\Users\{os.getenv('USER')}\Desktop\\CryptoAPI.csv"):
        df.to_csv(
            rf"C:\Users\{os.getenv('USER')}\Desktop\CryptoAPI.csv",
            header="column_names",
        )
    else:
        df.to_csv(
            rf"C:\Users\{os.getenv('USER')}\Desktop\CryptoAPI.csv",
            mode="a",
            header=False,
        )
