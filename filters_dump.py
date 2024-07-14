import os


# Export results
def filter_dump(metrics):
    for metric, df_top in metrics.items():
        if "index" in df_top.columns:
            df_top = df_top.drop(["index"], axis=1)
        # df_top.to_csv(
        #     rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/{metric}.csv",
        #     mode="w",
        #     header=True,
        #     index=False,
        # )
        if not os.path.isfile(
            rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/{metric}.csv"
        ):
            df_top.to_csv(
                rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/{metric}.csv",
                header="column_names",
            )
        else:
            df_top.to_csv(
                rf"C:/Users/{os.getenv('USER')}/Desktop/Analysis/{metric}.csv",
                mode="a",
                header=False,
            )
        print(f"{metric} exported!")
