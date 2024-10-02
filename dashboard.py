import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Title of the dashboard
st.title("Cryptocurrency Analysis Dashboard")

# Load and display price data (replace path as needed)
df = pd.read_csv(
    r"C:/Users/{}/Desktop/Analysis/PromisingCurrencies.csv".format(os.getenv("USER"))
)

st.header("Top Performing Currencies")
mdf = df[df["timestamp"] == max(df["timestamp"])]
top_currencies = mdf["name"].head().to_list()
st.write(top_currencies)

# Compute the mean price for each timestamp
df_mean_price = df.groupby("timestamp")["volatility_score"].mean().reset_index()
df_mean_price.columns = ["timestamp", "volatility_score"]

# Line chart for mean price trend visualization
st.header("Volatility")
fig = px.line(
    df_mean_price,
    x="timestamp",
    y="volatility_score",
    title="Market volatility Over Time",
)
st.plotly_chart(fig)

# Display data table for further analysis
st.header("Predicted Cryptos")
mdf = mdf.drop(columns=["Unnamed: 0"], errors="ignore")
mdf = mdf.reset_index(drop=True)

st.dataframe(mdf)
