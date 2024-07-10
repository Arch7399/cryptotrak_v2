# Cryptocurrency Analyzer

This project is an upgraded cryptocurrency analysis framework designed to identify promising currencies using multiple filters like price momentum, volume surges, liquidity, RSI, pump-and-dump detection, and market cap-to-volume discrepancies. The framework includes additional features such as **sensitivity analysis**, **historical data analysis**.


## Features

- **Customizable Filters**:
  - Detects **price spikes** and **volume surges** using statistical Z-scores.
  - Identifies **bullish momentum breakouts** and **reversal opportunities**.
  - Flags **price crashes**, **pump-and-dump schemes**, and **market cap-volume discrepancies**.
  - Filters for **low liquidity** and **false valuations**.

- **Advanced Technical Indicators**:
  - Calculates technical indicators such as **Simple Moving Averages (SMA)** (50 and 200 periods), **Exponential Moving Average (EMA)** (20-period), and trend-based signals like **uptrends** and **golden crosses**.
  - Detects **market dominance** based on market cap and **stability** of volume and price using rolling statistics.

- **Composite Scoring System**:
  - A robust **promise score** that combines market dominance, price and volume stability, RSI, trend indicators, and a flag-based severity system for positive and negative signals.
  - Custom weighting for each factor influencing the promise score, including **RSI deviation**, **uptrend indicators**, **z-score** and **golden crosses**.
  - Penalizes currencies with an excess of negative flags (e.g., too many pump flags or crashes), while rewarding those with no negative flags.

- **Time Decay Weighting**:
  - The model incorporates **time decay** to weigh more recent data more heavily, ensuring that currencies showing recent promising behavior are prioritized.

- **Normalization**:
  - Applies **MinMaxScaler** normalization to features like market dominance, price and volume stability, and RSI for more accurate scoring and comparison.

- **Flag Severity Calculation**:
  - The system calculates a **flag severity score**, quantifying how many positive vs. negative flags a currency has. This directly impacts the final score, ensuring balanced analysis between positive and negative signals.

- **CSV Data Export**:
  - Outputs the top promising currencies to a CSV file for further analysis, along with all relevant metrics like market dominance, price, volume, supply, and flag data.
  - Provides an easy-to-read ranked list of the top 5 performing currencies based on the promise score.

- **Extensible and Modular**:
  - Designed to be easily extensible. You can add new filters, indicators, or modify the scoring algorithm with minimal changes to the code.


## Further Ongoing Upgrades

- **Backtesting**:
  - The backtesting framework to evaluate how different threshold combinations would perform using historical data, helping to understand the effectiveness of filters over time.

- **Machine Learning Optimization**:
  - Using techniques like grid search or genetic algorithms, to optimize the thresholds based on historical performance.

- **UI Development**:
  - For better visualization of the analysis results.


## Main Components

- `config.py`: 
  - Stores all threshold values and allows customizable parameters for filters (e.g., `price_spike_threshold`, `volume_surge_threshold`).
  - Shortened alias names (e.g., `p_s_t`, `v_s_t`) for easier reference when passing variables into functions.
  
- `promising_currency_pipeline.py`:
  - The core module responsible for applying filters to the cryptocurrency data and scoring each currency based on defined metrics.

- `identify_promising_currencies.py`:
  - Identifies and scores promising cryptocurrencies based on price momentum, volume momentum, RSI, and other key metrics.

## Key Methods

- `sensitivity_analysis()`: Evaluates how different threshold values perform over a range of values and selects the best performing one.

- `identify_promising_currencies()`: Processes raw data, applies filters, and identifies top-performing cryptocurrencies.

## Usage

1. **Clone the repository**:
   ```sh
    git clone https://github.com/Arch7399/cryptotrak.git
    cd cryptotrak
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Configuration**:
    Create and Update your `.env` file with the following options:
    
    - **API_KEY**: Your API key for fetching cryptocurrency data.
    - **SENDER_EMAIL**: The email address from which notifications will be sent.
    - **PASSWORD**: Sender mail password
    - **USER**: User name for .csv path file
    - **RECIPIENT_EMAIL**: The email address to receive notifications.
    
    Example `.env`:
    
    ```python
    API_KEY=vvvvvvvv-wwww-xxxx-yyyy-zzzzzzzzzzzz
    SENDER_EMAIL=johny@gmail.com
    USER=ricky
    RECIPIENT_EMAIL=sam@gmail.com
    ```

4. **Create file paths**:
    Create an "Analysis/Tandem" folder structure in your desktop for analysis dump files from output. Or any other location of your choice provided you modify the associated paths in depending .py files.

4. **Run the application**:
    ```sh
    python main.py
    ```

## 🤝 Contributing

Contributions are welcome for further development of this project! Follow these steps to contribute:
    
1. **Fork the repository on GitHub.**
2. **Clone the forked repository to your local machine.**
   ```sh
   git clone https://github.com/Arch7399/cryptotrak.git
   cd cryptotrak
   git checkout -b feature/your-feature
   git commit -am 'Add some feature'
   git push origin feature/your-feature
3. Submit a pull request on GitHub