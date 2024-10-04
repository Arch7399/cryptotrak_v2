class Config:
    # MODIFY THE THRESHOLDS ACCORDING TO YOUR PREFERENCES

    # For bullish momentum breakouts
    price_spike_threshold = p_s_t = 10  # Threshold for price spikes
    volume_surge_threshold = v_s_t = 50  # Threshold for volume surges
    price_weight = p_w = 0.6  # Weight given to price in efficiency metric
    volume_weight = v_w = 0.4  # Weight given to volume in efficiency metric

    # For overbought conditions
    overbought_price_spike_threshold = o_p_s_t = (
        5  # Threshold for price spike in overbought conditions
    )

    # For pump and dump schemes
    pump_threshold = p_t = 50  # Threshold for detecting pump schemes
    dump_threshold = d_t = -30  # Threshold for detecting dump schemes
    p_d_volume_surge_threshold = p_d_v_s_t = (
        100  # Volume surge threshold for pump and dump schemes
    )

    # For reversal opportunities
    crash_threshold = c_t = -20  # Threshold for detecting price crashes
    low_rsi_threshold = l_rsi_t = 30  # RSI threshold for low RSI

    # For false valuation / market manipulation
    lower_threshold = l_t = 10  # Lower bound for false valuation score
    upper_threshold = u_t = 50  # Upper bound for false valuation score

    file_count = 1

    # CRYPTO BUDGET RANGE SELECTION TO DISPLAY THE PROMISING CURRENCIES AT YOUR RANGE
    min_usd_price = mn_p = 0.001
    max_usd_price = mx_p = 100000

    # INITIAL EQUITY IN $
    initial_equity = 10000
