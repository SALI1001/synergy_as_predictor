import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# Load the PID values from the CSV file
pid_df = pd.read_csv('pid_solana_results.csv')

# Ensure the columns we need are available
print(pid_df.columns)  # Check which columns are present (e.g., 'SYN', 'RED', etc.)

# Function to fetch Solana price data from CoinGecko
def get_solana_price(days):
    url = f"https://api.coingecko.com/api/v3/coins/solana/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days  # Number of days back from today (should match the PID data length)
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()['prices']
    else:
        print(f"Error fetching data for Solana: {response.status_code}")
        return None

# Fetch Solana price data for 365 days to match the PID data
solana_data = get_solana_price(365)

# Convert Solana data to a DataFrame if successful
if solana_data:
    solana_prices = {datetime.utcfromtimestamp(item[0] / 1000).strftime('%Y-%m-%d'): item[1] for item in solana_data}
    solana_df = pd.DataFrame(list(solana_prices.items()), columns=['Date', 'Solana Price'])
    solana_df.set_index('Date', inplace=True)
    
    # Align the fetched Solana price data with PID data (assuming same length)
    pid_df['Date'] = solana_df.index[:len(pid_df)]  # Assign corresponding dates to the PID data
    pid_df['Solana Price'] = solana_df['Solana Price'].values[:len(pid_df)]  # Assign Solana prices
    pid_df.set_index('Date', inplace=True)
else:
    print("Error: Could not fetch Solana price data.")
    exit()

def detect_syn_spikes(syn_series, window_size=5, threshold_factor=2):
    """
    Detects isolated Synergy spikes in the data.
    
    :param syn_series: Pandas series of Synergy values
    :param window_size: Number of previous and next points to consider for identifying isolated spikes
    :param threshold_factor: Factor by which a peak must be greater than its surroundings to be considered a spike
    :return: List of indices where Synergy spikes occur
    """
    spikes = []
    for i in range(window_size, len(syn_series) - window_size):
        window = syn_series.iloc[i - window_size:i + window_size + 1]
        current_value = syn_series.iloc[i]

        # Compare current value with surrounding values, excluding the current position
        surrounding_values = window.drop(window.index[window_size])  # Drop the middle value, which is current_value
        
        # Check if current value is a peak and significantly larger than the surrounding values
        if current_value == max(window) and current_value > threshold_factor * max(surrounding_values):
            # Ensure there is a resting period before the next spike
            after_window = syn_series.iloc[i + 1:i + window_size + 1]
            if all(after_window < current_value / threshold_factor):
                spikes.append(i)
    return spikes



# Define a function to detect dramatic price movements
def detect_dramatic_price_changes(price_series, window_size=5, price_change_threshold=0.05):
    """
    Detects dramatic price movements (up or down).
    
    :param price_series: Pandas series of price values
    :param window_size: Number of days to check for price movement
    :param price_change_threshold: Percentage change threshold for dramatic movements
    :return: List of tuples with start and end dates of dramatic price movements
    """
    dramatic_changes = []
    for i in range(len(price_series) - window_size):
        price_change = (price_series[i + window_size] - price_series[i]) / price_series[i]
        if abs(price_change) > price_change_threshold:
            dramatic_changes.append((i, price_change))  # Store index and percentage change
    return dramatic_changes

# Detect Synergy spikes
syn_spikes = detect_syn_spikes(pid_df['SYN'])

# Detect dramatic Solana price changes
price_changes = detect_dramatic_price_changes(pid_df['Solana Price'])

# Function to check if any Syn spikes precede dramatic price changes
def check_spikes_and_price_movements(syn_spikes, price_changes, time_window=5):
    """
    Check if Synergy spikes occur before or just as Solana price moves dramatically.
    
    :param syn_spikes: List of Synergy spike indices
    :param price_changes: List of price movement tuples (index, percentage change)
    :param time_window: Number of days to consider as the valid window for a Syn spike to precede a price change
    :return: List of Syn spike indices that precede or coincide with price movements
    """
    matches = []
    for spike_idx in syn_spikes:
        for change_idx, change_percent in price_changes:
            if spike_idx <= change_idx <= spike_idx + time_window:
                matches.append((spike_idx, change_idx, change_percent))
    return matches

# Check for correlations between Synergy spikes and price changes
spike_matches = check_spikes_and_price_movements(syn_spikes, price_changes)

# Report the results
print(f"Total Synergy Spikes Detected: {len(syn_spikes)}")
print(f"Total Dramatic Price Movements Detected: {len(price_changes)}")
print(f"Synergy Spikes Preceding Price Movements: {len(spike_matches)}")

# Print details of each match
for match in spike_matches:
    spike_idx, change_idx, price_change = match
    print(f"Syn Spike at {pid_df.index[spike_idx]} | Price Change: {price_change*100:.2f}% at {pid_df.index[change_idx]}")

# Plot the results
plt.figure(figsize=(12, 6))

# Plot Solana Price
plt.plot(pid_df['Solana Price'], label='Solana Price (USD)', color='orange', alpha=0.7)

# Plot Synergy with spikes marked
plt.plot(pid_df['SYN'], label='Synergy', color='green')
plt.scatter(pid_df.index[syn_spikes], pid_df['SYN'].iloc[syn_spikes], color='red', label='Syn Spikes', marker='o')

# Customize plot
plt.title('Solana Price and Synergy with Spike Detection')
plt.xlabel('Date')
plt.ylabel('Price / Synergy Value')
plt.legend()
plt.grid(True)
plt.show()



