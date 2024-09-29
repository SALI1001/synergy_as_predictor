import requests
import pandas as pd
from datetime import datetime
import inf_funcs
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Function to fetch historical data from CoinGecko
def get_historical_data(coin, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days  # Number of days back from today (e.g., 365 days)
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()['prices']
    else:
        print(f"Error fetching data for {coin}: {response.status_code}")
        return None

# Function to convert raw API data to a pandas DataFrame
def convert_to_dataframe(bitcoin_data, ethereum_data, solana_data):
    # Extract the timestamps and prices
    bitcoin_prices = {datetime.utcfromtimestamp(item[0] / 1000).strftime('%Y-%m-%d'): item[1] for item in bitcoin_data}
    ethereum_prices = {datetime.utcfromtimestamp(item[0] / 1000).strftime('%Y-%m-%d'): item[1] for item in ethereum_data}
    solana_prices = {datetime.utcfromtimestamp(item[0] / 1000).strftime('%Y-%m-%d'): item[1] for item in solana_data}
    
    # Combine into a DataFrame
    df = pd.DataFrame({
        'Date': list(bitcoin_prices.keys()),
        'Bitcoin': list(bitcoin_prices.values()),
        'Ethereum': list(ethereum_prices.values()),
        'Solana': list(solana_prices.values())
    })
    
    # Set the 'Date' column as the index
    df.set_index('Date', inplace=True)
    
    return df

# Fetch historical data for Bitcoin, Ethereum, and Solana (last 365 days)
bitcoin_data = get_historical_data('bitcoin', 365)
ethereum_data = get_historical_data('ethereum', 365)
solana_data = get_historical_data('solana', 365)

# Convert the fetched data into a DataFrame
if bitcoin_data and ethereum_data and solana_data:
    df = convert_to_dataframe(bitcoin_data, ethereum_data, solana_data)
    print(df)
else:
    print("Error fetching data for one or more coins")

# Function to transform prices into 1s (up/stays the same) and 0s (down)
def transform_to_binary(df):
    transformed_df = df.copy()
    
    # Iterate over each column (Bitcoin, Ethereum, Solana)
    for column in df.columns:
        # Create a new column with 1s and 0s based on price movement
        transformed_df[column] = (df[column].diff() >= 0).astype(int)
    
    return transformed_df

# Transform the DataFrame to 1s and 0s based on price movement
binary_df = transform_to_binary(df)
print(binary_df)

def compute_pid_over_windows(df):
    pid_results = []
    
    # Iterate through the dataframe in 14-day chunks
    for i in range(0, len(df) - 13):  # Ensure we only process full 14-day windows
        # Get the last 14 days of data for Bitcoin, Ethereum, and Solana
        bitcoin_chunk = df['Bitcoin'].iloc[i:i+14].values
        ethereum_chunk = df['Ethereum'].iloc[i:i+14].values
        solana_chunk = df['Solana'].iloc[i:i+14].values
        
        # Compute the PID values using your function
        RED, SYN, UN_x, UN_y = inf_funcs.pidMMI(bitcoin_chunk, ethereum_chunk, solana_chunk, _local=False)
        
        # Append the results to a list
        pid_results.append([RED, SYN, UN_x, UN_y])
    
    # Create a DataFrame to store the PID results
    pid_df = pd.DataFrame(pid_results, columns=['RED', 'SYN', 'UN_x', 'UN_y'])
    
    return pid_df

# Compute PID for 14-day windows
pid_df = compute_pid_over_windows(binary_df)

# Save PID results to a CSV
pid_df.to_csv('pid_solana_results.csv', index=False)

# Apply Savitzky-Golay filter to smooth each column
def smooth_column(data, window_size, poly_order):
    return savgol_filter(data, window_size, poly_order)

# Set window size and polynomial order for smoothing (adjust as needed)
window_size = 5  # This should be odd and less than the number of rows in pid_df
poly_order = 2   # Degree of the polynomial used for smoothing

# Apply smoothing to each column
pid_df['RED_smooth'] = smooth_column(pid_df['RED'], window_size, poly_order)
pid_df['SYN_smooth'] = smooth_column(pid_df['SYN'], window_size, poly_order)
pid_df['UN_x_smooth'] = smooth_column(pid_df['UN_x'], window_size, poly_order)
pid_df['UN_y_smooth'] = smooth_column(pid_df['UN_y'], window_size, poly_order)

# Plot the smoothed values
plt.figure(figsize=(10, 6))
plt.plot(pid_df['RED_smooth'], label='RED (smoothed)', color='red')
plt.plot(pid_df['SYN_smooth'], label='SYN (smoothed)', color='green')
plt.plot(pid_df['UN_x_smooth'], label='UN_x (smoothed)', color='blue')
plt.plot(pid_df['UN_y_smooth'], label='UN_y (smoothed)', color='purple')

plt.title('Smoothed PID Values Over Time')
plt.xlabel('Time (14-day windows)')
plt.ylabel('PID Values')
plt.legend()
plt.grid(True)
plt.show()
