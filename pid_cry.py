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
        'days': days  # Number of days back from today (up to 1825 days for 5 years)
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

# Fetch historical data for Bitcoin, Ethereum, and Solana (last 5 years = 1825 days)
bitcoin_data = get_historical_data('bitcoin', 1825)
ethereum_data = get_historical_data('ethereum', 1825)
solana_data = get_historical_data('solana', 1825)

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

# Modify the PID computation function to include dates
def compute_pid_over_windows(df):
    pid_results = []
    dates = []
    
    # Iterate through the dataframe in 14-day chunks
    for i in range(0, len(df) - 13):  # Ensure we only process full 14-day windows
        # Get the last 14 days of data for Bitcoin, Ethereum, and Solana
        bitcoin_chunk = df['Bitcoin'].iloc[i:i+14].values
        ethereum_chunk = df['Ethereum'].iloc[i:i+14].values
        solana_chunk = df['Solana'].iloc[i:i+14].values
        window_end_date = df.index[i + 13]  # Use the last day of the 14-day window
        
        # Compute the PID values using your function
        RED, SYN, UN_x, UN_y = inf_funcs.pidMMI(bitcoin_chunk, ethereum_chunk, solana_chunk, _local=False)
        
        # Append the results and the date to a list
        pid_results.append([RED, SYN, UN_x, UN_y])
        dates.append(window_end_date)
    
    # Create a DataFrame to store the PID results with corresponding dates
    pid_df = pd.DataFrame(pid_results, columns=['RED', 'SYN', 'UN_x', 'UN_y'], index=dates)
    
    return pid_df

# Compute PID for 14-day windows
pid_df = compute_pid_over_windows(binary_df)

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

# Plot the smoothed PID values along with Solana price deviations (fluctuations)
plt.figure(figsize=(10, 6))

# Plot Solana price deviations from 1
plt.plot(df.index, df['Solana'], label='Solana Price (USD)', color='orange', alpha=0.5)

# Plot smoothed PID values
plt.plot(pid_df.index, pid_df['RED_smooth'], label='RED (smoothed)', color='red')
plt.plot(pid_df.index, pid_df['SYN_smooth'], label='SYN (smoothed)', color='green')
plt.plot(pid_df.index, pid_df['UN_x_smooth'], label='UN_x (smoothed)', color='blue')
plt.plot(pid_df.index, pid_df['UN_y_smooth'], label='UN_y (smoothed)', color='purple')

# Customize plot
plt.title('Smoothed PID Values and Solana Price Over Time (Last 5 Years)')
plt.xlabel('Date')
plt.ylabel('PID Values / Solana Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

