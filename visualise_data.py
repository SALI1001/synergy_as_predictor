import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# Load PID values from CSV
pid_df = pd.read_csv('pid_solana_results.csv')

# Function to fetch Solana price data for a specific date range
def get_solana_price(days):
    url = f"https://api.coingecko.com/api/v3/coins/solana/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days  # Number of days back from today
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()['prices']
    else:
        print(f"Error fetching data for Solana: {response.status_code}")
        return None

# Fetch Solana price data (last 365 days to match the PID results)
solana_data = get_solana_price(365)

# Convert Solana data to DataFrame
if solana_data:
    solana_prices = {datetime.utcfromtimestamp(item[0] / 1000).strftime('%Y-%m-%d'): item[1] for item in solana_data}
    solana_df = pd.DataFrame(list(solana_prices.items()), columns=['Date', 'Solana Price'])
    solana_df.set_index('Date', inplace=True)
else:
    print("Error: Could not fetch Solana price data.")
    exit()

# Merge PID results with Solana price data (aligning on the last day of each 14-day window)
pid_df['Date'] = solana_df.index[-len(pid_df):]  # Assign dates from the Solana data
pid_df.set_index('Date', inplace=True)

# Normalization function (min-max scaling)
def normalize(series, min_value, max_value):
    return (series - series.min()) / (series.max() - series.min()) * (max_value - min_value) + min_value

# Normalize Synergy and Redundancy to the same scale as Solana Price
solana_min, solana_max = solana_df['Solana Price'].min(), solana_df['Solana Price'].max()
pid_df['SYN_normalized'] = normalize(pid_df['SYN'], solana_min, solana_max)
pid_df['RED_normalized'] = normalize(pid_df['RED'], solana_min, solana_max)

# Plot Solana price with normalized PID values (Synergy and Redundancy)
plt.figure(figsize=(12, 6))

# Plot Solana Price
plt.plot(solana_df['Solana Price'], label='Solana Price (USD)', color='orange', alpha=0.7)

# Plot normalized Synergy and Redundancy
plt.plot(pid_df['SYN_normalized'], label='Synergy (normalized)', color='green')
#plt.plot(pid_df['RED_normalized'], label='Redundancy (normalized)', color='red')

# Customize plot
plt.title('Solana Price and Normalized PID Values (Synergy, Redundancy)')
plt.xlabel('Date')
plt.ylabel('Price / Normalized PID Value')
plt.legend()
plt.grid(True)
plt.show()

