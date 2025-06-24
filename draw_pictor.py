import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def plot_asset_value_over_time(AllAsset):
    # Set seaborn style
    sns.set(style="whitegrid")

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot asset value line chart
    plt.plot(AllAsset.index, AllAsset.values, color='skyblue', linestyle='-', marker='o', markersize=3)

    # Add title and axis labels
    plt.title('Asset Value Over Time', fontsize=16)
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Asset Value', fontsize=14)

    # Show legend
    plt.legend(['Asset Value'], loc='upper left', fontsize=12)

    # Show plot
    plt.show()

def plot_assets_and_cash(assets_dict, cash_dict, name):
    # Convert dictionaries to DataFrames for plotting
    assets_df = pd.DataFrame.from_dict(assets_dict, orient='index', columns=['Assets'])
    cash_df = pd.DataFrame.from_dict(cash_dict, orient='index', columns=['Cash'])

    # Plot assets and cash over time
    plt.figure(figsize=(12, 6))
    plt.plot(assets_df.index, assets_df['Assets'], label='Assets')
    plt.plot(cash_df.index, cash_df['Cash'], label='Cash')
    plt.title('Assets and Cash Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig(name + '.png')

def plot_heatmap_from_files(location):
    # Get all file names in the folder
    file_names = os.listdir(location)
    
    # Filter all .npy files
    npy_files = [file for file in file_names if file.endswith('.npy')]

    stock_pair_num = dict()

    for npy_file in npy_files:
        pairs = np.load(os.path.join(location, npy_file), allow_pickle=True)

        for stock_pair in pairs:
            stock_pair_tuple = tuple(stock_pair)  # Convert numpy array to tuple
            if stock_pair_tuple in stock_pair_num:
                stock_pair_num[stock_pair_tuple] += 1
            else:
                stock_pair_num[stock_pair_tuple] = 1

    # Create DataFrame from frequency dictionary
    df = pd.DataFrame(stock_pair_num.values(), index=pd.MultiIndex.from_tuples(stock_pair_num.keys()), columns=['Value'])

    # Reshape for heatmap
    heatmap_data = df.unstack().fillna(0)['Value']
    location_parts = location.split('/')
    image_name = location_parts[-3] + ' ' + location_parts[-2]  # Use folder name as image title

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=False, cmap='YlGnBu', square=True)
    plt.title('Frequency of ' + image_name)
    plt.xlabel('Stock')
    plt.ylabel('Stock')
    plt.show()

def plot_trading_signals(account_dict):
    # Create 'position_pic' directory
    os.makedirs('position_pic', exist_ok=True)

    for date in account_dict.keys():
        # Create subfolder under 'position' for each date
        date_folder = os.path.join('position', date)
        os.makedirs(date_folder, exist_ok=True)

        for stock_pair in account_dict[date].keys():
            df = account_dict[date][stock_pair]

            # Create figure and axis
            fig, ax = plt.subplots()

            # Plot Spread
            ax.plot(df['Spread'], label='Spread', color='blue', linestyle='-')

            # Plot Position
            ax.plot(df['Position'], label='Position', color='green', linestyle='-', alpha=0.5)

            # Plot CLOSE lines
            ax.plot(df['CLOSE'], label='CLOSE', color='red', linestyle='--')
            ax.plot(-1 * df['CLOSE'], label='CLOSE', color='red', linestyle='--')

            # Plot OPEN lines
            ax.plot(df['OPEN'], label='OPEN', color='orange', linestyle='--')
            ax.plot(-1 * df['OPEN'], label='OPEN', color='orange', linestyle='--')

            # Plot STOP lines
            ax.plot(df['STOP'], label='STOP', color='purple', linestyle='--')
            ax.plot(-1 * df['STOP'], label='STOP', color='purple', linestyle='--')

            # Add legend
            ax.legend()

            # Add title
            ax.set_title('Trading Signals')

            # Show grid
            ax.grid(True)

            # Save figure to the corresponding date folder
            plt.savefig(os.path.join(date_folder, f'{stock_pair}.png'))

            # Close plot for next iteration
            plt.close()



