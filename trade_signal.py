import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import multiprocessing

def GetPosition(signal, prcLevel):
    position = [signal[0]]
    position_before = 0

    for i in range(1, len(signal)):
        prev_position = position[-1]
        current_signal = signal[i]

        # Open position if signal and prior position is small, and still within backtest period
        if current_signal == 1 and abs(position_before) < 1 and i / len(signal) < 0.9:
            position.append(1 + position_before)
        elif current_signal == -2 and abs(position_before) < 1 and i / len(signal) < 0.9:
            position.append(-1 + position_before)
        # Close position signals
        elif current_signal in [-1, 2, 3, -3]:
            position.append(0)
        else:
            position.append(prev_position)

        position_before = position[-1]

    position = pd.Series(position, index=prcLevel.index)
    position.iloc[-1] = 0  # Ensure position is closed at the end
    return position

# Define trading signal logic
def TradeSig(prcLevel):
    signal = np.zeros(len(prcLevel), dtype=np.int16)
    for i in range(1, len(prcLevel)):
        if prcLevel.iloc[i-1] == 1 and prcLevel.iloc[i] == 2:
            signal[i] = -2  # Open short
        elif prcLevel.iloc[i-1] == 0 and prcLevel.iloc[i] == 1:
            signal[i] = 2   # Close short
        elif prcLevel.iloc[i-1] == 2 and prcLevel.iloc[i] == 3:
            signal[i] = 3   # Stop loss on short
        elif prcLevel.iloc[i-1] == -1 and prcLevel.iloc[i] == -2:
            signal[i] = 1   # Open long
        elif prcLevel.iloc[i-1] == 0 and prcLevel.iloc[i] == -1:
            signal[i] = -1  # Close long
        elif prcLevel.iloc[i-1] == -2 and prcLevel.iloc[i] == -3:
            signal[i] = -3  # Stop loss on long
    return signal

# Simulate trading account and returns
def TradeSimSSD_append(priceX, priceY, position, nameX, nameY, TestSpread, level):
    n = len(position)
    size = 100000 // priceY.iloc[0]  # Fixed initial capital
    shareY = size * position
    shareX = [-shareY.iloc[0] * priceY.iloc[0] // priceX.iloc[0]]
    cash = [20000]

    for i in range(1, n):
        shareX.append(shareX[i-1])
        cash.append(cash[i-1])
        if position.iloc[i] - position.iloc[i-1] == 1 and position.iloc[i] != 0:
            shareX[i] = -shareY.iloc[i] * priceY.iloc[i] // priceX.iloc[i]
            cash[i] = cash[i-1] - ((shareY.iloc[i] - shareY.iloc[i-1]) * priceY.iloc[i] + 
                                   (shareX[i] - shareX[i-1]) * priceX.iloc[i])
        elif position.iloc[i] - position.iloc[i-1] == -1 and position.iloc[i] != 0:
            shareX[i] = -shareY.iloc[i] * priceY.iloc[i] // priceX.iloc[i]
            cash[i] = cash[i-1] - ((shareY.iloc[i] - shareY.iloc[i-1]) * priceY.iloc[i] + 
                                   (shareX[i] - shareX[i-1]) * priceX.iloc[i])
        elif position.iloc[i-1] > 0 and position.iloc[i] == 0:
            shareX[i] = 0
            cash[i] = cash[i-1] + (shareY.iloc[i-1] * priceY.iloc[i] + shareX[i-1] * priceX.iloc[i])
        elif position.iloc[i-1] < 0 and position.iloc[i] == 0:
            shareX[i] = 0
            cash[i] = cash[i-1] + (shareY.iloc[i-1] * priceY.iloc[i] + shareX[i-1] * priceX.iloc[i])

    cash = pd.Series(cash, index=position.index)
    shareY = pd.Series(shareY, index=position.index)
    shareX = pd.Series(shareX, index=position.index)
    stockvalue = shareY * priceY + shareX * priceX
    asset = cash + stockvalue
    account = pd.DataFrame({
        'Position': position,
        'ShareY': shareY,
        'PriceY': priceY,
        'ShareX': shareX,
        'PriceX': priceX,
        'StockValue': stockvalue,
        'Cash': cash,
        'Asset': asset,
        'Spread': TestSpread
    })

    # Add constant thresholds
    account['OPEN'] = level[5]
    account['CLOSE'] = level[4]
    account['STOP'] = level[6]

    return account

# Parallel backtest processor
def process_pair(pair, train_set, test_set, parimate):
    X1_code, X2_code = pair
    X1_train = train_set.loc[train_set.index.get_level_values('code') == X1_code, 'factor'].astype(float)
    X2_train = train_set.loc[train_set.index.get_level_values('code') == X2_code, 'factor'].astype(float)
    X1_train = X1_train.reset_index(level='code', drop=True)
    X2_train = X2_train.reset_index(level='code', drop=True)

    X1_test = test_set.loc[test_set.index.get_level_values('code') == X1_code, 'factor'].astype(float)
    X2_test = test_set.loc[test_set.index.get_level_values('code') == X2_code, 'factor'].astype(float)
    X1_test = X1_test.reset_index(level='code', drop=True)
    X2_test = X2_test.reset_index(level='code', drop=True)

    X1_price = test_set.loc[test_set.index.get_level_values('code') == X1_code, 'open'].astype(float)
    X2_price = test_set.loc[test_set.index.get_level_values('code') == X2_code, 'open'].astype(float)
    X1_price = X1_price.reset_index(level='code', drop=True)
    X2_price = X2_price.reset_index(level='code', drop=True)

    results = sm.OLS(X2_train, X1_train).fit()
    b = results.params['factor']

    TradSpread = X2_train - b * X1_train
    mean_pair = np.mean(TradSpread)
    sd_pair = np.std(TradSpread)

    close_bar, open_bar, stop_bar = parimate

    thresholdUpOpen = mean_pair + open_bar * sd_pair
    thresholdDownOpen = mean_pair - open_bar * sd_pair
    thresholdUpClose = mean_pair + close_bar * sd_pair
    thresholdDownClose = mean_pair - close_bar * sd_pair
    thresholdUpStopLoss = mean_pair + stop_bar * sd_pair
    thresholdDownStopLoss = mean_pair - stop_bar * sd_pair

    level = (
        float('-inf'),
        thresholdDownStopLoss,
        thresholdDownOpen,
        thresholdDownClose,
        thresholdUpClose,
        thresholdUpOpen,
        thresholdUpStopLoss,
        float('inf')
    )

    TestSpread = X2_test - b * X1_test
    prcLevel = pd.cut(TestSpread, level, labels=False) - 3
    signal = TradeSig(prcLevel)
    position = GetPosition(signal, prcLevel)

    return TradeSimSSD_append(X1_price, X2_price, position, X1_code, X2_code, TestSpread, level)

# Generate account dictionary from all pairs
def generate_account_dict(train_set, test_set, pairs, parimate):
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    results = []
    for pair in pairs:
        result = pool.apply_async(process_pair, (pair, train_set, test_set, parimate))
        results.append(result)

    pool.close()
    pool.join()

    account_dict = {}
    for i, result in enumerate(results):
        pair = tuple(pairs[i])
        account_dict[pair] = result.get()

    return account_dict

def generate_account_dict_for_dates(calendar, test_window, train_window, data,parimate,location):
    account_dict = dict()

    for rw in tqdm(range(len(calendar) // test_window - train_window//test_window)):
        date = str(calendar[rw * test_window + train_window])[:10]

        train_set = data.loc[str(calendar[rw * test_window]):str(calendar[rw * test_window + train_window - 1])]
        test_set = data.loc[str(calendar[rw*test_window+train_window]):str(calendar[(rw+1)*test_window+train_window-1])]

        pairs = np.load(location+date+'.npy', allow_pickle=True)

        account_dict[date] = generate_account_dict(train_set, test_set, pairs, parimate)

    return account_dict

def generate_positions_df(calendar, test_window, train_window, account_dict, code, location):
    positions_dict = {}
    dfs = []

    for rw in tqdm(range(len(calendar) // test_window - train_window//test_window)):
        #print(calendar[rw * test_window + train_window])
        date = str(calendar[rw * test_window + train_window])[:10]
        account_dict_daily = account_dict[date]
        positions_dict[date] = {i: 0 for i in code}
        for stock_pair, account_df in account_dict_daily.items():
            for idx, stock in enumerate(stock_pair):
                positions_dict[date][stock] += account_df[['ShareX', 'ShareY'][idx]]

        pd.DataFrame(positions_dict[date]).to_csv(location+date+'.csv')
        dfs.append(pd.DataFrame(positions_dict[date]))

    positions_df = pd.concat(dfs)
    return positions_df





