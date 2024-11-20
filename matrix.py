import pandas as pd
from tqdm import tqdm
import numpy as np

def calculate_assets(positions_df, price_df, trading_cost_rate=0.0015, initial_cash=100000):
    pre_date = 0
    cash = {}
    assets = {}
    cash_with_cost = {}
    assets_with_cost = {}

    for date in positions_df.index:
        if pre_date == 0:
            cash[date] = initial_cash
            cash_with_cost[date] = initial_cash
        else:
            cash[date] = cash[pre_date]
            cash_with_cost[date] = cash_with_cost[pre_date]

        cash_net = 0
        assets[date] = 0
        
        current_trading_cost = 0
        for stock in positions_df.columns:
            if pre_date == 0:
                cash_net -= positions_df.loc[date, stock] * price_df.loc[date, stock]
            else:
                cash_net -= (positions_df.loc[date, stock] - positions_df.loc[pre_date, stock]) * price_df.loc[date, stock]

                current_trading_cost += abs(positions_df.loc[date, stock] - positions_df.loc[pre_date, stock]) * price_df.loc[date, stock] * trading_cost_rate
            assets[date] += positions_df.loc[date, stock] * price_df.loc[date, stock]

            # if pre_date != 0:
            #     opening = abs(positions_df.loc[date, stock] - positions_df.loc[pre_date, stock].shift(1)).sum() / 2

            #     if positions_df.loc[date, stock] -  positions_df.loc[pre_date, stock] < 0:
            #         if price_df.loc[date, stock] - price_df.loc[pre_date, stock] > 0:
            #             profit_trade += 1
            #         else:
            #             unprofit_trade += 1

            #     elif positions_df.loc[date, stock] -  positions_df.loc[pre_date, stock] < 0:
            #         if price_df.loc[date, stock] - price_df.loc[pre_date, stock] < 0:
            #             profit_trade += 1
            #         else:
            #             unprofit_trade += 1


        cash[date] += cash_net

        cash_with_cost[date] += cash_net
        cash_with_cost[date] -= current_trading_cost

        assets_with_cost[date] = assets[date] + cash_with_cost[date]

        assets[date] += cash[date]

        pre_date = date

    return assets, cash, assets_with_cost, cash_with_cost

def calculate_max_drawdown(assets):
    values = list(assets.values())
    max_drawdown = 0
    peak = values[0]
    trough = values[0]

    for i in range(1, len(values)):
        if values[i] > peak:
            peak = values[i]
            trough = values[i]
        elif values[i] < trough:
            trough = values[i]
        
        drawdown = (peak - trough) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown

def calculate_metrics(calendar, test_window, train_window, account_dict, d):
    single_dict = dict()
    sum_dict = dict()

    for rw in tqdm(range(len(calendar) // test_window - train_window//test_window)):
        date = str(calendar[rw * test_window + train_window])[:10]
        account_rw = account_dict[date]
        single_dict[rw] = dict()
        sum_dict[rw] = dict()

        test_set = d.loc[str(calendar[rw*test_window+train_window]):str(calendar[(rw+1)*test_window+train_window-1])].copy()
        test_set.loc[:, '0'] = 0
        Sum = pd.DataFrame(index=calendar[rw*test_window+train_window:(rw+1)*test_window+train_window])
        Sum.index = pd.to_datetime(Sum.index)
        Sum['SumRet'] = 0
        Sum['SumStock'] = 0
        Sum['SumAsset'] = 0
        SumRet = Sum['SumRet']
        SumStock = Sum['SumStock']
        SumAsset = Sum['SumAsset']
        count = 0
        no_trade_count = 0 
        single_trade_count = 0 
        profit_count = 0 
        profit_trade = 0
        unprofit_trade = 0
        single_dict[rw]['opening'] = []
        single_dict[rw]['openingTime'] = []
        
        for stock_pair, account in account_rw.items():
            opening = 0 
            X1_code, X2_code = stock_pair
            X1_ret = test_set.loc[test_set.index.get_level_values('code') == X1_code, 'SReturns'].astype(float)
            X2_ret = test_set.loc[test_set.index.get_level_values('code') == X2_code, 'SReturns'].astype(float)

            X1_ret = X1_ret.reset_index(level='code', drop=True)
            X2_ret = X2_ret.reset_index(level='code', drop=True)
            
            ret = X1_ret + X2_ret
            DCRet = 20 * (account['Asset']/account['Asset'].shift(1)-1)
            
            count += 1
            if not (account['Position'] != 0).any():
                no_trade_count += 1
            
            opening = abs(account['Position'] - account['Position'].shift(1)).sum() / 2
            if opening == 1:
                single_trade_count += 1
            try:
                profit_trade += (((account['Cash'] - account['Cash'].shift(1)) > 0) & ((account['Asset'] - account['Asset'].shift(1)) > 0)).value_counts()[True]  #((account['Cash'] - account['Cash'].shift(1)) > 0).value_counts()[True]
            except:
                profit_trade += 0
            try:
                unprofit_trade += (((account['Cash'] - account['Cash'].shift(1)) < 0) & ((account['Asset'] - account['Asset'].shift(1)) < 0)).value_counts()[True]   #((account['Cash'] - account['Cash'].shift(1)) < 0).value_counts()[True]
            except:
                unprofit_trade += 0
                
            cumStock = np.cumprod(1 + ret) - 1
            cumTrade = np.cumprod(1 + DCRet) - 1
        
            TotcumStock = cumStock + 1
            TotcumTrade = cumTrade + 1
            drawdown_stock = (TotcumStock.cummax() - TotcumStock) / TotcumStock.cummax()
            drawdown_macd = (TotcumTrade.cummax() - TotcumTrade) / TotcumTrade.cummax()
            
            retmean = 252 * ret.mean()
            retstd = np.sqrt(252) * ret.std()
            DCRetmean = 252 * DCRet.mean()
            DCRetstd = np.sqrt(252) * DCRet.std()
            
            if np.array(TotcumTrade)[-1] > 1:
                profit_count += 1
            
            single_dict[rw]['opening'].append(opening)
            try:
                t = (account['Position'] != 0).value_counts()[True]
            except:
                t = 0
                
            single_dict[rw]['openingTime'].append(t / len(account['Position']))
            
            SumRet += DCRet
            SumStock = SumStock + account['Asset']-account['Cash']
            SumAsset = SumAsset + account['Asset']

        sum_dict[rw]['NoTradingPairs'] = no_trade_count
        sum_dict[rw]['TradingPairs'] = count
        sum_dict[rw]['SingleTradingPairs'] = single_trade_count
        sum_dict[rw]['opening'] = sum(single_dict[rw]['opening'])
        sum_dict[rw]['ProfitPairs'] = profit_count
        sum_dict[rw]['ProfitTrades'] = profit_trade
        sum_dict[rw]['UnProfitTrades'] = unprofit_trade
        sum_dict[rw]['openingTime'] = np.mean(single_dict[rw]['openingTime'])
        
        SumRet /= (count - no_trade_count)
        SumCumTrade = np.cumprod(1 + SumRet) - 1 
        TotSumCumTrade = SumCumTrade + 1
        drawdown_position = (TotSumCumTrade.cummax() - TotSumCumTrade) / TotSumCumTrade.cummax()
        
        if rw != 0:
            AllRet = pd.concat([AllRet, SumRet], axis=0, ignore_index=False)
            AllStock = pd.concat([AllStock, SumStock], axis=0, ignore_index=False)
            AllAsset = pd.concat([AllAsset, SumAsset * AllAsset.iloc[-2] / SumAsset.iloc[0]], axis=0, ignore_index=False)
        else:
            AllRet = SumRet
            AllStock = SumStock
            AllAsset = SumAsset
            
        allretmean = 252 * SumRet.mean()
        allretstd = np.sqrt(252) * SumRet.std()
        
        sum_dict[rw]['YearReturn'] = allretmean
        sum_dict[rw]['YearSTD'] = allretstd
        
        try:
            sum_dict[rw]['Daily returns > 0 (%)'] = (SumRet > 0).value_counts()[True] / len(SumRet)
        except:
            sum_dict[rw]['Daily returns > 0 (%)'] = 0
        sum_dict[rw]['drawdown'] = drawdown_position.max()

    return sum_dict, AllRet,AllStock,AllAsset
