import pandas as pd
import os
import datetime
import numpy as np

class DataProcessor:
    def __init__(
            self, start_date, end_date, 
            data_path = '../Mid-To-Long-Term Quantitative Researcher/sample_data'
            ):
        
        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
    
    def read_data(self, data_type='vwap'):
        combined_data = pd.DataFrame()
        
        for i in range(self.start_date, self.end_date + 1):
            file_path = os.path.join(self.data_path, f'{data_type}_{i:04d}.csv')
            
            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
                date = pd.to_datetime(i, format='%m%d')
                data['Minutes'] = date.strftime('%m-%d') + ' ' + data['Minutes']
                combined_data = pd.concat([combined_data, data], ignore_index=True)

        combined_data = combined_data.ffill()
        formatted_data = pd.melt(combined_data, id_vars=['Minutes'], var_name='code', value_name=data_type)
        formatted_data['date'] = pd.to_datetime(formatted_data['Minutes'], format='%m-%d %H:%M:%S')
        formatted_data = formatted_data[['date', 'code', data_type]]
        self.data = formatted_data
        
        return formatted_data
    
    def resample_df(self, df, freq='15T'):
        # resampled_df = df.set_index('date')
        # resampled_df = resampled_df.groupby('code').resample(freq, closed='left', label='right').mean()

        # formatted_data = resampled_df.reset_index().rename(columns={'level_0': 'date', 'level_1': 'code'})
        # formatted_data = formatted_data[['date', 'code', 'vwap']]
        # formatted_data = formatted_data.dropna()

        resampled_df = df.set_index('date')
        mask = (resampled_df.index.time == datetime.time(12, 0)) | (resampled_df.index.time == datetime.time(16, 0))
        resampled_df.loc[mask, 'vwap'] = np.nan

        formatted_data = resampled_df.groupby('code').resample(freq, closed='left', label='right').mean()

        formatted_data = formatted_data.reset_index().rename(columns={'level_0': 'date', 'level_1': 'code'})
        formatted_data = formatted_data[['date', 'code', 'vwap']]
        formatted_data = formatted_data.dropna()

        return formatted_data

    def volume_weighted_average_price(self, df, weight, freq='15T'):

        combined_data = pd.merge(df, weight, on = ['date','code'])
        combined_data['sum'] = combined_data['vwap'] * combined_data['volume']

        combined_data = combined_data.set_index('date')
        mask = (combined_data.index.time == datetime.time(12, 0)) | (combined_data.index.time == datetime.time(16, 0))
        combined_data.loc[mask, ['vwap','volume','sum']] = np.nan

        weighted_sum = combined_data.groupby('code').resample(freq, closed='left', label='right').sum()
        weighted_price = weighted_sum['sum']/ weighted_sum['volume']

        # 将多级索引转换为单个索引，并重新排列列的顺序
        formatted_data = weighted_price.reset_index().rename(columns={'level_0': 'date', 'level_1': 'code'})
        formatted_data = formatted_data[['date', 'code', 0]]
        formatted_data = formatted_data.rename(columns={0: 'vwap'})
        
        formatted_data = formatted_data[(formatted_data['date'].dt.time > pd.to_datetime('09:30:00').time()) & 
                                    (formatted_data['date'].dt.time <= pd.to_datetime('12:00:00').time()) |
                                    (formatted_data['date'].dt.time > pd.to_datetime('13:00:00').time()) & 
                                    (formatted_data['date'].dt.time <= pd.to_datetime('16:00:00').time())]

        trading_dates = combined_data.between_time('09:30', '11:30').index.date
        trading_dates = list(set(trading_dates)) 

        # 把非交易日的数据删掉
        formatted_data = formatted_data[formatted_data['date'].dt.date.isin(trading_dates)]
        
        formatted_data = formatted_data.ffill()

        return formatted_data

    def data_process(self,df):
        
        df['SReturns'] = df.groupby('code')['open'].pct_change()
        df['CCReturns'] = df.groupby('code')['open'].pct_change().add(1).cumprod().sub(1)
        result_data = df[['date', 'code', 'open', 'SReturns', 'CCReturns']]

        return result_data
    
    def factor_process(self,df,df_close):
        df['date'] = pd.to_datetime(df['date'])
        calendar = df.date.unique()
        code = df.code.unique()
        # d = df.set_index(['date','code'])
        # d.sort_index(inplace=True)

        df.columns = ['date', 'code', 'factor']
        # 按日期进行分组并计算每个日期的数据点数量
        daily_length = df.groupby(df['date'].dt.date).size()//100
        # 找到最常见的一天的数据长度
        day_length = daily_length.mode()[0]

        df_close = df_close.set_index(['date','code'])
        df = df.set_index(['date','code'])

        df_all = df.join(df_close)

        df_all['SReturns'] = df_all.groupby('code')['open'].pct_change()
        df_all['CCReturns'] = df_all.groupby('code')['open'].pct_change().add(1).cumprod().sub(1)

        df_all_sorted = df_all.sort_index()

        return calendar, code, df_all_sorted, day_length



