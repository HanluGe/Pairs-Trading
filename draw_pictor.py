import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def plot_asset_value_over_time(AllAsset):
    # 设置seaborn风格
    sns.set(style="whitegrid")

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制折线图
    plt.plot(AllAsset.index, AllAsset.values, color='skyblue', linestyle='-', marker='o', markersize=3)

    # 添加标题和标签
    plt.title('Asset Value Over Time', fontsize=16)
    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Asset Value', fontsize=14)

    # 显示图例
    plt.legend(['Asset Value'], loc='upper left', fontsize=12)

    # 显示图表
    plt.show()

def plot_assets_and_cash(assets_dict, cash_dict,name):
    # 将字典转换为DataFrame以便绘图
    assets_df = pd.DataFrame.from_dict(assets_dict, orient='index', columns=['Assets'])
    cash_df = pd.DataFrame.from_dict(cash_dict, orient='index', columns=['Cash'])

    # 绘制资产和现金的图像
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
    plt.savefig(name+'.png')


def plot_heatmap_from_files(location):
    # 获取文件夹下所有文件名
    file_names = os.listdir(location)
    # 筛选出所有.npy文件
    npy_files = [file for file in file_names if file.endswith('.npy')]

    stock_pair_num = dict()

    for npy_file in npy_files:
        pairs = np.load(os.path.join(location, npy_file), allow_pickle=True)

        for stock_pair in pairs:
            stock_pair_tuple = tuple(stock_pair)  # 将 numpy 数组转换为元组
            if stock_pair_tuple in stock_pair_num.keys():
                stock_pair_num[stock_pair_tuple] += 1
            else:
                stock_pair_num[stock_pair_tuple] = 1

    # 创建 DataFrame
    df = pd.DataFrame(stock_pair_num.values(), index=pd.MultiIndex.from_tuples(stock_pair_num.keys()), columns=['Value'])

    # 重塑数据以绘制热力图
    heatmap_data = df.unstack().fillna(0)['Value']
    location_parts = location.split('/')
    # print(location_parts[-2])
    image_name = location_parts[-3] +' '+location_parts[-2]  # 获取最后一个部分作为图片名称

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=False, cmap='YlGnBu', square=True)  # 将 annot 设置为 False
    plt.title('Frequency of '+image_name)
    plt.xlabel('Stock')
    plt.ylabel('Stock')
    plt.show()


def plot_trading_signals(account_dict):
    # 创建 position 文件夹
    os.makedirs('position_pic', exist_ok=True)

    for date in account_dict.keys():
        # 在 position 文件夹下为每个日期创建一个文件夹
        date_folder = os.path.join('position', date)
        os.makedirs(date_folder, exist_ok=True)

        for stock_pair in account_dict[date].keys():
            df = account_dict[date][stock_pair]

            # 创建图形和坐标轴
            fig, ax = plt.subplots()

            # 绘制 Spread 列
            ax.plot(df['Spread'], label='Spread', color='blue', linestyle='-')

            # 绘制 position 列
            ax.plot(df['Position'], label='position', color='green', linestyle='-', alpha=0.5)

            # 绘制 CLOSE 列
            ax.plot(df['CLOSE'], label='CLOSE', color='red', linestyle='--')
            ax.plot(-1 * df['CLOSE'], label='CLOSE', color='red', linestyle='--')

            # 绘制 OPEN 列
            ax.plot(df['OPEN'], label='OPEN', color='orange', linestyle='--')
            ax.plot(-1 * df['OPEN'], label='OPEN', color='orange', linestyle='--')

            # 绘制 STOP 列
            ax.plot(df['STOP'], label='STOP', color='purple', linestyle='--')
            ax.plot(-1 * df['STOP'], label='STOP', color='purple', linestyle='--')

            # 添加图例
            ax.legend()

            # 添加标题
            ax.set_title('Trading Signals')

            # 显示网格
            ax.grid(True)

            # 保存图像到对应的日期文件夹
            plt.savefig(os.path.join(date_folder, f'{stock_pair}.png'))

            # 关闭图形，以便下一次循环时创建新的图形
            plt.close()



