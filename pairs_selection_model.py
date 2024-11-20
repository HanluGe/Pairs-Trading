import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import statsmodels.api as sm
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller

import os
import multiprocessing

def calculate_EG(args):
    i, j, train_set, code = args
    X1 = train_set.loc[train_set.index.get_level_values('code') == code[i], 'factor'].astype(float)
    X2 = train_set.loc[train_set.index.get_level_values('code') == code[j], 'factor'].astype(float)

    # 将索引重置为普通的列
    X1 = X1.reset_index(level='code', drop=True)
    X2 = X2.reset_index(level='code', drop=True)

    if X1.max() != X1.min() and X2.max() != X2.min():
        X1 = sm.add_constant(X1)
        results = sm.OLS(X2.astype(float), X1.astype(float)).fit()

        # Get rid of the constant column
        X1 = X1['factor']

        b = results.params['factor']
        Z = X2.astype(float) - b * X1.astype(float)

        p_value = adfuller(Z)[1]

    else:
        p_value = np.nan

    return (i, j, p_value)

def EG_cointegrate(train_set, code):
    pairs_dict = {'99%': [], '95%': [], '90%': []}

    pool = multiprocessing.Pool()
    results = []

    for i in range(len(code)):
        for j in range(i+1, len(code)):
            args = (i, j, train_set, code)
            results.append(pool.apply_async(calculate_EG, (args,)))

    pool.close()
    pool.join()

    for res in results:
        i, j, p_value = res.get()
        if p_value < 0.1:
            pairs_dict['90%'].append((code[i], code[j]))
            if p_value < 0.05:
                pairs_dict['95%'].append((code[i], code[j]))
                if p_value < 0.01:
                    pairs_dict['99%'].append((code[i], code[j]))

    return pairs_dict

def Johansen_cointegrate(train_set, code):
    pairs_dict = dict()
    pairs_dict['99%'] = []
    pairs_dict['95%'] = []
    pairs_dict['90%'] = []

    for i in range(len(code)):
        for j in range(i+1, len(code)):
            X1 = train_set.loc[train_set.index.get_level_values('code') == code[i], 'factor'].astype(float)
            X2 = train_set.loc[train_set.index.get_level_values('code') == code[j], 'factor'].astype(float)
            if X1.max() != X1.min() and X2.max() != X2.min():
                arr = np.vstack((X1, X2))
                jres = coint_johansen(arr.T, det_order=0, k_ar_diff=10)

                if jres.cvt[0][0] < jres.lr1[0]: #大于90%临界值，拒绝原假设
                    pairs_dict['90%'].append((code[i],code[j]))
                if jres.cvt[0][1] < jres.lr1[0]: #大于95%临界值，拒绝原假设
                    pairs_dict['95%'].append((code[i],code[j]))
                if jres.cvt[0][2] < jres.lr1[0]: #大于99%临界值，拒绝原假设
                    pairs_dict['99%'].append((code[i],code[j]))
    return pairs_dict

def store_pairs(pairs_dict,date,method):
    for confidence_level in pairs_dict.keys():
        np.save('pairs/'+method+'/'+confidence_level+'/'+date+'.npy', pairs_dict[confidence_level])

def store_pairs(pairs_dict, method, date):
    for confidence_level in pairs_dict.keys():
        folder_path = os.path.join('pairs', method, confidence_level)
        os.makedirs(folder_path, exist_ok=True)  # Create folder if not exist
        file_path = os.path.join(folder_path, date + '.npy')
        np.save(file_path, pairs_dict[confidence_level])

def run_model(calendar, data, code, test_window, train_window,method):
    pairs_dict_daily = {}
    for rw in tqdm(range(len(calendar) // test_window - train_window//test_window)):
        train_set = data.loc[str(calendar[rw * test_window]):str(calendar[rw * test_window + train_window - 1])]
        
        if method == 'Johansen':
            pairs_dict = Johansen_cointegrate(train_set, code)
            store_pairs(pairs_dict,'johansen',str(calendar[rw * test_window + train_window])[:10])

        elif method == 'EG':
            pairs_dict = EG_cointegrate(train_set, code)
            store_pairs(pairs_dict,'EG',str(calendar[rw * test_window + train_window])[:10])

        pairs_dict_daily[str(calendar[rw * test_window + train_window])[:10]]=pairs_dict

    return pairs_dict_daily

