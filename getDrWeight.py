import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from scipy.stats import pearsonr


def get_pure_w123(af, wt):
    df = pd.merge(wt, af, left_index=True, right_index=True, how='inner')
    df = df.diff().dropna()
    df['fore_max_2'] = df.iloc[:, 1].shift(1)
    df['fore_max_3'] = df.iloc[:, 1].shift(2)
    df.dropna(inplace=True)
    data_x = df.iloc[:, 1:]
    data_y = df.iloc[:, 0]
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(data_x, data_y)
    return regr.coef_


def get_pure_w4(af, wt, air, n, w123):
    df = pd.merge(wt, af, left_index=True, right_index=True, how='inner')
    df = df.diff().dropna()
    df['fore_max_2'] = df.iloc[:, 1].shift(1)
    df['fore_max_3'] = df.iloc[:, 1].shift(2)
    df.dropna(inplace=True)
    df['pt1'] = df.iloc[:, 1] * w123[0] + df.iloc[:, 2] * w123[1] + df.iloc[:, 3] * w123[2]
    df = pd.merge(df, wt, left_index=True, right_index=True, how='inner')
    pt1 = df.iloc[:, 4] + df.iloc[:, 5]  # pt1
    ptn = wt.shift(-(n - 1))  # ptn
    ft_diff = air.iloc[:, 2 * n - 1] - air.iloc[:, 1]  # ftn - ft1
    ptn_1 = (ptn - pt1).dropna()
    ft_diff.name = 'ft_diff'
    ptn_1.name = 'ptn_1'
    df_regr = pd.merge(ft_diff, ptn_1, left_index=True, right_index=True, how='inner')
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(np.array(df_regr.iloc[:, 0]).reshape(-1, 1), np.array(df_regr.iloc[:, 1]).reshape(-1, 1))
    return regr.coef_


def get_mix_w12(af, wt):
    df = pd.merge(wt, af, left_index=True, right_index=True, how='inner')
    df['tem_diff'] = df.iloc[:, 0].diff()
    df['tem_diff_2'] = df['tem_diff'].shift()
    df['air_diff'] = df.iloc[:, 1].diff()
    df.dropna(inplace=True)
    data_x = df.iloc[:, 3:]
    data_y = df.iloc[:, 2]
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(data_x, data_y)
    return regr.coef_


def get_mix_w3(af, wt, air, n, w12):
    df = pd.merge(wt, af, left_index=True, right_index=True, how='inner')
    df['tem_diff'] = df.iloc[:, 0].diff()
    df['tem_diff_2'] = df['tem_diff'].shift()
    df['air_diff'] = df.iloc[:, 1].diff()
    df.dropna(inplace=True)
    mt1 = df.iloc[:, 3] * w12[0] + df.iloc[:, 4] * w12[1] + df.iloc[:, 0].shift()
    mt1.dropna(inplace=True)  # mt1
    mtn = wt.shift(-(n - 1))  # mtn
    ft_diff = air.iloc[:, 2 * n - 1] - air.iloc[:, 1]  # ftn - ft1
    mtn_1 = (mtn - mt1).dropna()
    ft_diff.name = 'ft_diff'
    mtn_1.name = 'mtn_1'
    df_regr = pd.merge(ft_diff, mtn_1, left_index=True, right_index=True, how='inner')
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(np.array(df_regr.iloc[:, 0]).reshape(-1, 1), np.array(df_regr.iloc[:, 1]).reshape(-1, 1))
    return regr.coef_


def get_coef_max(af, wt, airforecast):
    pure_w123 = list(get_pure_w123(af, wt))
    mix_w12 = list(get_mix_w12(af, wt))
    pure_w4 = []
    mix_w3 = []
    for i in range(2, 8):
        pure_w4.append(round(float(get_pure_w4(af, wt, airforecast, i, pure_w123)), 5))
        mix_w3.append(round(float(get_mix_w3(af, wt, airforecast, i, mix_w12)), 5))
    pure_w123.append(pure_w4)
    mix_w12.append(mix_w3)
    coef = {
        'pure_coef_max': pure_w123,
        'mix_coef_max': mix_w12,
    }
    return coef


def get_coef_min(af, wt, airforecast):
    pure_w123 = list(get_pure_w123(af, wt))
    mix_w12 = list(get_mix_w12(af, wt))
    pure_w4 = []
    mix_w3 = []
    for i in range(2, 8):
        pure_w4.append(round(float(get_pure_w4(af, wt, airforecast, i, pure_w123)), 5))
        mix_w3.append(round(float(get_mix_w3(af, wt, airforecast, i, mix_w12)), 5))
    pure_w123.append(pure_w4)
    mix_w12.append(mix_w3)
    coef = {
        'pure_coef_max': pure_w123,
        'mix_coef_max': mix_w12,
    }
    return coef


if __name__ == '__main__':
    airforecast = pd.read_csv('.\\airforecast.csv', index_col=0, parse_dates=True)
    ypj = pd.read_csv('.\\data\\ypj1.csv', index_col=0, parse_dates=True)
    jy = pd.read_csv('.\\data\\jy.csv', index_col=0, parse_dates=True)
    af = airforecast.iloc[:, 1]
    wt = ypj.iloc[:int(len(ypj) * 0.3), 0]
    print(get_coef_max(af, wt, airforecast))
