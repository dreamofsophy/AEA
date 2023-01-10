import cvxpy as cp
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yfinance as yf
import random
import copy
from math import *
from numpy import linalg as LA
import csv
import os
from financial_data import *
import pywt
from hog import *
from statistics import mean
#quandl.save_key('HtwBLPt3k37yZHTvy15K')
path = './'

def filter_bank(index_list, wavefunc='db4', lv=1, m=1, n=1):
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)   
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0
    for i in range(m,n+1):
        cD = coeff[i]
        Tr = np.sqrt(2*np.log2(len(cD)))*10 
        for j in range(len(cD)):
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) * (np.abs(cD[j]) -  Tr)
            else:
                coeff[i][j] = 0
    coeffs = {}
    for i in range(len(coeff)):
        coeffs[i] = copy.deepcopy(coeff)
        for j in range(len(coeff)):
            if j != i:
                coeffs[i][j] = np.zeros_like(coeff[j])

    for i in range(len(coeff)):
        coeff[i] = pywt.waverec(coeffs[i], wavefunc)
        if len(coeff[i]) > len(index_list):
            coeff[i] = coeff[i][:-1]

    return np.sum(coeff, axis=0)

class Label_Design:
    def __init__(self, mean, covmat):
        self.mean = mean
        self.covmat = covmat
        self.n_assets = self.mean.shape[0]

    def minvar(self, max_weight=1.):
        w = cp.Variable(self.n_assets)
        risk = cp.quad_form(w, self.covmat)
        prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, w <= max_weight, w >= 0])
        prob.solve()
        return w.value

    def maxret(self, max_weight=1.):
        w = cp.Variable(self.n_assets)
        ret = self.mean @ w
        prob = cp.Problem(cp.Maximize(ret), [cp.sum(w) == 1, w <= max_weight, w >= 0])
        prob.solve()
        return w.value

    def maxutil(self, gamma, max_weight=1.):
        w = cp.Variable(self.n_assets)
        ret = self.mean @ w
        risk = cp.quad_form(w, self.covmat)
        prob = cp.Problem(cp.Maximize(ret - gamma * risk), [cp.sum(w) == 1, w <= max_weight, w >= 0])
        prob.solve()
        return w.value

    def compute_portfolio_moments(self, weights):
        pmean = np.inner(weights, self.mean)
        pmean = pmean.item()
        pvar = weights @ self.covmat @ weights
        pvar = pvar.item()
        return pmean, pvar ** (1 / 2)

def synthetic_data(path, n_assets, n_timesteps):
    np.random.seed(21)
    mean_all = pd.read_csv(path + 'sp500_mean.csv', index_col=0, header=None).iloc[:, 0]
    covmat_all = pd.read_csv(path + 'sp500_covmat.csv', index_col=0)
    asset_ixs = np.random.choice(len(mean_all), replace=False, size=n_assets)
    mean = mean_all.iloc[asset_ixs]
    covmat = covmat_all.iloc[asset_ixs, asset_ixs]
    returns = pd.DataFrame(np.random.multivariate_normal(mean.values, covmat.values, size=n_timesteps,
                                                         check_valid='raise'), columns=mean.index)
    return returns

def data_download(n_stocks):
    sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500 = pd.read_html(sp_url, header=0)[0]
    sp500.loc[sp500[sp500['Date first added']=='1983-11-30 (1957-03-04)'].index,'Date first added'] = '1983-11-30'
    sp500['Date first added'] = pd.to_datetime(sp500['Date first added'],format='%Y-%m-%d')
    sp500 = sp500[sp500['Date first added']<'2007-01-01']
    np.random.seed(1792)
    universe_tickers = sp500['Symbol'].unique()
    portfolio_tickers = list(np.random.choice(universe_tickers,replace=False,size=n_stocks))
    ticker_list = portfolio_tickers
    cols = ['Adj Close','Volume']
    start_date = '2007-01-04'
    end_date = '2021-06-25'
    data = yf.download(
        tickers = ticker_list,
        period = '1y',
        interval = '1d',
        group_by = 'ticker',
        auto_adjust = False,
        prepost = False,
        threads = True,
        proxy = None
    )
    data = data.T
    for ticker in ticker_list:
        data.loc[(ticker,),].T.to_csv('yhist/' + ticker + '.csv', sep=',', encoding='utf-8')

def real_data(n_stocks):
    sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500 = pd.read_html(sp_url, header=0)[0]
    sp500.loc[sp500[sp500['Date first added']=='1983-11-30 (1957-03-04)'].index,'Date first added'] = '1983-11-30'
    sp500['Date first added'] = pd.to_datetime(sp500['Date first added'],format='%Y-%m-%d')
    sp500 = sp500[sp500['Date first added']<'2007-01-01']
    np.random.seed(95)
    universe_tickers = sp500['Symbol'].unique()
    portfolio_tickers = list(np.random.choice(universe_tickers,replace=False,size=n_stocks))
    tickers = portfolio_tickers
    cols = ['Adj Close','Volume']
    start_date = '2007-01-04'
    end_date = '2021-06-25'
    my_portfolio = FinancialData(
        tickers = portfolio_tickers,
        cols = ['Adj Close'],
        start = start_date,
        end = end_date)
    port_price = my_portfolio.get_data()
    port_returns = my_portfolio.get_returns(plot=False,subplots=True,figsize=(10,20))
    total_return = port_returns
    total_return.dropna(subset=port_returns.columns,inplace=True)
    total_price = port_price
    total_price.dropna(subset=port_price.columns,inplace=True)
    total_price = total_price.drop(total_price.index[[0]])
    return total_return, total_price

def train_val_test(total_df, splits):
    train_size, val_size, test_size, total_szie = splits
    window_size = 0
    train_df = total_df.iloc[:train_size,:]
    val_df = total_df.iloc[train_size-window_size:train_size+val_size,:]
    test_df = total_df.iloc[train_size+val_size-window_size:,:]
    return train_df, val_df, test_df

def raw_data(n_stocks):
    returns_r , price_r = real_data(n_stocks)
    train_s, val_s, test_s = 0.8, 0., 0.2
    total_size = len(returns_r)
    train_size = int(total_size*train_s)
    val_size = int(total_size*val_s)
    test_size = int(total_size*test_s)
    splits = [train_size, val_size, test_size, total_size]
    train_r, val_r, test_r = train_val_test(returns_r, splits)
    train_p, val_p, test_p = train_val_test(price_r, splits)
    return train_r, test_r, train_p, test_p    

def data_structure_r(returns, feature_params):
    n_timesteps = len(returns)
    lookback, gap, horizon, block, max_weight, gamma = feature_params
    X_list, y_list = [], []
    label_return = []
    for i in range(lookback, n_timesteps - horizon - gap + 1, block):
        X_list.append(returns.values[i - lookback: i, :])
        y_list.append(returns.values[i + gap: i + gap + horizon, :])
        label_return.append(returns.iloc[i + gap: i + gap + horizon, :])
    Label = []
    for y in y_list:    
        markowitz_emp = Label_Design(y.mean(axis=0), np.cov(y.T))
        w = markowitz_emp.maxutil(gamma=gamma, max_weight=max_weight)
        Label.append(w)
    D = X_list[0].reshape(-1, 1)
    for i in range(1, len(X_list)):
        D = np.c_[D, X_list[i].reshape(-1, 1, order='F')]
    return D, Label, label_return

def data_structure_p(price, feature_params):
    n_timesteps = len(price)
    lookback, gap, horizon, block, max_weight, gamma = feature_params
    X_list, y_list = [], []
    for i in range(lookback, n_timesteps - horizon - gap + 1, block):
        #X_list.append(price.iloc[i - lookback: i, :])
        price_block = price.iloc[i + gap: i + gap + horizon, :]
        y_list.append(get_normalized_prices(price_block))    
    return y_list

def get_normalized_prices(prices, start_date=None, prices_col='Adj Close', **kwargs):
    if not isinstance(prices_col,list):
        prices_col = [prices_col]
    if isinstance(start_date,type(None)):
        start_date = prices.index.min()
    base = prices.loc[prices.index==start_date].values
    norm_prices = prices/base
    return norm_prices

def generate_IID_datasets(num_nodes, Dtr, Label):
    P = Dtr.shape[1]
    Label_array = np.array(Label) 
    ind = np.random.permutation(P)
    Dr = Dtr[:, ind]
    label_tem = Label_array[ind, :]
    z = int(P/num_nodes)
    local_datasets = []
    local_label = []
    for k in range(num_nodes):
        Pk = Dr[:, k*z : (k+1)*z]
        label_k = Label[k*z : (k+1)*z]
        local_datasets.append(Pk)
        local_label.append(label_k)
    return local_datasets, local_label

def generate_non_IID_datasets(num_nodes, Dtr, rate, Label):
    Label_array = np.array(Label)
    random.seed(30)
    select_inds = random.sample(range(0, Dtr.shape[1], 1), num_nodes)
    local_tem = Dtr[:, select_inds]
    label_tem = Label_array[select_inds, :]
    Dtr = np.delete(Dtr, select_inds, 1)
    Label_array = np.delete(Label_array, select_inds, 0)
    local_datasets = []
    local_label = []
    for i in range(num_nodes):
        local_datasets.append(local_tem[:, i].reshape(-1,1))
        local_label.append(label_tem[i, :].reshape(1, -1))
    for j in range(num_nodes):
        random.seed(30 + j)
        num_add = random.randint(0, ceil(Dtr.shape[1]*rate))
        random.seed(40 + j)
        select_inds = random.sample(range(0, Dtr.shape[1], 1),num_add)
        local_tem = Dtr[:, select_inds]
        label_tem = Label_array[select_inds, :]
        Dtr = np.delete(Dtr, select_inds, 1)
        Label_array = np.delete(Label_array, select_inds, 0)
        local_datasets[j] = np.c_[local_datasets[j], local_tem]
        local_label[j] = np.r_[local_label[j], label_tem]
        if Dtr.shape[1] == 0:
            break
    return local_datasets, local_label

def Local_datasets(num_nodes, flag, Dtr, Label):
    if flag == 0:
        # IID
        local_datasets, local_label = generate_IID_datasets(num_nodes, Dtr, Label)
    else:
        # non-IID
        rate = 1.0
        local_datasets, local_label = generate_non_IID_datasets(num_nodes, Dtr, rate, Label)
    return local_datasets, local_label

def wavelet_denoising_old(D):
    coeff = filter_bank(D[:, 0], plot=False)
    Dw = coeff.reshape(-1, 1)
    for p in range(1, D.shape[1]):
        coeff = filter_bank(D[:, p], plot=False)
        Dw = np.c_[Dw, coeff]
    return Dw

def wavelet_denoising(X, W_list, delta):
    # Decomposing
    Wm, Wn, Wmi, Wni = W_list
    Z = Wm.dot(X).dot(Wn.T)
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn function
    Tr = np.sqrt(2*np.log2(Z.shape[1]))* delta
    for k in range(Z.shape[0]):
        for l in range(Z.shape[1]):
            if Z[k][l] >= Tr:
                Z[k][l] = sgn(Z[k][l]) * (np.abs(Z[k][l]) -  Tr)  # Shrink to zero
            else:
                Z[k][l] = 0   # Set to zero if smaller than threshold
    Xh = Wmi.dot(Z).dot(Wni.T)
    #Xh = np.maximum(Xh,0)
    #Xh = np.minimum(Xh,1)
    return Xh

def denoising_feature(D, n_stocks, W_list, delta):
    X = D[:, 0].reshape(-1, n_stocks)
    Xh = wavelet_denoising(X, W_list, delta)
    Dw = Xh.reshape(-1, 1)
    for p in range(1, D.shape[1]):
        X = D[:, p].reshape(-1, n_stocks)
        Xh = wavelet_denoising(X, W_list, delta).reshape(-1,1)
        Dw = np.c_[Dw, Xh]
    return Dw

def denoising_feature_old(D, n_stocks, W_list, delta):
    Dw = wavelet_denoising_old(D)
    return Dw

def feature_design(hog_params, sample_pairs, W_list, delta, n_stocks):
    Dtr, Dte = sample_pairs
    Dtr_h = hog_feature(Dtr, hog_params)
    Dte_h = hog_feature(Dte, hog_params)
    Dtr_w = denoising_feature(Dtr, n_stocks, W_list, delta)
    Dte_w = denoising_feature(Dte, n_stocks, W_list, delta)
    Dtr_hw = hog_feature(Dtr_w, hog_params)
    Dte_hw = hog_feature(Dte_w, hog_params)
    return Dtr_h, Dte_h, Dtr_hw, Dte_hw

def hog_feature(D, hog_params):
    d, B, n_stocks = hog_params
    T = D.shape[1]
    x0 = D[:, 0]
    m0 = x0.reshape(-1, n_stocks)
    H = hog(m0, d, B)
    for i in range(1, T):
        xi = D[:, i]
        mi = xi.reshape(-1, n_stocks)
        hi = hog(mi, d, B)
        H = np.c_[H, hi]
    return H

def get_portfolio_values(norm_prices, weights):
    portfolio_values = norm_prices*weights
    portfolio_values['Portfolio'] = portfolio_values.sum(axis=1)
    return portfolio_values['Portfolio']

def get_portfolio_returns(returns, weights):
    portfolio_return = returns*weights
    portfolio_return['Portfolio'] = portfolio_return.sum(axis=1)
    return portfolio_return['Portfolio']

def compute_cum_return(series):
    N = len(series)
    record = np.zeros(N)
    record[0] = series.values[0]
    for i in range(1, N):
        record[i] = record[i-1] + series.values[i]
    cum_return = (record[N-1]/(record[0]*N))-1
    return cum_return  

def cum_protfolio_value(portfolio_returns):
    a0 = 1
    N = len(portfolio_returns)
    cum_return = (a0 * (1 + N * portfolio_returns.mean()) - a0)/a0
    return cum_return  

def get_sharpe_ratio(portfolio_returns, rfr=0):
    portfolio_std = portfolio_returns.std()
    sharpe_ratio = (portfolio_returns-rfr).mean()/portfolio_std
    return sharpe_ratio

def get_performance_metrics(norm_prices, weights, returns):
    portfolio_values = get_portfolio_values(norm_prices, weights)
    cum_return = compute_cum_return(portfolio_values)
    portfolio_returns = get_portfolio_returns(returns, weights)
    sharpe_ratio = get_sharpe_ratio(portfolio_returns)
    metrics = portfolio_returns.agg(['mean','std'])
    metrics.loc['Cum Return'] = cum_return
    metrics.loc['Sharpe Ratio'] = sharpe_ratio
    return metrics


def price_block():
    price_tr = data_structure_p(train_p, feature_params)
    price_te = data_structure_p(test_p, feature_params)
    weights = [0.5, 0.5]
    portfolio_values = get_portfolio_values(price_tr[10], weights)
    cum_return = compute_cum_return(portfolio_values['Portfolio'])
    portfolio_returns = get_portfolio_returns(return_tr[10], weights)
    sharpe_ratio = get_sharpe_ratio(portfolio_returns)
    metrics = get_performance_metrics(price_tr[0], weights, return_tr[0])

def DCT_1D(N):
    n = np.ones((N, N))
    n[0, :] = 1 / np.sqrt(2)
    spatial = np.arange(N).reshape((N, 1))
    spectral = np.arange(N).reshape((1, N))
    spatial = 2 * spatial + 1
    spectral = (spectral * np.pi) / (2 * N)
    h = np.cos(spatial @ spectral)
    D = (1 / np.sqrt(N/2)) * n * (h.T)
    return D

def DWT_1D(data, level, part):
    if level == 1:
        (cA, cD) = pywt.dwt(data, 'haar')
        if part == 'a':
            WT_coeffs = cA
        elif part == 'd':
            WT_coeffs = cD
    else:
        mode='periodization'
        w = pywt.Wavelet('haar')
        WT_coeffs = pywt.downcoef(part, data, w, mode, level)
    return WT_coeffs
