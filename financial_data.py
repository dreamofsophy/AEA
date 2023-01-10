import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.optimize as spo
import tensorflow as tf

class FinancialData(object):
    def __init__(self,tickers=['SPY'],fillna=True,cols=None,  **kwargs):
        if isinstance(tickers,list):
            t = ' '.join(tickers)
            df = yf.download(t, **kwargs)
        
        elif isinstance(tickers,str):
            df = yf.download(tickers,**kwargs)

        if not isinstance(cols,type(None)):
            df = df[cols]

        if fillna:
            df.fillna(method='ffill',inplace=True)
            df.fillna(method='bfill',inplace=True)

        df = one_lvl_colnames(df,cols,tickers)
        self.tickers = tickers
        self.df = df
        self.columns = cols
    def rolling_statistics(self,cols='Adj Close',tickers=None,functions=None,
                      window=20,bollinger=False,roll_linewidth=1.5,**kwargs):
        df = self.df
        if isinstance(tickers,type(None)):
            tickers = self.get_tickers()
        col_names = return_names(cols,tickers)
        if isinstance(functions,type(None)):
            functions = [momentum, simple_moving_average, bollinger_bands]
        elif not isinstance(functions,list):
            functions = [functions]
        df = df[col_names]
        rolling_stats = df.rolling(window).agg(functions)
        rolling_stats = one_lvl_colnames(rolling_stats,col_names,functions)
        return rolling_stats
    
    def get_returns(self,cols='Adj Close',tickers=None, return_window=1, **kwargs):
        df = self.df
        if isinstance(tickers, type(None)):
            tickers = self.get_tickers()
        col_names = return_names(cols,tickers)
        returns = df[col_names].pct_change(return_window)

        self.returns = returns.dropna(how='all')
        self.return_window = return_window
        return self.returns
    
    def get_normalized_prices(self, start_date=None, prices_col='Adj Close',
                              title=None,x_label='Fecha',y_label='Norm P',
                              fontsize=15,**kwargs):
        # Define important variables:
        if not isinstance(prices_col,list):
            prices_col = [prices_col]
        prices_names = return_names(prices_col,self.get_tickers())
        prices = self.get_data()[prices_names]
        if isinstance(start_date,type(None)):
            start_date = prices.index.min()
        base = prices.loc[prices.index==start_date].values
        norm_prices = prices/base*100
        return norm_prices

    def get_tickers(self):
        return self.tickers

    def get_data(self):
        return self.df


class Portfolio(FinancialData):
    def __init__(self,tickers=['SPY'],fillna=True,cols=None,weights=[1],
                 **kwargs):
        FinancialData.__init__(self,tickers,fillna,cols,**kwargs)
        columns = [column+'_'+ticker for ticker in tickers]
        prices = self.prepare_data(fillna=fillna)
        self.prices = prices.loc[:,columns]
        self.weights = weights
    
    def normalize_prices(self,start_date=None,end_date=None,tickers=None,column='Close'):
        prices = self.prices
        if tickers == None:
            tickers = self.get_tickers()
        if start_date is None:
            start_date = prices.index.values.min()
        if end_date is None:
            end_date = prices.index.values.max()
        columns = [column+'_'+ticker for ticker in tickers]
        norm_prices = prices.loc[start_date:end_date,columns]/prices.loc[start_date,columns]
        return norm_prices
    
    def get_portfolio_values(self,start_date=None,end_date=None,tickers=None,column='Close'):
        prices = self.get_prices()
        weights = self.get_weights()
        if start_date == None:
            start_date = prices.index.values.min()
        if end_date == None:
            end_date = prices.index.values.max()
        if tickers == None:
            tickers = self.get_tickers()
        norm_prices = self.normalize_prices(start_date,end_date,tickers,column)
        portfolio_values = norm_prices*weights
        portfolio_values['Portfolio'] = portfolio_values.sum(axis=1)
        return portfolio_values

    def get_prices(self):
        return self.prices
    
    def get_weights(self):
        return self.weights
    
    def change_weights(self,weights):
        assert len(self.weights) == len(weights), "Wrong length of weights"
        self.weights = weights
    
    def get_returns(self,start_date=None,end_date=None,tickers=None,
                    column='Close',window=1,portfolio_returns=False):
        prices = self.get_prices()
        if start_date is None:
            start_date = prices.index.values.min()
        if end_date is None:
            end_date = prices.index.values.max()
        if tickers is None:
            tickers = self.get_tickers()
        columns = [column+'_'+ticker for ticker in tickers]
        prices = prices.loc[start_date:end_date,columns]
        returns = prices.pct_change(window).dropna(how='all')
        if portfolio_returns:
            weights = self.get_weights()
            returns['Portfolio'] = (returns*weights).sum(axis=1)
        return returns
    
    def get_performance_metrics(self,risk_free_rate=0,start_date=None,
                                end_date=None,**kwargs):
        if start_date == None:
            start_date = self.get_prices().index.values.min()
        if end_date == None:
            end_date = self.get_prices().index.values.max()
        portfolio_values = self.get_portfolio_values(start_date,end_date,
                                                    **kwargs)
        def compute_cum_return(series):
            mn = series.index.values.min()
            mx = series.index.values.max()
            cum_return = (series[mx]/series[mn])-1
            return cum_return  

        cum_return = compute_cum_return(portfolio_values['Portfolio'])
        returns = self.get_returns(start_date,end_date,
                                   portfolio_returns=True,**kwargs)
        sharpe_ratio = self.get_sharpe_ratio(risk_free_rate,start_date=start_date,
                                             end_date=end_date,**kwargs)
        metrics = returns['Portfolio'].agg(['mean','std'])
        metrics.loc['Cum Return'] = cum_return
        metrics.loc['Sharpe Ratio'] = sharpe_ratio
        return metrics

    def get_sharpe_ratio(self,weights=None,rfr=0,negative=False,**kwargs):
        returns = self.get_returns(**kwargs)
        if weights is None:
            weights = self.get_weights()
        portfolio_returns = (returns*weights).sum(axis=1)
        portfolio_std = portfolio_returns.std()
        sharpe_ratio = (portfolio_returns-rfr).mean()/portfolio_std
        if negative:
            sharpe_ratio *= -1
        return sharpe_ratio

    def optimize_portfolio(self,guess_weights=None,short=False,rfr=0,**kwargs):
        tickers = self.get_tickers()
        if guess_weights is None:
            guess_weights = [1/len(tickers) for i in range(len(tickers))]
        if not short:
            bounds = [(0,1) for i in range(len(tickers))]
        else:
            bounds = [(-1,1) for i in range(len(tickers))]
        weights_sum_to_1 = {'type':'eq', 'fun':lambda weights: np.sum(np.absolute( weights))-1}
        opt_weights = spo.minimize(
            self.get_sharpe_ratio,guess_weights,
            args=(rfr,True),
            method='SLSQP', options={'disp':False},
            constraints=(weights_sum_to_1),
            bounds=bounds
        )
        self.change_weights(opt_weights.x)
        return opt_weights.x

class Sample_Generator():
    def __init__(self,input_width=5,label_width=1,shift=1, train_df=None, val_df=None,
                 test_df=None, label_columns=None,batch_size=None,shuffle=False):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_columns = label_columns
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.batch_size = batch_size
        self.shuffle = shuffle
        if isinstance(label_columns,type(None)):
            self.label_columns_indices = {name:i for i,name in enumerate(label_columns)}
        self.column_indices = {name:i for i,name in enumerate(train_df.columns)}
        self.total_window_size = input_width+shift
        self.input_slice = slice(0,input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size-self.label_width
        self.labels_slice = slice(self.label_start,None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def split_window(self, features):
        inputs = features[:, self.input_slice,:]
        labels = features[:,self.labels_slice,:]
        if not isinstance(self.label_columns,type(None)):
            labels = tf.stack(
                [labels[:,:,self.column_indices[name]] for name in self.label_columns],
                axis = -1
            )
        inputs.set_shape([None,self.input_width,None])
        labels.set_shape([None,self.label_width,None])

        return inputs,labels
    
    def make_dataset(self,data):
        data = np.array(data,dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data = data,
            targets = None,
            sequence_length = self.total_window_size,
            sequence_stride = 1,
            shuffle = self.shuffle,
            batch_size = self.batch_size
        )
        ds = ds.map(self.split_window)
        return ds
   

def one_lvl_colnames(df,cols,tickers):
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(df.columns.values[0], tuple):
        columns = df.columns.values
        new_cols = []
        for col in columns:
            temp = []
            for name in col:
                if name != '':
                    temp.append(name)
            new_temp = '_'.join(temp)
            new_cols.append(new_temp)
        df.columns = new_cols
    elif isinstance(df.columns.values[0], str):
        col_names = [column+'_'+ticker for column in cols for ticker in tickers]
        df.columns = col_names
    return df

def momentum(prices):
    first = prices.iloc[0]
    last = prices.iloc[-1]
    momentum_df = last/first
    return momentum_df

def simple_moving_average(prices):
    mean = prices.mean()
    sma = prices[-1]/mean-1
    return sma
    
def bollinger_bands(prices):
    ma = prices.mean()
    std = prices.std()
    bb = (prices[-1]-ma)/(2*std)
    return bb

def daily_rate(x, periods_year=252):
    dr = np.power(1+x,1/periods_year)-1
    return dr

def sharpe_ratio(weights=None, rfr=0, negative=False, returns=0):
    num_assets = returns.shape[1]
    if isinstance(weights,type(None)):
        weights = [1/num_assets for i in range(num_assets)]
    portfolio_returns = (returns*weights).sum(axis=1)
    portfolio_std = portfolio_returns.std()
    sharpe_ratio = (portfolio_returns-rfr).mean()/portfolio_std
    if negative:
        sharpe_ratio *= -1
    return sharpe_ratio    
