# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:50:32 2018

@author: wuwangchuxin
"""
import numpy as np
import pandas as pd
from numpy import nan
#from pandas import Series,DataFrame
import statsmodels.api as sm
import os
os.chdir('C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/')

def poss_date(date):
    if len(date) == 10:
        return date[:4]+'-'+date[5:7]+'-'+date[8:]
    elif len(date) == 8:
        return date[:4]+'-0'+date[5]+'-0'+date[-1]
    elif date[-2] == r'/':
        return date[:4]+'-'+date[5:7]+'-0'+date[-1]
    else:
        return date[:4]+'-0'+date[5]+'-'+date[-2:]

financial_data = pd.read_csv('stock_financial_data_0_868.csv')
financial_data2 = pd.read_csv('stock_financial_data_869_2200.csv')
financial_data3 = pd.read_csv('stock_financial_data_2201_end.csv')
financial_data_all = financial_data.append(financial_data2,ignore_index=False).append(financial_data3,ignore_index=False)

# pe因子
pettm_month = pd.read_csv('pettm_month_data.csv',index_col=0)
pettm_month.columns = map(lambda x:poss_date(x),list(pettm_month.columns))




data_pe = pd.read_csv('internet_data/pe_ttm_120m.csv',index_col=0)
data_yield = pd.read_csv('internet_data/yield_data_120m.csv',index_col=0)
data_float_a_shares = pd.read_csv('internet_data/float_a_shares_120m.csv',index_col=0)
data_hs300 = pd.read_csv('internet_data/hs300_120m.csv',index_col=0)

# 
data_pe.index.name = 'date'
data_yield.index.name = 'data'
data_float_a_shares.index.name = 'data'

data_pe = data_pe.apply(lambda x:1./x)
print (data_pe)

T = len(data_yield.columns)
data_yield = data_yield.replace(0,nan)
data_float_col = data_float_a_shares.replace(0,nan)
t_values = []

for i in range(1,T):
    data_yield_col = data_yield.iloc[:,i].dropna()
    data_pe_col = data_pe.iloc[:,i-1].reindex(index=data_yield_col.index)
    #中位数去极值法
    data_pe_median = data_pe_col.median()
    data_pe_minus_median_ser=data_pe_col-data_pe_median
    data_pe_minus_median_ser_med = data_pe_minus_median_ser.abs().median()
    big_num=data_pe_median+5*data_pe_minus_median_ser_med
    small_num=data_pe_median-5*data_pe_minus_median_ser_med
    data_pe_col[data_pe_col>big_num]=big_num
    data_pe_col[data_pe_col<small_num]=small_num
    #标准化
    mean_col = data_pe_col.mean()
    std_col = data_pe_col.std()
    data_pe_col = data_pe_col.apply(lambda x:(x-mean_col)/std_col)
    data_pe_col=data_pe_col.replace(nan,0)
    
    data_float_col=data_float_a_shares.iloc[:,i-1].reindex(index=data_yield_col.index)
    # 个股相对于hs300的超额收益
    data_yield_col = data_yield_col.subtract(data_hs300.iloc[:,i].values[0])
    data=pd.concat([data_yield_col,data_pe_col,data_float_col],axis=1)
    data=data.dropna()
    # 以流通市值的平方根作为权值
    w=data.iloc[:,2].tolist()
    w=np.array([i**0.5 for i in w])
    mod_wls = sm.WLS(data.iloc[:,0],data.iloc[:,1],w).fit()
    t_values.append(mod_wls.tvalues[0])
    
t_values=np.array(t_values)
t_values_mean=t_values.mean()
t_values_mean_div_std=t_values_mean/t_values.std()
abs_t_values=np.abs(t_values)
abs_t_values_mean = abs_t_values.mean()

abs_over2 = np.where(abs_t_values<2,nan,abs_t_values)
abs_over2=abs_over2[np.logical_not(np.isnan(abs_over2))]
abs_over2_per = len(abs_over2)/len(abs_t_values)    
abs_t_values_mean,abs_over2_per,len(abs_over2),len(abs_t_values),t_values_mean,t_values_mean_div_std




