# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 13:41:32 2018

@author: wuwangchuxin
"""

import pandas as pd
import numpy as np
import os
os.chdir('D:/multiple_factors_models/')
import date_process_class as dpc

def get_stock_startdate_enddate():
    # code,marketdate as startdate,enddate
    add_data = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/'
    stockcode = pd.read_excel(add_data+'stock_code.xlsx')
    # 首次戴帽时间
    stockcode['dmzm_date'] = stockcode['dmzm_date'].apply(lambda x:str(x)[-8:-4]\
                               +'-'+str(x)[-4:-2]+'-'+str(x)[-2:] if not str(x)=='nan' else x)
    # 剔除B股等
    stockcode = stockcode[stockcode['code'].apply(lambda x:x[:2]).isin(['00','30','60'])]
    #stockcode['zhaipai_date'] = stockcode['zhaipai_date'].fillna(datetime.datetime.now().strftime('%Y-%m-%d')).apply(lambda x:str(x)[:10]) 
    # 最新取到2018-07-31
    stockcode['zhaipai_date'] = stockcode['zhaipai_date'].fillna('2018-08-01').apply(lambda x:str(x)[:10]) 
    stockcode.reset_index(drop=True,inplace=True)
    # 
    for i in range(len(stockcode['dmzm_date'])):
        if not stockcode.loc[i,'dmzm_date']==stockcode.loc[i,'dmzm_date']: #dmzm_date为空时
            stockcode.loc[i,'enddate'] = dpc.DateProcess(stockcode.loc[i,'zhaipai_date']).tdays_offest(-1)
        elif stockcode.loc[i,'dmzm_date'] < stockcode.loc[i,'zhaipai_date']:
            stockcode.loc[i,'enddate'] = dpc.DateProcess(stockcode.loc[i,'dmzm_date']).tdays_offest(-1)
        else:
            stockcode.loc[i,'enddate'] = dpc.DateProcess(stockcode.loc[i,'zhaipai_date']).tdays_offest(-1)
    stockcode['market_date'] = stockcode['market_date'].apply(lambda x:str(x)[:10])
    # 选取2009-01-01以后的数据
    stockcode = stockcode[stockcode['enddate']>='2009-01-01']
    stockcode.reset_index(drop=True,inplace=True)
    
    for j in range(len(stockcode['market_date'])):
        if stockcode.loc[j,'market_date']<'2009-01-01':
            stockcode.loc[j,'market_date'] = '2009-01-01'
    stockcode.to_csv(add_data+'stock_code.csv')
    #保存日期
    stockcode.market_date = stockcode.market_date.apply(lambda x:dpc.DateProcess(x).format_date())
    stockcode.enddate = stockcode.enddate.apply(lambda x:dpc.DateProcess(x).format_date())
    np.save(add_data+'prepared_data/start_date',stockcode.market_date)
    np.save(add_data+'prepared_data/end_date',stockcode.enddate)

if __name__=='__main__':
    get_stock_startdate_enddate()




























