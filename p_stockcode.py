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

add_data = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/'

def get_stock_startdate_enddate():
    # code,marketdate as startdate,enddate
    stockcode = pd.read_excel(add_data+'stock_code.xlsx')
    # 首次戴帽时间
    stockcode['dmzm_date'] = stockcode['dmzm_date'].apply(lambda x:str(x)[-8:-4]
                               +'-'+str(x)[-4:-2]+'-'+str(x)[-2:] if not str(x)=='nan' else x)
    # 剔除B股等
    stockcode = stockcode[stockcode['code'].apply(lambda x:x[:2]).isin(['00','30','60'])]
    #stockcode['zhaipai_date'] = stockcode['zhaipai_date'].fillna(datetime.datetime.now().strftime('%Y-%m-%d')).apply(lambda x:str(x)[:10]) 
    # 最新取到2018-07-31
    stockcode['zhaipai_date'] = stockcode['zhaipai_date'].fillna('2018-08-01').apply(lambda x:str(x)[:10]) 
    stockcode.reset_index(drop=True,inplace=True)
    # 设置enddate
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
    
    stockcode['start_date'] = np.nan
    for j in range(len(stockcode['market_date'])):
        if stockcode.loc[j,'market_date']<'2009-01-01':
            stockcode.loc[j,'start_date'] = '2009-01-01'
        else:
            stockcode.loc[j,'start_date'] = stockcode.loc[j,'market_date']
    stockcode.sort_values(by='code',inplace=True)
#    stockcode.to_csv(add_data+'stock_code.csv')
#    #保存日期
#    stockcode.start_date = stockcode.start_date.apply(lambda x:dpc.DateProcess(x).format_date())
#    stockcode.enddate = stockcode.enddate.apply(lambda x:dpc.DateProcess(x).format_date())
#    np.save(add_data+'prepared_data/start_date',stockcode.start_date)
#    np.save(add_data+'prepared_data/end_date',stockcode.enddate)

    #剔除上市6个月之内的新股
    stock_info = pd.read_csv(add_data+'stock_code_spare2.csv')
    stock_info['market_date_defer6m']=np.nan
    for si in stock_info.index:
        d = dpc.DateProcess(stock_info.loc[si,'market_date'])
        stock_info.loc[si,'market_date_defer6m'] = d.tmonths_offset(6)
    # start_date推迟6个月
    stock_info['start_date_defer6m'] = np.nan
    for j in range(len(stock_info['market_date_defer6m'])):
        if stock_info.loc[j,'market_date_defer6m']<'2009-01-01':
            stock_info.loc[j,'start_date_defer6m'] = '2009-01-01'
        else:
            stock_info.loc[j,'start_date_defer6m'] = stock_info.loc[j,'market_date_defer6m']
    stock_info.to_csv(add_data+'stock_code.csv')

    #保存日期,保存的是3225个股票的日期，没有删除次新股
#    stock_info.start_date_defer6m = stock_info.start_date_defer6m.apply(lambda x:dpc.DateProcess(x).format_date())
#    stock_info.enddate = stock_info.enddate.apply(lambda x:dpc.DateProcess(x).format_date())
#    np.save(add_data+'prepared_data/del_cixin/stock_tdate_start',stock_info.start_date_defer6m)
#    np.save(add_data+'prepared_data/del_cixin/stock_tdate_end',stock_info.enddate)

if __name__=='__main__':
    get_stock_startdate_enddate()














