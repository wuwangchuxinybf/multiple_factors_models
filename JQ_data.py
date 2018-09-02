# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
#import datetime
import os
os.chdir('D:/multiple_factors_models/')
import date_process_class as dpc

add_data = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/'
add_winddata = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/wind/'
add_ready = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/prepared_data/'
add_pic = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/20180830report/'

from jqdatasdk import *
#auth("13699278766", "278766") #shifan
#auth("13683552188", "552188") #chenzhenxing
auth("13249882580", "882580") #huhuimin
#auth("18201159007", "B6486082")
#auth("17701255381","198261")#群截图

#factors= "pe_ttm,val_pe_deducted_ttm,pb_lf,ps_ttm,pcf_ncf_ttm,pcf_ocf_ttm,\
#             fcff,mkt_cap_ard,dividendyield2,ev2_to_ebitda,profit_ttm"  #字段
             
#q = query(valuation).filter(valuation.code == '000001.XSHE')
#df = get_fundamentals(q, '2015-10-15')

stock_code = pd.read_csv(add_data+'JQ_stock_code.csv')
stock_code.market_date = stock_code.market_date.apply(lambda x:dpc.DateProcess(x).format_date())
stock_code.enddate = stock_code.enddate.apply(lambda x:dpc.DateProcess(x).format_date())

#trade_date = pd.read_excel(add_data+'market_tradedate.xlsx')
#trade_date_list = trade_date['date'].apply(lambda x:str(x)[:10])

#stock_code = stock_code.iloc[2558:,:]
#stock_code.reset_index(drop=True,inplace=True)

res=pd.DataFrame()
for n in range(len(stock_code)):
    if stock_code.loc[n,'code'][-2:] == 'SH':
        sname = stock_code.loc[n,'code'][:-2]+'XSHG'
    elif stock_code.loc[n,'code'][-2:] == 'SZ':
        sname = stock_code.loc[n,'code'][:-2]+'XSHE'
# 财务数据
#    q = query(
##            valuation.code,
##            valuation.day,
#            valuation.pe_ratio,
#            valuation.pb_ratio,
#            valuation.ps_ratio,
#            valuation.pcf_ratio,  
#            ).filter(valuation.code == sname)
        
# 行情数据
    df = get_price(sname, start_date=stock_code.loc[n,'market_date'], \
                   end_date=stock_code.loc[n,'enddate'],frequency='daily', \
                   fields=None, skip_paused=True, fq='pre')
    finance.run_query(query(finance.STK_EXCHANGE_TRADE_INFO).filter\
                      (finance.STK_EXCHANGE_TRADE_INFO.exchange_code==exchange_code).limit(n)
    df['code'] = stock_code.loc[n,'code']
    res = res.append(df)
    print (sname+' done')
res.to_csv(add_data+'Qdata_MarketDaily_end.csv')

#w.wsd("300150.SZ", "pct_chg", "2009-01-01", "2018-07-31", "Period=M;PriceAdj=F")





# 指数权重历史数据
trade_date = np.load(add_ready+'month_end_tdate.npy').reshape(1,-1) #月末交易日
for index_id in ['000905.XSHG']: #'000300.XSHG',
    res = pd.DataFrame()
    for tdate in trade_date[0,:][::-1]:
        if tdate=='2018-07-31':
            res = get_index_weights(index_id, date=tdate)
            print (res.shape,'first done')
        else:
            res = res.append(get_index_weights(index_id, date=tdate))
        print (tdate,'done')
    res.reset_index(inplace=True)
    res['code'] = res['code'].apply(lambda x: x[:-4]+'SH' if x[-4:]=='XSHG' else x[:-4]+'SZ')
    res.to_csv(add_data+'index_weights_%s.csv'%index_id[:-5])

#申万一级行业分类成分股
get_industry_stocks(industry_code, date=None)

#指数行业权重

#len(set(res.date))
#
#
#tmp = pd.read_csv(add_data+'index_weights_000300.csv')
#
#
#len(set(tmp.date))
#
#
#tmp.date = tmp.date.apply(lambda x: dpc.DateProcess(x).format_date())


