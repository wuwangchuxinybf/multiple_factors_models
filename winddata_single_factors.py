# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:14:40 2018 

@author: wuwangchuxin
"""
import pandas as pd
import WindPy as wp
import os
os.chdir('C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/')

stock_info = pd.read_csv('stock_code.csv')
#stock_info = stock_info.iloc[:54,:]
wp.w.start()

res = pd.DataFrame()
#factors = "pe_ttm" # pe
#factors = "pct_chg" #涨跌幅
#factors = "val_floatmv" #自由流通市值
#factors = "pb_lf" #市净率（lf）
factors = "ps_ttm" #市销率

suc=[]
unsuc = []
for j_code in stock_info['code']:
    j_begine_date = stock_info[stock_info['code'] == j_code].reset_index()['market_date'][0]
    j_end_date = stock_info[stock_info['code'] == j_code].reset_index()['enddate'][0]
    fin_result=wp.w.wsd(j_code, factors, j_begine_date, j_end_date, "Period=M;PriceAdj=F")
    if fin_result.ErrorCode == 0:
        data_df = pd.DataFrame(fin_result.Data,columns=fin_result.Times,index=[j_code])
        res = res.append(data_df,ignore_index=False)
        suc.append(j_code)
        print (j_code,'done')
    else:
        unsuc.append(j_code)
        print (j_code,fin_result.ErrorCode)
#res.to_csv('pettm_month_data.csv')
#res.to_csv('float_market_value.csv')
#res.to_csv('pb_lf_data.csv')
res.to_csv('psttm_month_data.csv')

#fp = open('evaluation.csv', 'w',newline='')
#writer = csv.writer(fp)

#        for k in range(0,len(fin_result.Times)):
#            temp = []
#            temp.append(fin_result.Codes)   
#            temp.append(fin_result.Times[k])      
#            for g in range(0, len(fin_result.Fields)):
#                temp.append(fin_result.Data[g][k])
#            writer.writerow(temp)     
#fp.close()
