# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 14:49:34 2018
    
@author: wuwangchuxin
"""

import numpy as np
import pandas as pd
import os
os.chdir('D:/multiple_factors_models/')
import date_process_class as dpc

add_winddata = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/wind/'
add_ready = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/prepared_data/'
add_pic = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/20180223report/'

# df 为wind导出的月度数据（index为stockcode，columns为month end tradedate+其它中间tradedate）
# 剔除掉中间tradedate的列
def Drop_nan_columns(df):
    date_se = pd.Series(df.columns)
    date_max = date_se.groupby(date_se.apply(lambda x:x[:7])).max() #按月分组取最大日期
    return df.loc[:,df.columns.isin(date_max)]

# 所有季度财务因子数据
financial_data = pd.read_csv(add_winddata+'stock_financial_data_0_868.csv')
financial_data2 = pd.read_csv(add_winddata+'stock_financial_data_869_2200.csv')
financial_data3 = pd.read_csv(add_winddata+'stock_financial_data_2201_end.csv')
financial_data_all = financial_data.append(financial_data2,ignore_index=False).append(financial_data3,ignore_index=False)
financial_data_all.DATE = financial_data_all.DATE.apply(lambda x :dpc.DateProcess(x).format_date())
np.save(add_ready+'windfactors_financial_q',financial_data_all)

# pe因子
pettm_month = pd.read_csv(add_winddata+'pettm_month_data.csv',index_col=0)
pettm_month = Drop_nan_columns(pettm_month)
pettm_month.columns = map(lambda x:dpc.DateProcess(x).format_date(),list(pettm_month.columns))
#保存原始pe值
np.save(add_ready+'windfactors_pe',np.array(pettm_month.values)) #pe值，3225*115
#取pe因子的倒数
data_ep = pettm_month.apply(lambda x:1./x)
np.save(add_ready+'windfactors_ep',np.array(data_ep.values)) #pe值，3225*115
np.save(add_ready+'stockscode',np.array(data_ep.index)) #股票代码,3225
np.save(add_ready+'month_end_tdate',np.array(data_ep.columns)) #交易日期,115

# stocks return per month
return_month1 = pd.read_csv(add_winddata+'return_month_plus.csv',index_col=0) #前54个股票
return_month1 = Drop_nan_columns(return_month1)
return_month2 = pd.read_csv(add_winddata+'return_month.csv',index_col=0) #剩下的股票
return_month2 = Drop_nan_columns(return_month2)
return_month = return_month1.append(return_month2)
return_month.columns = map(lambda x:dpc.DateProcess(x).format_date(),list(return_month.columns))
np.save(add_ready+'wind_return_month',return_month.values) #月收益率，3225*115

# 申万一级行业分类-28个行业
industry_sw1 = pd.read_excel(add_winddata+'industry_sw1_class.xlsx')

#industry_dict = {'交通运输':'JTYS','休闲服务':'XXFW','传媒':'CM','公用事业':'GYSY',
#                 '农林牧渔':'NLMY','化工':'HG','医药生物':'YYSW','商业贸易':'SYMY',
#                 '国防军工':'GFJG','家用电器':'JYDQ','建筑材料':'JZCL','建筑装饰':'JZZS',
#                 '房地产':'FDC','有色金属':'YSJS','机械设备':'JXSB','汽车':'QC',
#                 '电子':'DZ','电气设备':'DQSB','纺织服装':'FZFZ','综合':'ZH',
#                 '计算机':'JSJ','轻工制造':'QGZZ','通信':'TX','采掘':'CJ','钢铁':'GT',
#                 '银行':'YH','非银金融':'FYJR','食品饮料':'SPYL'}
## 名称替换为英文形式
#for i in np.arange(len(industry_sw1.industry_1class)):
#    industry_sw1.loc[i,'industry_1class'] = industry_dict[industry_sw1.loc[i,'industry_1class']]

industry_sw1['value']=1
#industry_sw1['industry_1class'] = industry_sw1['industry_1class'].replace(0,'综合')
industry_sw1 = industry_sw1.pivot('code','industry_1class')
industry_sw1.fillna(0,inplace=True)
industry_sw1.columns = list(map(lambda x:x[1],industry_sw1.columns))
np.save(add_ready+'industry_sw1',industry_sw1.values) #行业虚拟变量矩阵
np.save(add_ready+'industry_sw1_name',np.array(industry_sw1.columns)) #行业名称

#流通市值
float_mv = pd.read_csv(add_winddata+'float_market_value.csv',index_col=0)
float_mv = Drop_nan_columns(float_mv)
float_mv.columns = map(lambda x:dpc.DateProcess(x).format_date(),list(float_mv.columns))
np.save(add_ready+'wind_float_mv',np.array(float_mv.values))
    
# pb因子
pb_lf_month = pd.read_csv(add_winddata+'pb_lf_data.csv',index_col=0)
pb_lf_month = Drop_nan_columns(pb_lf_month)
pb_lf_month.columns = map(lambda x:dpc.DateProcess(x).format_date(),list(pb_lf_month.columns))
#保存原始pb值
np.save(add_ready+'windfactors_pb',np.array(pb_lf_month.values)) #pb值，3225*115
#取pb因子的倒数
data_bp = pb_lf_month.apply(lambda x:1./x)
np.save(add_ready+'windfactors_bp',np.array(data_bp.values)) #bp值，3225*115

# ps因子
psttm_month = pd.read_csv(add_winddata+'psttm_month_data.csv',index_col=0)
psttm_month = Drop_nan_columns(psttm_month)
psttm_month.columns = map(lambda x:dpc.DateProcess(x).format_date(),list(psttm_month.columns))
#保存原始ps值
np.save(add_ready+'windfactors_ps',np.array(psttm_month.values)) #pb值，3225*115
#取ps因子的倒数
data_sp = psttm_month.apply(lambda x:1./x)
np.save(add_ready+'windfactors_sp',np.array(data_sp.values)) #pe值，3225*115




