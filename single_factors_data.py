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
add_jqdata = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/jqdata/'
add_ready = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/prepared_data/'
add_pic = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/20180223report/'

# df 为wind导出的月度数据（index为stockcode，columns为month end tradedate+其它中间tradedate）
# 剔除掉中间tradedate的列
def Drop_nan_columns(df):
    df.columns = [dpc.DateProcess(x).format_date() for x in df.columns]
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
#pettm_month = pd.read_csv(add_winddata+'pettm_month_data.csv',index_col=0)
#pettm_month = Drop_nan_columns(pettm_month)
#pettm_month.sort_index(inplace=True)
pe_ttm = pd.read_csv(add_winddata+'pe_ttm.csv',index_col=0)
pe_ttm = Drop_nan_columns(pe_ttm)
pe_ttm.sort_index(inplace=True)

#保存原始pe值
np.save(add_ready+'windfactors_pe',np.array(pe_ttm.values)) #pe值，3225*115
#取pe因子的倒数
data_ep = pe_ttm.apply(lambda x:1./x)
np.save(add_ready+'windfactors_ep',np.array(data_ep.values)) #pe值，3225*115
np.save(add_ready+'stockscode',np.array(data_ep.index)) #股票代码,3225
np.save(add_ready+'month_end_tdate',np.array(data_ep.columns)) #交易日期,115

# stocks return per month
return_month1 = pd.read_csv(add_winddata+'return_month_plus.csv',index_col=0) #前54个股票
return_month1 = Drop_nan_columns(return_month1)
return_month2 = pd.read_csv(add_winddata+'return_month.csv',index_col=0) #剩下的股票
return_month2 = Drop_nan_columns(return_month2)
return_month = return_month1.append(return_month2)
return_month.sort_index(inplace=True)
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
industry_sw1.sort_index(inplace=True)
np.save(add_ready+'industry_sw1',industry_sw1.values) #行业虚拟变量矩阵
np.save(add_ready+'industry_sw1_name',np.array(industry_sw1.columns)) #行业名称

#流通市值
float_mv = pd.read_csv(add_winddata+'float_market_value.csv',index_col=0)
float_mv = Drop_nan_columns(float_mv)
float_mv.sort_index(inplace=True)
np.save(add_ready+'wind_float_mv',np.array(float_mv.values))
    
# pb因子
#pb_mrq = pd.read_csv(add_winddata+'pb_mrq.csv',index_col=0)
#pb_mrq = Drop_nan_columns(pb_mrq)
#pb_mrq.sort_index(inplace=True)

pb_lf_data = pd.read_csv(add_winddata+'pb_lf_data.csv',index_col=0)
pb_lf_data = Drop_nan_columns(pb_lf_data)
pb_lf_data.sort_index(inplace=True)

#保存原始pb值
np.save(add_ready+'windfactors_pb',np.array(pb_lf_data.values)) #pb值，3225*115
#取pb因子的倒数
data_bp = pb_lf_data.apply(lambda x:1./x)
np.save(add_ready+'windfactors_bp',np.array(data_bp.values)) #bp值，3225*115

# ps因子
psttm_month = pd.read_csv(add_winddata+'psttm_month_data.csv',index_col=0)
psttm_month = Drop_nan_columns(psttm_month)
psttm_month.sort_index(inplace=True)
#保存原始ps值
np.save(add_ready+'windfactors_ps',np.array(psttm_month.values)) #pb值，3225*115
#取ps因子的倒数
data_sp = psttm_month.apply(lambda x:1./x)
np.save(add_ready+'windfactors_sp',np.array(data_sp.values)) #pe值，3225*115


#沪深300指数成分股权重
weights_000300 = pd.read_csv(add_jqdata+'index_weights_000300.csv')
weights_000300.date = weights_000300.date.apply(lambda x: dpc.DateProcess(x).format_date())
weights_000300.rename(columns={'index':'code','display_name':'sname'},inplace=True)
weights_000300 = weights_000300[['code','date','weight']]

weights_000300_plus = pd.read_excel(add_jqdata+'index_weights_000300_plus.xlsx')
weights_000300_plus = weights_000300_plus[['code','date','weight']]
weights_000300_plus.date = weights_000300_plus.date.apply(lambda x:str(x)[:10])
weights_000300 = weights_000300_plus.append(weights_000300)
#wind缺失这两条数据，手动补上
weights_000300.loc[34498] = ['600001.SH','2009-12-31',0.028]
weights_000300.loc[34499] = ['600357.SH','2009-12-31',0.012]

weights_000300 = weights_000300.sort_values(by=['date','weight'],ascending=(True,False))
weights_000300.reset_index(drop=True,inplace=True)
weights_000300_arr=weights_000300.pivot('code','date')
np.save(add_ready+'weights_000300',np.array(weights_000300_arr.values))
np.save(add_ready+'weights_000300_stocklist',np.array(weights_000300_arr.index))
# 中证500个股权重
weights_000905 = pd.read_csv(add_jqdata+'index_weights_000905.csv')
weights_000905.date = weights_000905.date.apply(lambda x: dpc.DateProcess(x).format_date())
weights_000905 = weights_000905[['code','date','weight']]
weights_000905 = weights_000905.sort_values(by=['date','weight'],ascending=(True,False))
weights_000905.reset_index(drop=True,inplace=True)
weights_000905_arr=weights_000905.pivot('code','date')
np.save(add_ready+'weights_000905',np.array(weights_000905_arr.values))
np.save(add_ready+'weights_000905_stocklist',np.array(weights_000905_arr.index))

# 沪深300指数申万一级行业权重
#industry = pd.read_excel(add_winddata+'industry_sw1_class.xlsx')  #原始数据
#weights_000300['sw_1class']=np.nan
#no_list=[]
#for code in set(weights_000300['code']):
#    try:
#        weights_000300.loc[weights_000300['code']==code,'sw_1class']= \
#            industry.loc[industry['code']==code,'industry_1class'].values[0]
#    except:
#        no_list.append(code)
#for code in no_list:
#    industry[industry['code']==code]

hs300_sw_1class_weight = pd.read_excel(add_winddata+'hs300_sw_1class_weight.xlsx')
hs300_sw_1class_weight = hs300_sw_1class_weight[['industry_name','date','weight']]
hs300_sw_1class_weight['date'] = hs300_sw_1class_weight['date'].apply(lambda x:str(x)[:10])
hs300_sw_1class_weight_arr=hs300_sw_1class_weight.pivot('industry_name','date')
np.save(add_ready+'hs300_sw_1class_weight',np.array(hs300_sw_1class_weight_arr.values))
np.save(add_ready+'hs300_sw_1class_weight_industrynames',np.array(hs300_sw_1class_weight_arr.index))
#tmp = hs300_sw_1class_weight.groupby('date').count()

# fa_roenp_ttm #净资产收益率（TTM）
fa_roenp_ttm = pd.read_csv(add_winddata+'fa_roenp_ttm.csv',index_col=0)
fa_roenp_ttm = Drop_nan_columns(fa_roenp_ttm)
fa_roenp_ttm.sort_index(inplace=True)
np.save(add_ready+'windfactors_roe',np.array(fa_roenp_ttm.values))

# fa_roaebit_ttm #总资产收益率（TTM）
fa_roaebit_ttm = pd.read_csv(add_winddata+'fa_roaebit_ttm.csv',index_col=0)
fa_roaebit_ttm = Drop_nan_columns(fa_roaebit_ttm)
fa_roaebit_ttm.sort_index(inplace=True)
np.save(add_ready+'windfactors_roaebit',np.array(fa_roaebit_ttm.values))

# fa_ebitda_ttm #息税折旧摊销前利润（TTM）
fa_ebitda_ttm = pd.read_csv(add_winddata+'fa_ebitda_ttm.csv',index_col=0)
fa_ebitda_ttm = Drop_nan_columns(fa_ebitda_ttm)
fa_ebitda_ttm.sort_index(inplace=True)
np.save(add_ready+'windfactors_ebitda',np.array(fa_ebitda_ttm.values))

# val_evtoebitda2 #企业倍数
val_evtoebitda2 = pd.read_csv(add_winddata+'val_evtoebitda2.csv',index_col=0)
val_evtoebitda2 = Drop_nan_columns(val_evtoebitda2)
val_evtoebitda2.sort_index(inplace=True)
np.save(add_ready+'windfactors_evtoebitda',np.array(val_evtoebitda2.values))


# pcf_ocf_ttm #市现率PCF（经营现金流TTM）
pcf_ocf_ttm_part1 = pd.read_csv(add_winddata+'pcf_ocf_ttm_part1.csv',index_col=0)
pcf_ocf_ttm_part1 = Drop_nan_columns(pcf_ocf_ttm_part1)
pcf_ocf_ttm_part2 = pd.read_csv(add_winddata+'pcf_ocf_ttm_part2.csv',index_col=0)
pcf_ocf_ttm_part2 = Drop_nan_columns(pcf_ocf_ttm_part2)
pcf_ocf_ttm = pcf_ocf_ttm_part1.append(pcf_ocf_ttm_part2)
#检查缺少哪些股票
#set(val_evtoebitda2.index).difference(set(pcf_ocf_ttm.index))
#set(pcf_ocf_ttm.index).difference(set(val_evtoebitda2.index))

pcf_ocf_ttm.sort_index(inplace=True)
np.save(add_ready+'windfactors_pcf_ocf',np.array(pcf_ocf_ttm.values))

# val_ortomv_ttm,#营收市值比
val_ortomv_ttm_part1 = pd.read_csv(add_winddata+'val_ortomv_ttm_part1.csv',index_col=0)
val_ortomv_ttm_part1 = Drop_nan_columns(val_ortomv_ttm_part1)
val_ortomv_ttm_part2 = pd.read_csv(add_winddata+'val_ortomv_ttm_part2.csv',index_col=0)
val_ortomv_ttm_part2 = Drop_nan_columns(val_ortomv_ttm_part2)
val_ortomv_ttm = val_ortomv_ttm_part1.append(val_ortomv_ttm_part2)
val_ortomv_ttm.sort_index(inplace=True)
np.save(add_ready+'windfactors_val_ortomv',np.array(val_ortomv_ttm.values))


# dividendyield2 ##股息率（近12个月），后复权
dividendyield2 = pd.read_csv(add_winddata+'dividendyield2.csv',index_col=0)
dividendyield2 = Drop_nan_columns(dividendyield2)
dividendyield2.sort_index(inplace=True)
#set(stock_info['code']).difference(set(dividendyield2.index))
#set(dividendyield2.index).difference(set(stock_info['code']))
np.save(add_ready+'windfactors_dividendyield',np.array(dividendyield2.values))


# fa_roicebit_ttm,#投入资本回报率ROIC(TTM)
fa_roicebit_ttm_part1 = pd.read_csv(add_winddata+'fa_roicebit_ttm_part1.csv',index_col=0)
fa_roicebit_ttm_part1 = Drop_nan_columns(fa_roicebit_ttm_part1)
fa_roicebit_ttm_part2 = pd.read_csv(add_winddata+'fa_roicebit_ttm_part2.csv',index_col=0)
fa_roicebit_ttm_part2 = Drop_nan_columns(fa_roicebit_ttm_part2)
fa_roicebit_ttm = fa_roicebit_ttm_part1.append(fa_roicebit_ttm_part2)
fa_roicebit_ttm.sort_index(inplace=True)
np.save(add_ready+'windfactors_roicebit',np.array(fa_roicebit_ttm.values))

#fa_gpmgr_ttm,增长率_毛利率(TTM)
fa_gpmgr_ttm = pd.read_csv(add_winddata+'fa_gpmgr_ttm.csv',index_col=0)
fa_gpmgr_ttm = Drop_nan_columns(fa_gpmgr_ttm)
fa_gpmgr_ttm.sort_index(inplace=True)
np.save(add_ready+'windfactors_gpmgr',np.array(fa_gpmgr_ttm.values))

#fa_invturn_ttm,存货周转率(TTM)
fa_invturn_ttm_part1 = pd.read_csv(add_winddata+'fa_invturn_ttm_part1.csv',index_col=0)
fa_invturn_ttm_part1 = Drop_nan_columns(fa_invturn_ttm_part1)
fa_invturn_ttm_part2 = pd.read_csv(add_winddata+'fa_invturn_ttm_part2.csv',index_col=0)
fa_invturn_ttm_part2 = Drop_nan_columns(fa_invturn_ttm_part2)
fa_invturn_ttm = fa_invturn_ttm_part1.append(fa_invturn_ttm_part2)
fa_invturn_ttm.sort_index(inplace=True)
np.save(add_ready+'windfactors_invturn',np.array(fa_invturn_ttm.values))


#fa_orgr_ttm,增长率_营业收入(TTM)
fa_orgr_ttm = pd.read_csv(add_winddata+'fa_orgr_ttm.csv',index_col=0)
fa_orgr_ttm = Drop_nan_columns(fa_orgr_ttm)
fa_orgr_ttm.sort_index(inplace=True)
np.save(add_ready+'windfactors_orgr',np.array(fa_orgr_ttm.values))


#fa_profittomv_ttm,收益市值比(TTM)
fa_profittomv_ttm = pd.read_csv(add_winddata+'fa_profittomv_ttm.csv',index_col=0)
fa_profittomv_ttm = Drop_nan_columns(fa_profittomv_ttm)
fa_profittomv_ttm.sort_index(inplace=True)
np.save(add_ready+'windfactors_profittomv',np.array(fa_profittomv_ttm.values))

#fa_grossprofitmargin_ttm,销售毛利率(TTM)
fa_grossprofitmargin_ttm = pd.read_csv(add_winddata+'fa_grossprofitmargin_ttm.csv',index_col=0)
fa_grossprofitmargin_ttm = Drop_nan_columns(fa_grossprofitmargin_ttm)
fa_grossprofitmargin_ttm.sort_index(inplace=True)
np.save(add_ready+'windfactors_grossprofitmargin',np.array(fa_grossprofitmargin_ttm.values))

#fa_taturn_ttm,总资产周转率(TTM)
fa_taturn_ttm = pd.read_csv(add_winddata+'fa_taturn_ttm.csv',index_col=0)
fa_taturn_ttm = Drop_nan_columns(fa_taturn_ttm)
fa_taturn_ttm.sort_index(inplace=True)
np.save(add_ready+'windfactors_taturn',np.array(fa_taturn_ttm.values))

#fa_tagr,增长率-总资产
fa_tagr_part1 = pd.read_csv(add_winddata+'fa_tagr_part1.csv',index_col=0)
fa_tagr_part1 = Drop_nan_columns(fa_tagr_part1)
fa_tagr_part2 = pd.read_csv(add_winddata+'fa_tagr_part2.csv',index_col=0)
fa_tagr_part2 = Drop_nan_columns(fa_tagr_part2)
fa_tagr = fa_tagr_part1.append(fa_tagr_part2)
fa_tagr.sort_index(inplace=True)
np.save(add_ready+'windfactors_tagr',np.array(fa_tagr.values))

#stmnote_RDexptosales,研发支出总额占营业收入比例 好像没数据
stmnote_RDexptosales = pd.read_csv(add_winddata+'stmnote_RDexptosales.csv',index_col=0)
stmnote_RDexptosales = Drop_nan_columns(stmnote_RDexptosales)
stmnote_RDexptosales.sort_index(inplace=True)
np.save(add_ready+'windfactors_stmnote_RDexptosales',np.array(stmnote_RDexptosales.values))

#west_netprofit_fy1_6m,一致预测净利润（FY1）的变化率_6M
west_netprofit_fy1_6m_part1 = pd.read_csv(add_winddata+'west_netprofit_fy1_6m_part1.csv',index_col=0)
west_netprofit_fy1_6m_part1 = Drop_nan_columns(west_netprofit_fy1_6m_part1)
west_netprofit_fy1_6m_part2 = pd.read_csv(add_winddata+'west_netprofit_fy1_6m_part2.csv',index_col=0)
west_netprofit_fy1_6m_part2 = Drop_nan_columns(west_netprofit_fy1_6m_part2)
west_netprofit_fy1_6m = west_netprofit_fy1_6m_part1.append(west_netprofit_fy1_6m_part2)
west_netprofit_fy1_6m.sort_index(inplace=True)
np.save(add_ready+'windfactors_tagr',np.array(west_netprofit_fy1_6m.values))

#fa_arturn_ttm,应收账款周转率(TTM)
fa_arturn_ttm_part1 = pd.read_csv(add_winddata+'fa_arturn_ttm_part1.csv',index_col=0)
fa_arturn_ttm_part1 = Drop_nan_columns(fa_arturn_ttm_part1)
fa_arturn_ttm_part2 = pd.read_csv(add_winddata+'fa_arturn_ttm_part2.csv',index_col=0)
fa_arturn_ttm_part2 = Drop_nan_columns(fa_arturn_ttm_part2)
fa_arturn_ttm_part3 = pd.read_csv(add_winddata+'fa_arturn_ttm_part3.csv',index_col=0)
fa_arturn_ttm_part3 = Drop_nan_columns(fa_arturn_ttm_part3)
fa_arturn_ttm_part4 = pd.read_csv(add_winddata+'fa_arturn_ttm_part4.csv',index_col=0)
fa_arturn_ttm_part4 = Drop_nan_columns(fa_arturn_ttm_part4)

fa_arturn_ttm = fa_arturn_ttm_part1.append(fa_arturn_ttm_part2).append(fa_arturn_ttm_part3).append(fa_arturn_ttm_part4)
fa_arturn_ttm.sort_index(inplace=True)
np.save(add_ready+'windfactors_arturn',np.array(fa_arturn_ttm.values))

#
#return_month = np.load(add_ready+'wind_return_month.npy')/100+1
#return_month_roll3 = np.roll(return_month,3)
#return_month_roll2 = np.roll(return_month,2)
#return_month_roll1 = np.roll(return_month,1)
#momemtum_3m = return_month_roll1*return_month_roll2*return_month_roll3-1


