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
stock_info.sort_values(by='code',inplace=True)
stock_info.reset_index(drop=True,inplace=True)

wp.w.start()

###总体指标#####################################################################
val_floatmv, #自由流通市值，前复权  ### ？？

###估值维度：估值因子############################################################
#指标：市盈率,市净率,市销率,企业价值倍数,股息率
pe_ttm,# 市盈率，后复权
#val_pe_deducted_ttm, #市盈率，扣除非经常损益，plus
#val_petohist20, # 过去一个月PE的均值，plus
#val_petohist60, # 过去三个月PE的均值，plus
#val_petohist120, # 过去6个月PE的均值，plus
#val_petohist250, # 过去一年PE的均值，plus
##pb_mrq, # 最新季报市净率,后复权，wind没有数据
#根据字面意思下面的指标由于指定日股本或财务数据在数据提取日之后可能存在上市公司披露变更情形
pb_lf,# 暂时用的这个pb指标，前复权 ### ？？
ps_ttm, # 市销率，前复权,SP=销售收入/市值 ### ？？
pcf_ocf_ttm, #市现率PCF（经营现金流TTM），后复权,plus
dividendyield2, #股息率（近12个月），后复权
#val_ortomv_ttm, #营收市值比，后复权，plus，=营业收入/市值,和ps类似，废弃,
# 营业收入 = 主营业务收入（也叫销售收入）+其他业务收入
val_evtoebitda2 #企业倍数，后复权
###质量维度一：盈利能力##########################################################
#指标：ROA,ROE,ROIC,长期平均资产回报率,长期平均资本回报率,毛利增长率,平均毛利率/毛利率标准差 #######
fa_roenp_ttm  净资产收益率（TTM），后复权  
#归属于母公司的净利润(TTM)／归属于母公司的股东权益(MRQ)*100%
##fa_roe_ttm #权益回报率， 净利润(TTM)*2/(本期股东权益(MRQ)+上年同期股东权益(MRQ))*100
fa_roaebit_ttm 总资产收益率（TTM），后复权
fa_roicebit_ttm，投入资本回报率ROIC(TTM),后复权
fa_gpmgr_ttm,增长率_毛利率(TTM),后复权
fa_grossprofitmargin_ttm,销售毛利率(TTM),后复权
#fa_roaavg_5y,5年平均资产回报率，不支持的指标！！！
fa_orgr_ttm,增长率_营业收入(TTM),后复权,plus
fa_npgr_ttm,增长率_净利润(TTM),plus ？？
fa_cfogr_ttm,增长率_经营活动产生的现金流量净额(TTM),plus ？？
###质量维度二：经营效率##########################################################
#指标：RNOA,ATO,经营效率边际变化,存货周转率 ##############
fa_invturn_ttm,存货周转率(TTM)，后复权
fa_arturn_ttm,应收账款周转率(TTM)，后复权，plus
#ATO,经营资产周转率,=营业收入/平均净经营资产，没有，曲线救国 ？？
fa_taturn_ttm,总资产周转率(TTM)，后复权
#fa_naturn_ttm,净资产周转率(TTM)

###质量维度三：盈余质量##########################################################
#指标：应计利润 经营性净资产 #####################
fa_profittomv_ttm,收益市值比(TTM),后复权,plus
#应计利润=净利润-经营性净现金流量
# 先不实用绝对值
#fa_operactcashflow_ttm,经营活动现金净流量(TTM) ？？
#fa_profit_ttm,净利润(TTM) ？？
#fa_pttomvavg5y,5年平均收益市值比,plus
#fa_pbttoor_ttm,利润总额/营业收入(TTM),plus 和毛利率差不多，剔除

###质量维度四：投融资决策########################################################
#指标：总资产增长 fama五因子模型 #############
fa_tagr,增长率-总资产 ,后复权
#投资决策
#融资决策
#总资产增量效应

###质量维度五：无形资产##########################################################
#指标：研发支出、创新效率、无形资产 #############
stmnote_RDexptosales,研发支出总额占营业收入比例,后复权

###添加维度：杠杆因子############################################################
fa_debttoasset,资产负债率,plus
fa_current,流动比率,plus

###利用市场信号修正策略##########################################################
#市场参与者行为
#指标：分析师投资评级修正、盈余预测修正、净买入比率
west_netprofit_fy1_6m,一致预测净利润（FY1）的变化率_6M，后复权
#west_sales_fy1_6m,一致预测营业收入的变化率
#west_freturn,一致预测目标价上升空间
#holder_sumsqupcttop10,前十大股东持股比例平方之和

#市场价格信号
#指标：1个月动量、3个月动量、6个月动量
#risk_variance120,120日收益方差
tech_revs60,过去3个月的价格动量,后复权
#tech_turnoverrate60,60日平均换手率

#市场情绪信号
#指标：封闭式基金折价率、共同基金净买入、市场中的总投资额、BW指标
#tech_bullpower,多头力道
#tech_cyf,市场能量指标
tech_rstr12,12月相对强势




#factors = "pe_ttm" # pe  
#factors = "pct_chg" #涨跌幅  
#factors = "val_floatmv" #自由流通市值 
#factors = "pb_lf" #市净率（lf）  ##作废
#factors = "ps_ttm" #市销率
#factors = "fa_roenp_ttm" #净资产收益率（TTM） 
#factors = "fa_roaebit_ttm" #总资产收益率（TTM）
#factors = "fa_ebitda_ttm" 
#factors = "val_evtoebitda2"
#factors = "pcf_ocf_ttm"
#factors = "pb_mrq"
#factors = "val_ortomv_ttm"
#factors = "dividendyield2"
#factors = "fa_roicebit_ttm"
#factors = "fa_gpmgr_ttm"
#factors = "fa_roaavg_5y"  不支持
#factors = "fa_invturn_ttm"
#factors = "fa_orgr_ttm"
#factors = "fa_profittomv_ttm"
#factors = "pe_ttm"
#factors = "ps_ttm"
#factors = "fa_grossprofitmargin_ttm" 
#factors = "fa_taturn_ttm"
#factors = "fa_tagr" 
#factors = "stmnote_RDexptosales"
#factors = "west_netprofit_fy1_6m"
#factors = "fa_arturn_ttm"
factors = "tech_revs60"

res = pd.DataFrame()
suc=[]
unsuc = []
for j_code in stock_info['code']:
    j_begine_date = stock_info[stock_info['code'] == j_code].reset_index()['market_date'][0]
    j_end_date = stock_info[stock_info['code'] == j_code].reset_index()['enddate'][0]
    # 之前采用前复权数据，现在一律改用后复权
    fin_result=wp.w.wsd(j_code, factors, j_begine_date, j_end_date, "Period=M;PriceAdj=B") #PriceAdj=F
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
#res.to_csv('psttm_month_data.csv')
#res.to_csv('fa_roenp_ttm.csv')
#res.to_csv('fa_roaebit_ttm.csv')
#res.to_csv('fa_ebitda_ttm.csv')
#res.to_csv('val_evtoebitda2.csv')
#res.to_csv('pcf_ocf_ttm_part1.csv')
#res.to_csv('pcf_ocf_ttm_part2.csv')
#res.to_csv('pb_mrq.csv')
#res.to_csv('val_ortomv_ttm_part1.csv')
#res.to_csv('val_ortomv_ttm_part2.csv')
#res.to_csv('dividendyield2.csv')
#res.to_csv('fa_roicebit_ttm_part1.csv')
#res.to_csv('fa_roicebit_ttm_part2.csv')
#res.to_csv('fa_invturn_ttm_part1.csv')
#res.to_csv('fa_invturn_ttm_part2.csv')
#res.to_csv('fa_orgr_ttm.csv')
#res.to_csv('fa_profittomv_ttm.csv')
#res.to_csv('pe_ttm.csv')
#res.to_csv('fa_grossprofitmargin_ttm.csv')
#res.to_csv('fa_taturn_ttm.csv')
#res.to_csv('fa_tagr_part1.csv')
#res.to_csv('stmnote_RDexptosales.csv')
#res.to_csv('west_netprofit_fy1_6m_part1.csv')
#res.to_csv('fa_arturn_ttm_part1.csv')
res.to_csv('tech_revs60.csv')


#添加漏掉的报错的
#stock_info = pd.read_csv('stock_code.csv')
#L_tmp = ['000929.SZ','600835.SH']

#np.save(add_ready+'unsuc_fa_arturn3',np.array(unsuc))
#import numpy as np
#L_tmp = np.load(add_ready+'unsuc_fa_arturn3.npy')
#factors = "fa_arturn_ttm"
#
#res = pd.DataFrame()
#suc=[]
#unsuc = []
#for j_code in L_tmp:
#    j_begine_date = stock_info[stock_info['code'] == j_code].reset_index()['market_date'][0]
#    j_end_date = stock_info[stock_info['code'] == j_code].reset_index()['enddate'][0]
#    # 之前采用前复权数据，现在一律改用后复权
#    fin_result=wp.w.wsd(j_code, factors, j_begine_date, j_end_date, "Period=M;PriceAdj=B") #PriceAdj=F
#    if fin_result.ErrorCode == 0:
#        data_df = pd.DataFrame(fin_result.Data,columns=fin_result.Times,index=[j_code])
#        res = res.append(data_df,ignore_index=False)
#        suc.append(j_code)
#        print (j_code,'done')
#    else:
#        unsuc.append(j_code)
#        print (j_code,fin_result.ErrorCode)
#res.to_csv('fa_arturn_ttm_part4.csv')