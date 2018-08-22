# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 11:14:40 2018 

@author: wuwangchuxin
"""
#import csv

#res = wp.w.wsd("000651.SZ", , "2017-06-01", "2018-07-20", "unit=1;PriceAdj=F")
############华泰研报
#factors_Value= "pe_ttm,val_pe_deducted_ttm,pb_lf,ps_ttm,pcf_ncf_ttm,pcf_ocf_ttm,\
#             fcff,mkt_cap_ard,dividendyield2,ev2_to_ebitda,profit_ttm"  #字段
#factors_Growth = "oper_rev,or_ttm,net_profit_is,profit_ttm,net_cash_flows_oper_act,\
#                   operatecashflow_ttm,roe_diluted,tot_equity"
#factors_financial = "qfa_roe,qfa_roa,qfa_grossprofitmargin,nptocostexpense_qfa,roic,qfa_operateincometoebt,\
#                    qfa_deductedprofittoprofit,qfa_ocftosales,ocftocf_qfa,currentdebttodebt,current,assetsturn"
#factors_Leverage = "current,cashtocurrentdebt,wgsd_com_eq_paholder,wgsd_assets"
#factors_size = "mkt_freeshares"
#factors_Momentum = "return_1m,return_3m,return_6m,return_12m,wgt_return_1m,wgt_return_3m,wgt_return_6m,\
#                     wgt_return_12m,exp_wgt_return_1m,exp_wgt_return_3m,exp_wgt_return_6m,exp_wgt_return_12m"
#factors_Volatility = "stdevr"
#factors_Turnover = "turn"
#factors_Shareholder = "holder_sumsqupcttop5,holder_sumsqupcttop10"
#factors_zijinliuxiang = "mfd_buyamt_d,mfd_buyvol_d,mfd_buyord,mfd_buyamt_a,mfd_buyvol_a"

 东吴金工分享会因子
#基本面因子
资产类因子：资产：流动性资产，存货，固定资产；
           负债：短期负债，长期负债，总负债，应付账款；
           权益：资本，留存收益，总权益
 所有资产类因子，都需要除以总资产，以标准化
 重点：流动性、杠杆比例、存货、应付账款、留存收益

盈利类因子：主营业务利润、非主营业务利润、总利润、净利润    计算得到 毛利润
# 盈利水平：资产回报率=毛利润/总资产，反映总体资产带来的回报水平，不考虑杠杆的影响
#          权益回报率=毛利润/股东权益，反映股东投资的回报水平，会因为杠杆比例而放大
#          毛利率=毛利润/总收入，反映公司销售的利润率，利润率越高说明公司的议价能力越强，竞争优势越明显
# 盈利质量：（经营现金流-主营业务利润）/总资产，该指标值越大，盈利质量越好
#           公司的经营成果有两项：账面利润和现金流，
#           账面利润水平高并不代表公司经营好，上市公司可以通过会计操纵，虚增利润
#           如果现金流显著低于利润，会计报表被操纵的可能性越大，利润质量较低
#
# 盈利增长：同比和环比，但其实环比增长是没有意义的；增长使用的周期，有三个: 累计、单季、TTM
#          按因子的来源划分：资产类增长，盈利类增长，现金流类增长；
#            资产的增长主要看的是负债、存货、总资产，这三大资产的增长和未来股票收益率都是呈反比的。
#            收益也就是利润的增长有几个: 收入的增长、费用的增长、利润的增长。收入和利润增长通常是比较好的，
#            费用通常是不好的。费用又分为三大费用。总的来说，收入要看总收入，利润要看毛利润，费用要看三大费用。
#            现金流，有三大现金流，经营现金流、投资现金流、融资现金流。 
#            对于经营现金流，我们希望它为正，希望它增长，且越高越好。那么投资现金流呢? 
#            一家公司的投资在增长是好还是不好?这个不能明显确定，但实证表明通常是不好。
#             如果说投资是因为一家公司有很好的增长机会，那么这个时候，投资就是比较好的。
#             融资现金流。融资现金流越多，公司未来股票收益率是越低还是越高? 实证表明答案是越低。
#          按增长的计算方式划分：同比增长，环比增长
#          按增长的时间跨度划分：累计增长，单季增长，TTM增长
##金融市场类因子
#股票是可交易的品种，可交易的属性会给它带来价值，我们就要去分析它作为一个金融产品的特征，
#比如: 估值、收益率、流动性、波动性。
#估值类：市盈率、市净率、PEG，PWC，估值越高，未来股票收益率越低
#   可以用它公司自身的指标，也可以做行业调整。但是我并不建议做行业调整，因为我觉得市盈率是一个绝对的判断，
#   当用行业标准看的时候，就不是那么可观了，所以我自己并不喜欢做行业的调整，当然也可以做，
#   主要是看做了行业调整以后，预测效果会不会好一些。
#  收益率不用太讲究，一般都用过往收益率，回看过去一到六周。过往收益率和未来股票收益率通常呈反比，意味着反转现象。
#流动性：成交金额，换手率，Amihud，流动性和未来股票收益成反比，通常会采取对数的形式，让流动性指标更加正态分布
# 主要看三个指标:交易量/交易金额、换手率、Amihud
# Amihud，日度收益率绝对值/日度成交金额，代表单位交易金额产生的股票的绝对收益率，或者每单位交易金额能够带来股价的变动
#  Amihud 是从根本上对流动性进行衡量，是一个 非常重要的指标。
#波动性：衡量股票波动性最重要的指标是标准差，通常标准差怎么算? 需要周度收益标准差还是月度? 
#       这里存在一个取舍问题。我们计算标准差的时候，至少需要60个数据，如果用周度数据，就会用到一年以前的数据，
#       但是一年前的数据与现在的相关性比较低; 那如果我们使用最近的日度数据，虽然提高了相关性，但又存在波动性太大的问题。
#       两相取舍用哪个? 答案是用日度的，主要是因为我们不想用那么长时间之前的数据，太久以前的数据根本反映不了现在的情况。
# 标准差计算：利用过去120个交易日的日度收益率数据；计算日度收益率的标准差，观察值不少于60个，超过120的一半
# 特质波动率：把股票收益率对市场收益率进行回归，计算残差的波动率
# 其它比如 β系数，上下行风险；
#   市场上升的时候，我们希望个股收益对市场收益的敏感度越高越好; 当市场下跌的时候，我们希望敏感度越低越好。
#
# Fama-MacBeth 回归，横截面回归，识别能够解释或预测未来股价回报率的因子，滞后回归
#   这里的回归一定是滞后回归，因为我们是在做预测，比如将这个周的收益对上个周末的因子值进行回归。
#   我们每周都进行一次回归，得到很多β系数，我们怎么判断这个因子到底好不好呢? 其实就是要对这个β系数进行T统计量的检验。
#   我们要求T统计量的绝对值一定要大于2.5，这样才足够显著，这个因子才能用。
#   我们还会关注一个正负的比例，我们首先要求正负比例和T值方向是一致的，比如T值是正的，
#   我们就会要求因子的β系数为正的比例更高一些，且阀值定一般定为 0.55
## Fama-MacBeth 回归注意事项：
#   回归前进行描述性统计，有效观察值比例要大于80%
#   T统计量的绝对值一定要大于2.5
#   若T统计量为正，则要求系数为正的比例要大于55%，反之亦然。
##多因素模型在选股中的应用
#1.打分法-区间法，分组法
#2.排序法-条件排序，无条件排序
#3.回归法：我们每周都做一个滞后回归，估计出因子的β系数，然后根据最新的因子值，计算出对下周收益率的预测值。
#  多重共线性
## 量化当中的风险控制
# 市场择时，控制回撤，模型失效，交易成本

import pandas as pd
import WindPy as wp
import os
os.chdir('C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/')

stock_info = pd.read_csv('stock_code.csv')
#stock_info = stock_info.iloc[2200:,:].reset_index(drop=True)
wp.w.start()
### 基本面量化投资
factors = "val_pe_deducted_ttm,pe_ttm,pb_lf,ps_ttm,pcf_ocf_ttm,dividendyield2,fa_ebitda_ttm,\
            mkt_cap_ard,tot_liab,fa_cce,fa_roenp_ttm,fa_roa_ttm,fa_roicebit_ttm,fa_taturn_ttm,\
            west_netprofit_fy1_6m,tech_revs5,tech_revs5m20,val_lnfloatmv"

res = pd.DataFrame(columns=['VAL_PE_DEDUCTED_TTM','PE_TTM','PB_LF','PS_TTM','PCF_OCF_TTM','DIVIDENDYIELD2',\
                            'FA_EBITDA_TTM','MKT_CAP_ARD','TOT_LIAB','FA_CCE','FA_ROENP_TTM',\
                            'FA_ROA_TTM','FA_ROICEBIT_TTM','FA_TATURN_TTM','WEST_NETPROFIT_FY1_6M',\
                             'TECH_REVS5','TECH_REVS5M20','VAL_LNFLOATMV','STOCK_CODE'])
suc=[]
unsuc = []
for j_code in stock_info['code']:
    j_begine_date = stock_info[stock_info['code'] == j_code].reset_index()['market_date'][0]
    j_end_date = stock_info[stock_info['code'] == j_code].reset_index()['enddate'][0]
    fin_result=wp.w.wsd(j_code, factors, j_begine_date, j_end_date, "unit=1;rptType=1;Period=Q;PriceAdj=F")
    if fin_result.ErrorCode == 0:
        data_df = pd.DataFrame(fin_result.Data,columns=fin_result.Times,index=fin_result.Fields).T
        data_df['STOCK_CODE'] = j_code
        res = res.append(data_df,ignore_index=False)
        suc.append(j_code)
        print (j_code,'done')
    else:
        unsuc.append(j_code)
        print (j_code,fin_result.ErrorCode)
res.to_csv('stock_financial_data.csv')


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
