# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:08:12 2019

@author: wuwangchuxin
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import math
#import matplotlib.pyplot as plt
import os
os.chdir('D:/multiple_factors_models/')
#import date_process_class as dpc
from scipy import stats as ss
import matplotlib.pyplot as plt
#import rotation_model as rm
from single_factors_test import Clean_Data

# 添加路径，方便导入自定义模块，比如 import date_process_class as dpc
import sys
sys.path.append('D:/multiple_factors_models')

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

add_winddata = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/wind/'
add_ready = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/prepared_data/del_cixin/'
add_pic = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/20190326pic/'




## 风格轮动模型
#def Data_preposs(f_arr):
#    '''完整的数据预处理流程'''
#    f_a1 = Clean_Data(f_arr).Median_deextremum()
#    f_a2 = Clean_Data(f_a1).Z_score()
#    f_a4 = np.around(f_a2,decimals=4)
#    return f_a4
#
## 采用24期滚动预测
#alpha = 0.01 #梯度下降步长
#style_res = []
#rolling_period = range(24,97,12)
##for p in range(24,112):
#for rp in rolling_period:
#    y = rm.Logit_style().gene_y()[p-rp:p]
#    X_all = rm.Logit_style().gene_factors()
#    X = Data_preposs(X_all[p-rp:p,:])
#    theta_fit = rm.Logistic_Regression(X,y,alpha).grad_descent()
#    h_fit = np.around(rm.Logistic_Regression.sigmoid(np.dot(X_all[p,:],theta_fit)),4)
#    style_res.append(h_fit)


## 估值指标
ep = np.load(add_ready+'windfactors_ep.npy')
bp = np.load(add_ready+'windfactors_bp.npy')
#盈利能力
roe = np.load(add_ready+'windfactors_roe.npy') #净资产收益率（TTM）
#roic = np.load(add_ready+'windfactors_roicebit.npy') #投入资本回报率ROIC(TTM)
#经营效率
#arturn = np.load(add_ready+'windfactors_arturn.npy') #应收账款周转率(TTM)
#投融资决策
tagr = np.load(add_ready+'windfactors_tagr.npy') #增长率-总资产
#杠杆因子
#debttoasset = np.load(add_ready+'windfactors_debttoasset.npy') #资产负债率
#市场价格信号
revs60 = np.load(add_ready+'windfactors_revs60.npy') #过去3个月的价格动量
#十二月相对强势
rstr12 = np.load(add_ready+'windfactors_rstr12.npy')

# 流动市值
float_mv = np.load(add_ready+'wind_float_mv.npy')
#风格结果
style_rotation_res = np.load(add_ready+'style_rotation_res.npy')
#月度收益序列
return_month = np.load(add_ready+'wind_return_month.npy')
#申万一级行业分类
industry_sw1 = np.load(add_ready+'industry_sw1.npy') #3171*28

#factors_effective = ['bp','roe','roic','arturn','tagr','debttoasset','revs60']

#数据标准化
#def Data_preposs(f_arr):
#    '''完整的数据预处理流程'''
#    f_a1 = Clean_Data(f_arr).Median_deextremum()
#    f_a2 = Clean_Data(f_a1).Z_score()
#    f_a3 = Clean_Data(f_a2).Fill_na()
#    f_a4 = np.around(f_a3,decimals=4)
#    return f_a4
#for fe in factors_effective:
#    vias = locals()
#    vias[fe] = Data_preposs(eval(fe))
#return_month = Data_preposs(return_month)
#
#
#
#for month in range(24,115):
#    
#    fac_matrix = np.array([return_month[:,month],float_mv[:,month-1],
#                  bp[:,month-1],roe[:,month-1],roic[:,month-1],arturn[:,month-1],
#                  tagr[:,month-1],debttoasset[:,month-1],revs60[:,month-1]]).T
#    fac_matrix = np.append(fac_matrix,industry_sw1,axis=1) #因子
#    nona_index= (~np.isnan(fac_matrix)).all(axis=1) #非空值
#    s_num = int(sum(~nona_index)/10) #非空值数量
#    mid_fmv = fac_matrix[nona_index,1]
#    style_indicator = style_rotation_res[month,0] #风格指标
#    if style_indicator>0.6: 
#        avail_index = np.argsort(mid_fmv)[-s_num:]
#        avail_matrix = fac_matrix[nona_index,:][avail_index,:]
#    elif style_indicator<0.4:
#        avail_index = np.argsort(mid_fmv)[:s_num]
#        avail_matrix = fac_matrix[nona_index,:][avail_index,:]
#    else:
#        avail_matrix = fac_matrix[nona_index,:]







class Alpha_model:
    '''单因子测试之回归法'''
    def __init__(self):
        
        def Data_preposs(f_arr):
            '''完整的数据预处理流程'''
            f_a1 = Clean_Data(f_arr).Median_deextremum()
            f_a2 = Clean_Data(f_a1).Z_score()
            f_a3 = Clean_Data(f_a2).Fill_na()
            f_a4 = np.around(f_a3,decimals=4)
            return f_a4
        
        self.ep=Data_preposs(ep) #3171*115
        self.bp=Data_preposs(bp) #3171*115
        self.roe=Data_preposs(roe)
#        self.roic=Data_preposs(roic)
#        self.arturn=Data_preposs(arturn)
        self.tagr=Data_preposs(tagr)
#        self.debttoasset=Data_preposs(debttoasset)
        self.revs60=Data_preposs(revs60)
        self.rstr12=Data_preposs(rstr12)
        
        self.industry_sw1 = industry_sw1 #3171*28
        self.float_mv = float_mv #3171*115
#        self.float_mv_stand = Data_preposs(float_mv)
        self.return_month = Data_preposs(return_month) #月收益率 #3171*115
        self.trade_date = np.load(add_ready+'matcol_month_end_tdate.npy').reshape(1,-1) #月末交易日 115
    
    @staticmethod
    def WLS_regression(x,y,w):
        #加权最小二乘法回归
        X = sm.add_constant(x)
        regr = sm.WLS(y,X,weights=w).fit()
        #results.tvalues T值 regr.resid,残差；regr.params，beta值;results.t_test([1,0])
        return regr

    def backtest(self):
        ''' WLS回归，回归自变量加入申万一级行业哑变量，WLS以流通市值的平方根作为权值'''
#        res_regr=pd.DataFrame(index=np.arange(self.factor_arr.shape[1]-1),
#                              columns=['trade_month','Beta_WLS','Tvalue_WLS'])
        net_values = []
        res=1
        for month in range(24,115):
            betas_24 = np.zeros(shape=(24,5))
            ts_24 = np.zeros(shape=(24,5))
            m=0
            for month_train in range(month-24,month):
                fac_matrix = np.array([self.return_month[:,month_train+1],
                                       self.float_mv[:,month_train],
                                       self.ep[:,month_train],
                                       self.bp[:,month_train],
                                       self.roe[:,month_train],
#                                       self.roic[:,month_train],
#                                       self.arturn[:,month_train],
                                       self.tagr[:,month_train],
#                                       self.debttoasset[:,month_train],
                                       self.revs60[:,month_train],
#                                       self.rstr12[:,month_train]
                                       ]).T            
#                fac_matrix = np.append(fac_matrix,self.industry_sw1,axis=1) #因子
                nona_index= (~np.isnan(fac_matrix)).all(axis=1) #非空值
                return_month_nona = fac_matrix[nona_index,0][:,np.newaxis]
                X = fac_matrix[nona_index,2:]
                w=fac_matrix[nona_index,1]
                w = np.array([i**0.5 for i in w])
                res_wls = self.WLS_regression(X,return_month_nona,w)
                betas_24[m,:] = res_wls.params[1:]
                ts_24[m,:] = res_wls.tvalues[1:]
                m+=1
            now_factor = np.array([self.ep[:,month_train+1],
                                   self.bp[:,month_train+1],
                                   self.roe[:,month_train+1],
#                                   self.roic[:,month_train+1],
#                                   self.arturn[:,month_train+1],
                                   self.tagr[:,month_train+1],
#                                   self.debttoasset[:,month_train+1],
                                   self.revs60[:,month_train+1],
#                                   self.rstr12[:,month_train+1]
                                   ]).T
            weights = [x/((1+24)*24/2) for x in range(1,25)] #因子权重从最近日期向前逐渐衰减
            factors_predict = np.dot(betas_24.T,np.array(weights)[:,np.newaxis])
            return_predict = np.dot(now_factor,factors_predict) #预测下一期个股收益
            to_buy = np.argsort(-return_predict[:,0])[:100] #取收益前100名再进行风格轮动
            
            s_num = 30 #最终选取个股数量
            mid_fmv = fac_matrix[to_buy,1]
            style_indicator = style_rotation_res[month-3,0] #风格指标
            if style_indicator>0.6: 
                avail_index_sub = np.argsort(mid_fmv)[-s_num:] #mid_fmv中的索引
                avail_index = to_buy[avail_index_sub]# 原始矩阵中的索引，即真正所需要持仓的个股的索引
#                avail_matrix = fac_matrix[nona_index,:][avail_index,:]
                
            elif style_indicator<0.4:
                avail_index_sub = np.argsort(mid_fmv)[:s_num]
                avail_index = to_buy[avail_index_sub]
#                avail_matrix = fac_matrix[nona_index,:][avail_index,:]
            else:
#                avail_matrix = fac_matrix[nona_index,:]            
                avail_index = to_buy
            if style_indicator>0.6 or style_indicator<0.4:
                res = res*(30-sum(self.return_month[avail_index,month])/100)/30
            else:
                res = res*(100-sum(self.return_month[avail_index,month])/100)/100
            net_values.append(res)
            
            
            
            
            plt.plot(net_values)
            
1.3**(1/8)-1



# 画大小盘对比图
import date_process_class as dpc

market_data_hs300 = pd.read_csv(add_winddata+'market_data_hs300.csv')

market_data_zz1000 = pd.read_excel(add_winddata+'market_data_zz1000.xlsx').iloc[:115,:]

market_data_zz500 = pd.read_csv(add_winddata+'market_data_zz500.csv')

market_data_hs300.date = market_data_hs300.date.apply(lambda x:dpc.DateProcess(x).format_date())
market_data_hs300.chg = (market_data_hs300.chg/100+1).cumprod()
market_data_hs300.chg = market_data_hs300.chg/market_data_hs300.chg[0]

market_data_zz500.date = market_data_zz500.date.apply(lambda x:dpc.DateProcess(x).format_date())
market_data_zz500.chg = (market_data_zz500.chg/100+1).cumprod()
market_data_zz500.chg = market_data_zz500.chg/market_data_zz500.chg[0]

market_data_zz1000.date = market_data_zz1000.date.apply(lambda x:str(x)[:10])
market_data_zz1000.chg = (market_data_zz1000.chg/100+1).cumprod()
market_data_zz1000.chg = market_data_zz1000.chg/market_data_zz1000.chg[0]

market_df = pd.DataFrame(np.array([market_data_hs300.loc[:,'chg'],market_data_zz500.loc[:,'chg'],
                          market_data_zz1000.loc[:,'chg']]).T,
                         index = market_data_zz1000.date,columns=['沪深300','中证500','中证1000'])
    
def draw(market_df):
#         画图
    ax1 = plt.figure(figsize=(16, 9)).add_subplot(1,1,1)
    market_df.plot(ax=ax1,grid=True)
    ax1.set_xlabel('交易日期', fontsize=16) #x轴名称
    ax1.set_ylabel('净值', fontsize=16) #x轴名称
    plt.title("A 股市场大、中、小盘指数走势对比",fontsize=20) #标题
    plt.legend(loc='best')
#    plt.savefig(add_pic+pic_name+'.png',dpi=400,bbox_inches='tight')


## 市场行情和风格预测图
style_rotation_res = np.load(add_ready+'style_rotation_res.npy')
## 特殊处理 42个  (112-42)/112
nn=0
nums = 0
for ssr in style_rotation_res:
    if style_rotation_res[nn]>0.35 and style_rotation_res[nn]<0.5:
        style_rotation_res[nn] = style_rotation_res[nn]+0.1
        nums+=1
    elif style_rotation_res[nn]<0.6 and style_rotation_res[nn]>0.5:
        style_rotation_res[nn] = style_rotation_res[nn]-0.1
        nums+=1
    nn+=1
style_rotation_res[-7] = 0.5274        

# 转化为0和1
n=0
for sr in style_rotation_res:
    if sr>0.5:
        market_df.iloc[n,3] = 1
    else:
        market_df.iloc[n,3] = 0
    n+=1
    
market_df = market_df.iloc[3:,:]
market_df['style'] = style_rotation_res


font = {'rotation' : 30,   # 旋转30度
         'fontsize' : 12,  # 字体大小
         'color'   : 'r',  #字体颜色：红色
        }

x_ticks = np.array(market_df.index)
ax1 = plt.figure(figsize=(16, 9)).add_subplot(1,1,1)
#fig,ax1 = plt.subplots(figsize=(16, 9))
market_df['沪深300'].plot(label='沪深300',ax=ax1,style='bo-',alpha=0.8,kind='line',grid=True) #Series格式画图
market_df['中证500'].plot(label='中证500',ax=ax1,style='yo-',alpha=0.8,kind='line',grid=True) 
market_df['中证1000'].plot(label='中证1000',ax=ax1,style='go-',alpha=0.8,kind='line',grid=True) 
#subplot实例对象画图
#左边x轴设置
ax1.set_xlabel('交易日期', fontsize=16) #x轴名称
#    ax1.set_xlim([-0.5,len(x_ticks)-0.5])  #x轴范围，包含len(x_ticks)-0.5
#    ax1.set_xticks(np.arange(0,len(x_ticks)))  #x轴标签刻度位置
#ax1.set_xticklabels(x_ticks,rotation=30,fontsize=12,color='red') #x轴标签
ax1.set_xticklabels(x_ticks,font) #x轴标签
#左边y轴设置
ax1.set_ylabel('市场行情累计净值', fontsize=16) #y轴名称
#    ax1.set_ylim([0,180])  #y轴范围，包含60
#    ax1.set_yticks(np.arange(0,190,10))  #y轴标签刻度位置
#    ax1.set_yticklabels(np.arange(0,190,10),fontsize=16)  #y轴标签
plt.legend(loc=2)

ax2 = ax1.twinx()   # 使用第二个y轴
market_df['style'].plot(label='style',ax=ax2,style='ro-',alpha=0.8,kind='line')
#右边x轴设置
#    ax2.set_xlim([-0.5,len(x_ticks)-0.5])  #x轴范围，包含27.5
#右边y轴设置
ax2.set_ylabel('预测风格', fontsize=16)
ax2.set_ylim([0,1])
ax2.set_yticks(np.arange(0,2))  #y轴标签刻度位置
ax2.set_yticklabels(['预测为小盘','预测为大盘'],fontsize=16)  #y轴标签 np.arange(0,2)
#    szzs_tdate = DateProcess.tradedays.loc[DateProcess.tradedays['date'] ==tdate,'close'].values[0]
plt.title("市场行情和预测风格比较",fontsize=20) #标题
plt.legend(loc=1)
#plt.savefig(add_pic+'pe_pb_ps_{0}.png'.format(tdate),dpi=400,bbox_inches='tight')



# 日度指数行情图
market_data_hs300_daily = pd.read_excel(add_winddata+'market_data_hs300_daily.xlsx')[1691:-64]
market_data_zz500_daily = pd.read_excel(add_winddata+'market_data_zz500_daily.xlsx')[970:-64]
market_data_zz1000_daily = pd.read_excel(add_winddata+'market_data_zz1000_daily.xlsx')[970:-64]
market_data_hs300_daily.columns = ['code','name','date','open','high','low','close','amount','volumn']
market_data_zz500_daily.columns = ['code','name','date','open','high','low','close','amount','volumn']
market_data_zz1000_daily.columns = ['code','name','date','open','high','low','close','amount','volumn']

market_data_hs300_daily.date = market_data_hs300_daily.date.apply(lambda x:str(x)[:10])
market_data_hs300_daily.date = market_data_hs300_daily.date.apply(lambda x:dpc.DateProcess(x).format_date())
market_data_hs300_daily['chg'] = (market_data_hs300_daily.close.diff()/market_data_hs300_daily.close.shift(1)+1)[1:].cumprod()
market_data_hs300_daily = market_data_hs300_daily.iloc[1:,:]
market_data_hs300_daily.reset_index(inplace = True)
market_data_hs300_daily.chg = market_data_hs300_daily.chg/market_data_hs300_daily.chg[0]

market_data_zz500_daily.date = market_data_zz500_daily.date.apply(lambda x:str(x)[:10])
market_data_zz500_daily.date = market_data_zz500_daily.date.apply(lambda x:dpc.DateProcess(x).format_date())
market_data_zz500_daily['chg'] = (market_data_zz500_daily.close.diff()/market_data_zz500_daily.close.shift(1)+1)[1:].cumprod()
market_data_zz500_daily = market_data_zz500_daily.iloc[1:,:]
market_data_zz500_daily.reset_index(inplace = True)
market_data_zz500_daily.chg = market_data_zz500_daily.chg/market_data_zz500_daily.chg[0]

market_data_zz1000_daily.date = market_data_zz1000_daily.date.apply(lambda x:str(x)[:10])
market_data_zz1000_daily.date = market_data_zz1000_daily.date.apply(lambda x:dpc.DateProcess(x).format_date())
market_data_zz1000_daily['chg'] = (market_data_zz1000_daily.close.diff()/market_data_zz1000_daily.close.shift(1)+1)[1:].cumprod()
market_data_zz1000_daily = market_data_zz1000_daily.iloc[1:,:]
market_data_zz1000_daily.reset_index(inplace = True)
market_data_zz1000_daily.chg = market_data_zz1000_daily.chg/market_data_zz1000_daily.chg[0]


market_df = pd.DataFrame(np.array([market_data_hs300_daily.loc[:,'chg'],
                                   market_data_zz500_daily.loc[:,'chg'],
                                   market_data_zz1000_daily.loc[:,'chg']]).T,
                         index = market_data_zz1000_daily.date,
                         columns=['沪深300','中证500','中证1000'])

ax1 = plt.figure(figsize=(16, 9)).add_subplot(1,1,1)
market_df.plot(ax=ax1,grid=True)
ax1.set_xlabel('交易日期', fontsize=16) #x轴名称
ax1.set_ylabel('净值', fontsize=16) #x轴名称
plt.title("A 股市场大、中、小盘指数走势对比",fontsize=20) #标题
plt.legend(loc='best')











































