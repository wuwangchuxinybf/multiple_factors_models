# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 21:48:18 2018

@author: wuwangchuxin
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
#import matplotlib.pyplot as plt
import os
os.chdir('D:/multiple_factors_models/')
#import date_process_class as dpc
from scipy import stats as ss
import matplotlib.pyplot as plt

# 添加路径，方便导入自定义模块，比如 import date_process_class as dpc
import sys
sys.path.append('D:/multiple_factors_models')

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

add_winddata = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/wind/'
add_ready = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/prepared_data/'
add_pic = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/20180830report/'

# load data
##################################################################################
financial_q = np.load(add_ready+'windfactors_financial_q.npy') #季度财务因子
## 月度因子
pe = np.load(add_ready+'windfactors_pe.npy')
ep = np.load(add_ready+'windfactors_ep.npy')
pb = np.load(add_ready+'windfactors_pb.npy')
bp = np.load(add_ready+'windfactors_bp.npy')
ps = np.load(add_ready+'windfactors_ps.npy')
sp = np.load(add_ready+'windfactors_sp.npy')
#
#float_mv = np.load(add_ready+'wind_float_mv.npy')
#
#return_month = np.load(add_ready+'wind_return_month.npy') #月收益率
#
#stockcode = np.load(add_ready+'stockscode.npy').reshape(-1,1) #股票代码
#trade_date = np.load(add_ready+'month_end_tdate.npy').reshape(1,-1) #月末交易日
#
#industry_sw1 = np.load(add_ready+'industry_sw1.npy') #申万一级行业哑变量
#industry_sw1_name = np.load(add_ready+'industry_sw1_name.npy').reshape(1,-1) #申万一级行业分类名称
##industry = pd.read_excel(add_winddata+'industry_sw1_class.xlsx')  #原始数据
#
#start_date = np.load(add_ready+'stock_tdate_start.npy').reshape(-1,1) #个股有效期起始时间
#end_date = np.load(add_ready+'stock_tdate_end.npy').reshape(-1,1) #个股有效期终止时间
# 
#weights_000300 = np.load(add_ready+'weights_000300.npy') #沪深300指数个股权重
#weights_000300_stocklist = np.load(add_ready+'weights_000300_stocklist.npy')
#
#hs300_sw_1class_weight = np.load(add_ready+'hs300_sw_1class_weight') #沪深300指数申万一级行业分类权重
#hs300_sw_1class_weight_industrynames = np.save(add_ready+'hs300_sw_1class_weight_industrynames')
##################################################################################

class Clean_Data:
    #数据格式：ndarray，3225*115,3225只个股，115个月
    def __init__(self,arr):
        self.arr=arr
        self.trade_date = np.load(add_ready+'month_end_tdate.npy').reshape(1,-1) #月末交易日
        self.start_date = np.load(add_ready+'stock_tdate_start.npy').reshape(-1,1) #个股有效期起始时间
        self.end_date = np.load(add_ready+'stock_tdate_end.npy').reshape(-1,1) #个股有效期终止时间
    
    def Median_deextremum(self,n=5):
        # 中位数去极值法
        med = np.nanmedian(self.arr,axis=0)
        mad = np.nanmedian(np.abs(self.arr - med),axis=0)
        res = np.where(self.arr>(med+n*mad),(med+n*mad),
                       np.where(self.arr<(med-n*mad),(med-n*mad),self.arr))
        return res

    def Ordinal_values(self):
        # 求数据集的排序值
        # argsort 不能忽略nan值
        res = self.arr.copy()
        not_nan_num = np.sum(~np.isnan(self.arr),axis=0) #非空值数量
        #将nan值赋值为大数字然后排序
        mid = np.argsort(np.where(np.isnan(self.arr),99999,self.arr),axis=0) 
        #末尾插入一列递增数列，作为名次标记
        mid2 = np.insert(mid,mid.shape[1],values=np.arange(1,mid.shape[0]+1),axis=1) 
        # 逐列进行计算
        for n in np.arange(self.arr.shape[1]):
            mid_tmp = mid2[:,[n,self.arr.shape[1]]] #当前列和标记列，即所在列的位置和排名
            #第一列排序，其位置归位，标记列为排名。得到原来所在位置的元素在所在列的排名
            mid_tmp2 = mid_tmp[mid_tmp[:,0].argsort(axis=0)][:,1]
            #恢复为空值的列
            mid_tmp3 = np.where(mid_tmp2>=not_nan_num[n]+1,np.nan,mid_tmp2)
            res[:,n] = mid_tmp3
        return res

    def Z_score(self):
        #z_score标准化
        return (self.arr - np.nanmean(self.arr))/np.nanstd(self.arr)
    
    def Fill_na(self):
        #将本来应该有但是却为nan值的位置填充为0
        mid = np.where((self.trade_date>=self.start_date) & (self.trade_date<=self.end_date),
                       self.arr,99999)
        res = np.where(np.isnan(mid),0,self.arr)
        return res

class Single_factors_test_regression:
    # 单因子测试之回归法
    def __init__(self,factor_arr):
        self.factor_arr=factor_arr
        self.industry_sw1 = np.load(add_ready+'industry_sw1.npy')
        self.float_mv = np.load(add_ready+'wind_float_mv.npy')
        self.return_month = np.load(add_ready+'wind_return_month.npy') #月收益率
        self.trade_date = np.load(add_ready+'month_end_tdate.npy').reshape(1,-1) #月末交易日
    
    @staticmethod
    def OLS_regression(x,y):
        # 普通最小二乘法回归
        X = sm.add_constant(x)
        regr = sm.OLS(y, X).fit() #regr.resid,残差；regr.params，beta值;regr.tvalues,T值
        return regr
    
    @staticmethod
    def WLS_regression(x,y,w):
        #加权最小二乘法回归
        #w=data.iloc[:,2].tolist()
        #w=np.array([i**0.5 for i in w])
        X = sm.add_constant(x)
        regr = sm.WLS(y,X,weights=w).fit()
        #results.tvalues T值 regr.resid,残差；regr.params，beta值;results.t_test([1,0])
        return regr

    def single_factor_regress(self):
        
        res_regr=pd.DataFrame(index=np.arange(self.factor_arr.shape[1]-1),
                              columns=['trade_month','Beta_OLS','Tvalue_OLS',
                                       'Beta_WLS','Tvalue_WLS','IC'])
        for n in np.arange(self.factor_arr.shape[1]-1):
            #剔除空值
            nona_index = (~np.isnan(self.factor_arr[:,n])) & (~np.isnan(self.return_month[:,n+1]))
            factor_arr_nona = self.factor_arr[nona_index,n][:,np.newaxis]
            return_month_nona = self.return_month[nona_index,n+1][:,np.newaxis]
            industry_sw1_nona = self.industry_sw1[nona_index,:]
            X = np.hstack((industry_sw1_nona,factor_arr_nona))
            #OLS
            res_ols = self.OLS_regression(X,return_month_nona)
            #WLS
            w=self.float_mv[:,n][nona_index]
            w = np.where(np.isnan(w),np.nanmean(w),w) #用均值填充nan值
            w=np.array([i**0.5 for i in w])
            res_wls = self.WLS_regression(X,return_month_nona,w)
            # IC,先用因子值对行业和市值回归取残差，然后再和次月收益求IC
            float_mv_nona = self.float_mv[nona_index,n][:,np.newaxis]
            float_mv_nona=np.where(np.isnan(float_mv_nona),np.nanmean(float_mv_nona),float_mv_nona)
            X_IC = np.hstack((industry_sw1_nona,float_mv_nona))
            res_ols_IC = self.OLS_regression(X_IC,factor_arr_nona)
            mid_ic = np.corrcoef(res_ols_IC.resid,return_month_nona[:,0])[0,1]
            #结果
            res_regr.iloc[n,:] = [self.trade_date[0,n],res_ols.params[-1],res_ols.tvalues[-1],
                                  res_wls.params[-1],res_wls.tvalues[-1],mid_ic]
        return res_regr
    
    @staticmethod
    def T_analysis(regress_df):
        # T abs mean
        T_res = pd.DataFrame(index=['OLS','WLS'],
                              columns=['t_abs_mean','port_greater_2','beta_mean',
                                       'beta_t','t_mean_div_std'])
        L1 = ['Tvalue_OLS','Tvalue_WLS']
        for col1 in L1:
            t_abs_mean = np.mean(regress_df[col1].apply(lambda x:abs(x)))
            port_greater_2 = np.sum(regress_df[col1].apply(lambda x:abs(x))>2)/len(regress_df[col1])
#            t_mean_div_std = t_abs_mean/np.std(regress_df[col1].apply(lambda x:abs(x)))
            t_mean_div_std = abs(np.mean(regress_df[col1]))/np.std(regress_df[col1])
            T_res.loc[col1[-3:],'t_abs_mean'] = t_abs_mean
            T_res.loc[col1[-3:],'port_greater_2'] = port_greater_2
            T_res.loc[col1[-3:],'t_mean_div_std'] = t_mean_div_std
            
        L2 = ['Beta_OLS','Beta_WLS']
        for col2 in L2:
            beta_mean = np.mean(regress_df[col2])
            beta_t = ss.ttest_1samp(regress_df[col2], popmean = 0)[0]
            T_res.loc[col2[-3:],'beta_mean'] = beta_mean
            T_res.loc[col2[-3:],'beta_t'] = beta_t
        return T_res
    
    @staticmethod        
    def IC_analysis(regress_df):
        IC_res=pd.DataFrame(index=['fac'],columns=['IC_mean','IC_std','IC_IR','IC_positive_port'])
        IC_mean = regress_df['IC'].mean()
        IC_std = regress_df['IC'].std()
        IC_IR = IC_mean/IC_std
        IC_positive_port = np.sum(regress_df['IC']>0)/len(regress_df['IC'])
        IC_res.loc['fac',:] = IC_mean,IC_std,IC_IR,IC_positive_port
        return IC_res

class Single_factors_test_group:
    '''单因子测试之分组法
       数据格式：ndarray，3225*115,3225只个股，115个月 self=Single_factors_test_group(ep)'''
    capital_initial = 100000000
    def __init__(self,factor_arr):
        self.factor_arr=factor_arr
#        self.industry_sw1 = np.load(add_ready+'industry_sw1.npy')
#        self.float_mv = np.load(add_ready+'wind_float_mv.npy')
        self.return_month = np.load(add_ready+'wind_return_month.npy') #月收益率
        self.trade_date = np.load(add_ready+'month_end_tdate.npy')#.reshape(1,-1) #月末交易日
        self.industry = pd.read_excel(add_winddata+'industry_sw1_class.xlsx')
        self.stockcode = np.load(add_ready+'stockscode.npy')#.reshape(-1,1) #股票代码
        self.hs300_sw_1class_weight = np.load(add_ready+'hs300_sw_1class_weight.npy') #沪深300指数申万一级行业分类权重
        self.industrynames = np.load(add_ready+'hs300_sw_1class_weight_industrynames.npy')
        
    def group_net_value(self):
        '''行业中性分组，按单因子值排序分五组构建投资组合，
           并构建多空组合，添加沪深300和中证500指数作为比较'''
           
       industry_w_df=pd.DataFrame(self.hs300_sw_1class_weight,index=self.industrynames,
                                  columns= self.trade_date)
       industry_w_df.fillna(0,inplace=True)
                                 
        factor_df = pd.DataFrame(self.factor_arr,index=self.stockcode,columns=self.trade_date) 
        return_month_df = pd.DataFrame(self.return_month,index=self.stockcode,columns=self.trade_date)
        #资金变化结果df
        capital_df = pd.DataFrame(index = self.trade_date,
                                   columns=['group1','group2','group3','group4','group5']) 
        capital_df.iloc[0,:]=self.capital_initial
        for n in range(len(factor_df.columns)-1):
            mid_df=pd.merge(factor_df[[factor_df.columns[n]]],
                        return_month_df[[factor_df.columns[n+1]]],
                        left_index=True,right_index=True)
            mid_df = pd.merge(mid_df,self.industry,left_index=True,right_on='code',how='inner')
            mid_df.rename(columns={factor_df.columns[n]:'factor',
                                   factor_df.columns[n+1]:'return_month'},inplace = True)
            mid_df.dropna(inplace=True)
            grouped =mid_df.groupby(by='industry_1class') #按行业分组
            grouped_df = pd.DataFrame(columns=['factor','return_month','group_NO'])
            for indus_name,value in grouped:
                value.sort_values(by='factor',ascending=False,inplace=True)
#                value.dropna(inplace=True)
                value.reset_index(drop=True,inplace=True)
                group_amount = int(value.shape[0]/5) #每组股票数量
                for group_n in range(5):
                    if group_n==4:
                        mid_indus_group = value.loc[group_n*group_amount:,['factor','return_month']]
                        mid_indus_group['group_NO'] = 5
                        grouped_df = grouped_df.append(mid_indus_group)
                    else:
                        mid_indus_group = value.loc[group_n*group_amount:(group_n+1)*group_amount-1,
                                                    ['factor','return_month']]
                        mid_indus_group['group_NO'] = group_n+1
                        grouped_df = grouped_df.append(mid_indus_group)
            grouped_df.reset_index(drop=True,inplace=True)
            #资金等权
            for group_num in range(5):
                grouped_indus = grouped_df[grouped_df['group_NO']==(group_num+1)]
                capital_df.iloc[n+1,group_num] = capital_df.iloc[n,group_num] + \
                             int(capital_df.iloc[n,group_num]/(grouped_indus.shape[0]))*\
                             (grouped_indus['return_month'].sum()/100)
            print (n,'done')
        # 计算多空组合
        if capital_df.iloc[-1,0]>capital_df.iloc[-1,-1]:
            capital_df['long_short'] = capital_df['group1']-capital_df['group5']+self.capital_initial
        else:
            capital_df['long_short'] = capital_df['group5']-capital_df['group1']+self.capital_initial
        netvalue_df = capital_df/self.capital_initial
        # 加上基准
        
        return netvalue_df
        
    def backtest_indicates(netvalue_df):
        pass
    
    @staticmethod
    def draw(netvalue_df):
#         画图
        ax1 = plt.figure(figsize=(16, 9)).add_subplot(1,1,1)
        netvalue_df.plot(ax=ax1,grid=True)
        ax1.set_xlabel('交易日期', fontsize=16) #x轴名称
        ax1.set_ylabel('净值', fontsize=16) #x轴名称
        plt.title("PE分组净值曲线",fontsize=20) #标题
        plt.legend(loc='best')
        
 
if __name__=='__main__':
    test = Single_factors_test_group(ep)
    test.group_net_value()
    
#    #pe倒数值
#    ep_med_de = Clean_Data(ep).Median_deextremum()
#    ep_m_zscore = Clean_Data(ep_med_de).Z_score()
#    ep_num = Clean_Data(ep_m_zscore).Fill_na()
#    
#    Single_factors_test_ins = Single_factors_test_regression(ep_num)
#    res_ep_num = Single_factors_test_ins.single_factor_regress()
#    T_ep_num_res = Single_factors_test_ins.T_analysis(res_ep_num)
#    IC_ep_num_res = Single_factors_test_ins.IC_analysis(res_ep_num)
    
    
    
    
    
#    #pe倒数序数值
#    ep_ord = Clean_Data(ep).Ordinal_values()
#    ep_o_zscore = Clean_Data(ep_ord).Z_score()
#    ep_ordinal = Clean_Data(ep_o_zscore).Fill_na()
#    res_ep_ord = Single_factors_test_regression(ep_ordinal).single_factor_regress()
