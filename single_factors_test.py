# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 21:48:18 2018

@author: wuwangchuxin
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
os.chdir('D:/multiple_factors_models/')
#import date_process_class as dpc

add_ready = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/prepared_data/'

# load data
pe = np.load(add_ready+'windfactors_pe.npy')
ep = np.load(add_ready+'windfactors_ep.npy')
return_month = np.load(add_ready+'wind_return_month.npy')
stockcode = np.load(add_ready+'stockscode.npy').reshape(-1,1)
trade_date = np.load(add_ready+'month_end_tdate.npy').reshape(1,-1)
start_date = np.load(add_ready+'stock_tdate_start.npy').reshape(-1,1)
end_date = np.load(add_ready+'stock_tdate_end.npy').reshape(-1,1)
float_mv = np.load(add_ready+'wind_float_mv.npy')
#float_mv = np.delete(float_mv,21,axis=1) #有一行空值

class Clean_Data():
    def __init__(self,arr):
        self.arr=arr
    
    def Median_deextremum(self,n=5):
        # 中位数去极值法
        med = np.nanmedian(self.arr,axis=0)
        mad = np.nanmedian(np.abs(self.arr - med),axis=0)
        res = np.where(self.arr>n*mad,n*mad,np.where(self.arr<(-n)*mad,(-n)*mad,self.arr))
        return res

    def Ordinal_values(self):
        # 求数据集的排序值
        # argsort 不能忽略nan值
        res = self.arr.copy()
        not_nan_num = np.sum(~np.isnan(self.arr),axis=0) #非空值数量
        mid = np.argsort(np.where(np.isnan(self.arr),99999,self.arr),axis=0) #将nan值赋值为大数字然后排序
        #末尾插入一列递增数列，作为名次标记
        mid2 = np.insert(mid,mid.shape[1],values=np.arange(mid.shape[0]),axis=1) 
        # 逐列进行计算
        for n in np.arange(self.arr.shape[1]):
            mid_tmp = mid2[:,[n,115]] #当前列和标记列，即所在列的位置和排名
            #第一列排序，其位置归位，标记列为排名。得到原来所在位置的元素在所在列的排名
            mid_tmp2 = mid_tmp[mid_tmp[:,0].argsort(axis=0)][:,1]
            #恢复为空值的列
            mid_tmp3 = np.where(mid_tmp2>=not_nan_num[n],np.nan,mid_tmp2)
            res[:,n] = mid_tmp3
        return res

    def Z_score(self):
        #z_score标准化
        return (self.arr - np.nanmean(self.arr))/np.nanstd(self.arr)
    
    def Fill_na(self):
        #将本来应该有但是却为nan值的位置填充为0
#        arr_after_zscore = self.Z_score()
        mid = np.where((trade_date>=start_date) & (trade_date<=end_date),self.arr,99999)
        res = np.where(np.isnan(mid),0,self.arr)
        return res


def OLS_regression(x,y):
    # 普通最小二乘
    X = sm.add_constant(x)
    regr = sm.OLS(y, X).fit() #regr.resid,残差；regr.params，beta值;regr.tvalues,T值
    return regr

def WLS_regression(x,y,w):
    #加权最小二乘法回归
    #w=data.iloc[:,2].tolist()
    #w=np.array([i**0.5 for i in w])
    X = sm.add_constant(x)
    regr = sm.WLS(y,X,weights=w).fit()
    #results.tvalues T值 regr.resid,残差；regr.params，beta值;results.t_test([1,0])
    return regr




if __name__=='__main__':
    #pe倒数值
    ep_med_de = Clean_Data(ep).Median_deextremum()
    ep_m_zscore = Clean_Data(ep_med_de).Z_score()
    ep_traditional = Clean_Data(ep_m_zscore).Fill_na()
    #pe倒数序数值
    ep_ord = Clean_Data(ep).Ordinal_values()
    ep_o_zscore = Clean_Data(ep_ord).Z_score()
    ep_ordinal = Clean_Data(ep_o_zscore).Fill_na()

    # ep_traditional回归
    res_regr_tra=pd.DataFrame(index=np.arange(ep_traditional.shape[1]-1),
                                columns=['trade_month','Beta_OLS','Tvalue_OLS',\
                                         'Beta_WLS','Tvalue_WLS','IC'])
    for n in np.arange(ep_traditional.shape[1]-1):
        #剔除空值
        nona_index = (~np.isnan(ep_traditional[:,n])) & (~np.isnan(return_month[:,n+1]))
        ep_traditional_nona = ep_traditional[:,n][nona_index]
        return_month_nona = return_month[:,n+1][nona_index]
        #OLS
        res_ols = OLS_regression(ep_traditional_nona,return_month_nona)
        #WLS
        w=float_mv[:,n][nona_index]
        w = np.where(np.isnan(w),np.nanmean(w),w) #用均值填充nan值
        w=np.array([(i**0.5)/(10**9) for i in w])
        res_wls = WLS_regression(ep_traditional_nona,return_month_nona,w)
        # IC
        mid_ic = np.corrcoef(ep_traditional_nona,return_month_nona)[0,1]
        #结果
        res_regr_tra.iloc[n,:] = [trade_date[0,n],res_ols.params[1],res_ols.tvalues[1],\
                                    res_wls.params[1],res_wls.tvalues[1],mid_ic]

    # ep_ordinal回归
    res_regr_ord=pd.DataFrame(index=np.arange(ep_ordinal.shape[1]-1),
                                columns=['trade_month','Beta_OLS','Tvalue_OLS',\
                                         'Beta_WLS','Tvalue_WLS','IC'])
    for n in np.arange(ep_ordinal.shape[1]-1):
        #剔除空值
        nona_index = (~np.isnan(ep_ordinal[:,n])) & (~np.isnan(return_month[:,n+1]))
        ep_ord_nona = ep_ordinal[:,n][nona_index]
        return_month_nona = return_month[:,n+1][nona_index]
        #OLS
        res_ols = OLS_regression(ep_ord_nona,return_month_nona)
        #WLS
        w=float_mv[:,n][nona_index]
        w = np.where(np.isnan(w),np.nanmean(w),w) #用均值填充nan值
        w=np.array([i**0.5 for i in w])
        res_wls = WLS_regression(ep_ord_nona,return_month_nona,w)
        # IC
        mid_ic = np.corrcoef(ep_ord_nona,return_month_nona)[0,1]
        #结果
        res_regr_ord.iloc[n,:] = [trade_date[0,n],res_ols.params[1],res_ols.tvalues[1],\
                                    res_wls.params[1],res_wls.tvalues[1],mid_ic]

        # T值结果分析
        # T abs mean
        tam_ols_tra = np.mean(res_regr_tra['Tvalue_OLS'].apply(lambda x:abs(x)))
        tam_wls_tra = np.mean(res_regr_tra['Tvalue_WLS'].apply(lambda x:abs(x)))
        tam_ols_ord = np.mean(res_regr_ord['Tvalue_OLS'].apply(lambda x:abs(x)))
        tam_wls_ord = np.mean(res_regr_ord['Tvalue_WLS'].apply(lambda x:abs(x)))
        #t 值序列绝对值大于2 的占比
        port_ols_tra = np.sum(res_regr_tra['Tvalue_OLS'].apply(lambda x:abs(x))>2)/len(res_regr_tra['Tvalue_OLS']) 
        port_wls_tra = np.sum(res_regr_tra['Tvalue_WLS'].apply(lambda x:abs(x))>2)/len(res_regr_tra['Tvalue_WLS']) 
        port_ols_ord = np.sum(res_regr_ord['Tvalue_OLS'].apply(lambda x:abs(x))>2)/len(res_regr_tra['Tvalue_OLS']) 
        port_wls_ord = np.sum(res_regr_ord['Tvalue_WLS'].apply(lambda x:abs(x))>2)/len(res_regr_tra['Tvalue_WLS']) 
        # 因子收益率序列均值
        Beta_ols_tra = np.mean(res_regr_tra['Beta_OLS'])
        Beta_wls_tra = np.mean(res_regr_tra['Beta_WLS'])
        Beta_ols_ord = np.mean(res_regr_ord['Beta_OLS'])
        Beta_wls_ord = np.mean(res_regr_ord['Beta_WLS'])
        # 
        TIR_ols_tra = tam_ols_tra/np.std(res_regr_tra['Tvalue_OLS'].apply(lambda x:abs(x)))
        TIR_wls_tra = tam_wls_tra/np.std(res_regr_tra['Tvalue_WLS'].apply(lambda x:abs(x)))
        TIR_ols_ord = tam_ols_ord/np.std(res_regr_ord['Tvalue_OLS'].apply(lambda x:abs(x)))
        TIR_wls_ord = tam_wls_ord/np.std(res_regr_ord['Tvalue_WLS'].apply(lambda x:abs(x)))
        
#画图
#res_regr_tra
#res_regr_ord
#
#fig = plt.figure()
#ax1 = fig.add_subplot(1, 2, 1)
#
#ax2 = fig.add_subplot(1, 2, 2)
#
#fig_IC = plt.figure()
#ax_IC = fig.add_subplot(1, 1, 1)
#plt.plot(res_regr_tra['IC'])
#plt.plot(res_regr_ord['IC'])
#ax_IC.hist(res_regr_tra['IC'], bins=20, alpha=0.3)
#ax_IC.hist(res_regr_ord['IC'], bins=20, color='g', alpha=0.3)
#
#ax3 = fig.add_subplot(2, 2, 3)







