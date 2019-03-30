# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:17:08 2018

@author: wuwangchuxin
"""

#大类因子分析
import numpy as np

def factors_corrcoef(fac_arr):
    return np.corrcoef(fac_arr)

#月度数据大类因子分析的思路：假如总共100个月的因子月度数据，对每个月某一大类因子内部的K个因子计算相关系数矩阵，
#这样就能得到100个相关系数矩阵序列，然后计算两两因子之间的相关系数统计值，
#比如相关系数的均值，中位数，或者进行T检验。


def category_analysis(fac_arr):
    arr_jiemian = np.vstack(())
#    L = [str(x) for x in fac_arr]
    for date in fac_arr[0].shape(1):
        arr_jiemian = np.vstack(())
        res = np.corrcoef(fac_arr,arr_jiemian)

    return res

if __name__=='__main__':
    category_analysis(facs)

L=[]
for n in range(115):
    nona_index = (~np.isnan(ep[:,n])) & (~np.isnan(bp[:,n]))
    ress = np.corrcoef(ep[nona_index,n],bp[nona_index,n])[0,1]
    L.append(ress)

L2= list(map(lambda x:'%.4f'%x,L))

L3 = list(map(lambda x:float(x),L2))

L4 = (L3 - np.nanmean(L3))/np.nanstd(L3)


ss.ttest_1samp(L4, popmean = 0)[0]







