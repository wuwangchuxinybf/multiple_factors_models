# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:49:45 2019

@author: wuwangchuxin
"""

import pandas as pd
import numpy as np
from single_factors_test import Clean_Data

add_winddata = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/wind/'

class Logistic_Regression:
    def __init__(self,X,y,alpha):
        self.X = X
        self.y = y
        self.alpha = alpha
        
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def gradient_function(self,theta):
        '''Gradient of the function J definition.'''
        h = self.sigmoid(np.dot(self.X,theta))     
        return np.dot(self.X.transpose(),h-self.y[:,np.newaxis])  

    def grad_descent(self):
        '''梯度下降算法'''
        theta = np.ones((self.X.shape[1], 1))  #初始化回归系数（n, 1)   np.array([1, 1]).reshape(2, 1)     
        gradient = self.gradient_function(theta)
        n=1
#        while not np.all(np.absolute(gradient) <= 1e-5):
        maxCycle = 100000
        for i in range(maxCycle):
            theta = theta - self.alpha*gradient  #梯度
            gradient = self.gradient_function(theta)
            n+=1
            print (n)
        return theta


class Logit_style:
    '''使用逻辑回归实现风格轮动'''
    def __init__(self):
        self.market_hs300 = pd.read_csv(add_winddata+'market_data_hs300.csv')
        self.market_zz1000 = pd.read_excel(add_winddata+'market_data_zz1000.xlsx').iloc[:115,:]
        self.market_exchange = pd.read_excel(add_winddata+'market_exchange.xlsx').iloc[:115,:]
        self.market_M2 = pd.read_excel(add_winddata+'market_M2.xls').iloc[:115,:]
        self.market_CPI = pd.read_excel(add_winddata+'market_CPI.xls').iloc[:115,:]
        self.market_hs300_pettm = pd.read_excel(add_winddata+'market_hs300_pettm.xlsx').iloc[:115,:]
        self.market_zz500_pettm = pd.read_excel(add_winddata+'market_zz500_pettm.xlsx').iloc[:115,:]
        self.market_hs300_pblf = pd.read_excel(add_winddata+'market_hs300_pblf.xlsx').iloc[:115,:]
        self.market_zz500_pblf = pd.read_excel(add_winddata+'market_zz500_pblf.xlsx').iloc[:115,:]
        self.market_zzgz = pd.read_excel(add_winddata+'market_zzgz.xlsx').iloc[:115,:]
        self.market_zzqz = pd.read_excel(add_winddata+'market_zzqz.xlsx').iloc[:115,:]
        self.market_hs300_vol = pd.read_excel(add_winddata+'market_hs300_vol.xlsx').iloc[:115,:]
        self.market_zz500_vol = pd.read_excel(add_winddata+'market_hs300_vol.xlsx').iloc[:115,:]
        
    def gene_y(self):
        '''生成0-1序列作为被解释变量，如果沪深300指数涨跌幅大于中证1000指数，则为1，否则为0'''
        classLabels = np.array(self.market_hs300['chg']>self.market_zz1000['chg']).astype(int)[3:]
        return classLabels
            
    def gene_factors(self):
        '''滞后一期：汇率变化，PE 差，PB 差，股债收益溢价，指数波动率差
           滞后二期：货币供应量，消费价格水平
           滞后一期和三期：大小盘指数涨幅差滞后项'''       
        fac = np.array([self.market_exchange['chg'][2:-1], 
                        self.market_M2['chg'][1:-2],
                        self.market_CPI['chg'][1:-2],
                        self.market_zz500_pettm['pe_ttm'][2:-1]-self.market_hs300_pettm['pe_ttm'][2:-1],
                        self.market_zz500_pblf['chg'][2:-1]-self.market_hs300_pblf['chg'][2:-1],
                        self.market_zzqz['chg'][2:-1]-self.market_zzgz['chg'][2:-1],
                        self.market_zz500_vol['chg'][2:-1]-self.market_hs300_vol['chg'][2:-1],
                        self.market_hs300['chg'][2:-1]-self.market_zz1000['chg'][2:-1],
                        self.market_hs300['chg'][:-3]-self.market_zz1000['chg'][:-3]]).T
        dataMatIn = np.insert(fac, 0, 1, axis=1)
        return dataMatIn

if __name__=='__main__':
    alpha = 0.1
    def Data_preposs(f_arr):
        '''完整的数据预处理流程'''
        f_a1 = Clean_Data(f_arr).Median_deextremum()
        f_a2 = Clean_Data(f_a1).Z_score()
        f_a4 = np.around(f_a2,decimals=4)
        return f_a4   
    y = Logit_style().gene_y()
    X = Logit_style().gene_factors()
    X = Data_preposs(X)
    res = Logistic_Regression(X,y,alpha).grad_descent()






























