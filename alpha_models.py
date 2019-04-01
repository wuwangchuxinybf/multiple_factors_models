# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 20:21:05 2019

@author: wuwangchuxin
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import math
import os
os.chdir('D:/multiple_factors_models/')
from scipy import stats as ss
import matplotlib.pyplot as plt

import sys
sys.path.append('D:/multiple_factors_models')

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

add_winddata = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/wind/'
add_ready = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/prepared_data/del_cixin/'
add_pic = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/20190326pic/'

# load data
##################################################################################
## 估值指标
#ep = np.load(add_ready+'windfactors_ep.npy')
bp = np.load(add_ready+'windfactors_bp.npy')
#盈利能力
roe = np.load(add_ready+'windfactors_roe.npy') #净资产收益率（TTM）
roic = np.load(add_ready+'windfactors_roicebit.npy') #投入资本回报率ROIC(TTM)
#经营效率
arturn = np.load(add_ready+'windfactors_arturn.npy') #应收账款周转率(TTM)
#投融资决策
tagr = np.load(add_ready+'windfactors_tagr.npy') #增长率-总资产
#杠杆因子
debttoasset = np.load(add_ready+'windfactors_debttoasset.npy') #资产负债率

#利用市场信号修正策略
#市场价格信号
revs60 = np.load(add_ready+'windfactors_revs60.npy') #过去3个月的价格动量


roe_roic = 0.5*roe+0.5*roic   #合成因子


def









































































