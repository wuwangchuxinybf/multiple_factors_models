# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 14:46:33 2018

@author: wuwangchuxin
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
os.chdir('D:/multiple_factors_models/')
from single_factors_test import Clean_Data
#import date_process_class as dpc

add_winddata = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/wind/'
add_ready = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/prepared_data/'
#
# load data
pe = np.load(add_ready+'windfactors_pe.npy')
return_month = np.load(add_ready+'wind_return_month.npy')
stockcode = np.load(add_ready+'stockscode.npy').reshape(-1,1)
trade_date = np.load(add_ready+'month_end_tdate.npy').reshape(1,-1)
start_date = np.load(add_ready+'stock_tdate_start.npy').reshape(-1,1)
end_date = np.load(add_ready+'stock_tdate_end.npy').reshape(-1,1)
float_mv = np.load(add_ready+'wind_float_mv.npy')
industry_sw1 = np.load(add_ready+'industry_sw1.npy')
industry_sw1_name = np.load(add_ready+'industry_sw1_name.npy').reshape(1,-1)

industry = pd.read_excel(add_winddata+'industry_sw1_class.xlsx')

#industry.code.groupby(industry.industry_1class)

pe = Clean_Data(pe).Median_deextremum()
pe_df = pd.DataFrame(pe,columns=trade_date[0,:],index=stockcode[:,0])

res = pe_df.groupby(np.array(industry.industry_1class)).mean()

#group_list = list(pe_df.groupby(np.array(industry.industry_1class)))





















































