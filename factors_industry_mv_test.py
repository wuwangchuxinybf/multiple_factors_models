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
industry_dict = {'交通运输':'JTYS','休闲服务':'XXFW','传媒':'CM','公用事业':'GYSY',
                 '农林牧渔':'NLMY','化工':'HG','医药生物':'YYSW','商业贸易':'SYMY',
                 '国防军工':'GFJG','家用电器':'JYDQ','建筑材料':'JZCL','建筑装饰':'JZZS',
                 '房地产':'FDC','有色金属':'YSJS','机械设备':'JXSB','汽车':'QC',
                 '电子':'DZ','电气设备':'DQSB','纺织服装':'FZFZ','综合':'ZH',
                 '计算机':'JSJ','轻工制造':'QGZZ','通信':'TX','采掘':'CJ','钢铁':'GT',
                 '银行':'YH','非银金融':'FYJR','食品饮料':'SPYL'}
# 名称替换为英文形式
for i in np.arange(len(industry.industry_1class)):
    industry.loc[i,'industry_1class'] = industry_dict[industry.loc[i,'industry_1class']]


#industry.code.groupby(industry.industry_1class)

pe = Clean_Data(pe).Median_deextremum()
pe_df = pd.DataFrame(pe,columns=trade_date[0,:],index=stockcode[:,0])

res = pe_df.groupby(np.array(industry.industry_1class)).mean()

#group_list = list(pe_df.groupby(np.array(industry.industry_1class)))




ax = plt.figure().add_subplot(1,1,1)
res['2018-07-31'].plot(label='PE',ax=ax,style='bo-',alpha=0.61,
                       kind='bar',rot=30)
plt.legend(loc='best')                  



s = pd.Series(np.random.randn(10).cumsum(), index=np.arange(10, 110, 10))
ax_test = plt.figure().add_subplot(1,1,1)
s.plot(label='label',ax=ax_test,style='bo-',alpha=0.61,
       kind='line',logx=False,logy=False,rot=30,use_index=True,
       xticks=[10,30,50,70,90],yticks=[-5,-2.5,0,2.5,5],
       xlim=[10,80],ylim=[-5,5],grid=True)
plt.legend(loc='best')



res['2018-07-31'].sort_values()




















































