# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:43:27 2018

@author: wuwangchuxin
"""
import numpy as np
import pandas as pd

add_winddata = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/wind/'
add_ready = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/prepared_data/'
add_pic = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/20180223report/'

class Data_basic():
    def __init__(self):
        self.pe = np.load(add_ready+'windfactors_pe.npy')
        self.pb = np.load(add_ready+'windfactors_pb.npy')
        self.ps = np.load(add_ready+'windfactors_ps.npy')
        self.industry_sw1 = np.load(add_ready+'industry_sw1.npy')
        self.industry_sw1_name = np.load(add_ready+'industry_sw1_name.npy').reshape(1,-1)
        self.industry = pd.read_excel(add_winddata+'industry_sw1_class.xlsx')
        #industry_dict = {'交通运输':'JTYS','休闲服务':'XXFW','传媒':'CM','公用事业':'GYSY',
        #                 '农林牧渔':'NLMY','化工':'HG','医药生物':'YYSW','商业贸易':'SYMY',
        #                 '国防军工':'GFJG','家用电器':'JYDQ','建筑材料':'JZCL','建筑装饰':'JZZS',
        #                 '房地产':'FDC','有色金属':'YSJS','机械设备':'JXSB','汽车':'QC',
        #                 '电子':'DZ','电气设备':'DQSB','纺织服装':'FZFZ','综合':'ZH',
        #                 '计算机':'JSJ','轻工制造':'QGZZ','通信':'TX','采掘':'CJ','钢铁':'GT',
        #                 '银行':'YH','非银金融':'FYJR','食品饮料':'SPYL'}
        # 名称替换为英文形式
        #for i in np.arange(len(industry.industry_1class)):
        #    industry.loc[i,'industry_1class'] = industry_dict[industry.loc[i,'industry_1class']]
        self.stockcode = np.load(add_ready+'stockscode.npy').reshape(-1,1)
        self.trade_date = np.load(add_ready+'month_end_tdate.npy').reshape(1,-1)