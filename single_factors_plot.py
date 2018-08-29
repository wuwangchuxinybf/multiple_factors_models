# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 17:27:55 2018

@author: wuwangchuxin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import os
os.chdir('D:/multiple_factors_models/')
from single_factors_test import Clean_Data
# 解决X轴名称不能显示中文的问题
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
#%matplotlib inline

add_winddata = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/wind/'
add_ready = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/prepared_data/'
add_pic = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/20180830report/'

class Single_factors_draw():
    def __init__(self):
#        self.pe = np.load(add_ready+'windfactors_pe.npy')
#        self.pb = np.load(add_ready+'windfactors_pb.npy')
#        self.ps = np.load(add_ready+'windfactors_ps.npy')
        self.industry = pd.read_excel(add_winddata+'industry_sw1_class.xlsx')
        self.stockcode = np.load(add_ready+'stockscode.npy').reshape(-1,1)
        self.trade_date = np.load(add_ready+'month_end_tdate.npy').reshape(1,-1)
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

        
    def mean_industry_process(self,factor):
        #去极值并按行业分类求均值
        mid_factor = Clean_Data(factor).Median_deextremum()
        factor_df = pd.DataFrame(mid_factor,columns=self.trade_date[0,:],index=self.stockcode[:,0])
        factor_mean_industry = factor_df.groupby(np.array(self.industry.industry_1class)).mean()      
        return factor_mean_industry
    
    def mean_industry_pe(self):
        return self.mean_industry_process(self.pe)
    def mean_industry_pb(self):
        return self.mean_industry_process(self.pb)
    def mean_industry_ps(self):
        return self.mean_industry_process(self.ps)
    
    @staticmethod
    def ordinal_industry_process(factor):
        # 每年最后一个月的因子值按照行业分类排序
        factor_year = factor.groupby([x[:4] for x in factor.columns],axis=1).last()
        factor_year_arr = np.array(factor_year.values)
        factor_year_ord = Clean_Data(factor_year_arr).Ordinal_values()
        factor_year_ord =pd.DataFrame(factor_year_ord,index = factor_year.index,
                                      columns =factor_year.columns)
        return factor_year_ord
    
    def ordinal_industry_pe(self):
        processed_pe = self.ordinal_industry_process(self.mean_industry_pe())
        processed_pe.to_csv(add_pic+'pe_indus_ord.csv')
        return processed_pe
    def ordinal_industry_pb(self):
        processed_pb = self.ordinal_industry_process(self.mean_industry_pb())
        processed_pb.to_csv(add_pic+'pb_indus_ord.csv')
        return processed_pb
    def ordinal_industry_ps(self):
        processed_ps = self.ordinal_industry_process(self.mean_industry_ps())
        processed_ps.to_csv(add_pic+'ps_indus_ord.csv')
        return processed_ps
    
    @staticmethod
    def fomat_df(pe,pb,ps):
        #将全部的月度截面因子数据合并为dataframe类型,目的是以最近一期的PE值排序，
        #之前的各期均以最近一期的行业排序绘图
        n=1
        res=pd.DataFrame()
        dates_selected = pe.columns[::-1]
        for i in dates_selected:
            mid_df = pd.DataFrame([pe[i],pb[i],ps[i]]).T
            mid_df.columns=['PE_%s'%n,'PB_%s'%n,'PS_%s'%n]
            if n==1:
                mid_df.sort_values(by='PE_1',ascending=False,inplace=True)
            if len(res):
                res = pd.merge(res,mid_df,how='inner',left_index=True,right_index=True)
            else:
                res = mid_df
            n+=1
        return res

    @staticmethod
    def draw(factors_df,pe_c,pb_c,ps_c,tdate):
        # 单因子横截面数据按行业分类平均比较图,原始因子PE,PB,PS
        # 画图参数        
        font = {'rotation' : 30,   # 旋转30度
                 'fontsize' : 12,  # 字体大小
                 'color'   : 'r',  #字体颜色：红色
                }
        
        x_ticks = np.array(factors_df.index)
        y1 = factors_df[pe_c]
        y2 = factors_df[pb_c]
        y3 = factors_df[ps_c]
        
        ax1 = plt.figure(figsize=(16, 9)).add_subplot(1,1,1)
        #fig,ax1 = plt.subplots(figsize=(16, 9))
        y1.plot(label='PE',ax=ax1,style='bo-',alpha=0.8,kind='bar',grid=True) #Series格式画图
        #subplot实例对象画图
        #左边x轴设置
        ax1.set_xlabel('申万一级行业分类', fontsize=16) #x轴名称
        ax1.set_xlim([-0.5,len(x_ticks)-0.5])  #x轴范围，包含len(x_ticks)-0.5
        ax1.set_xticks(np.arange(0,len(x_ticks)))  #x轴标签刻度位置
        #ax1.set_xticklabels(x_ticks,rotation=30,fontsize=12,color='red') #x轴标签
        ax1.set_xticklabels(x_ticks,font) #x轴标签
        #左边y轴设置
        ax1.set_ylabel('PE values', fontsize=16) #y轴名称
        ax1.set_ylim([0,180])  #y轴范围，包含60
        ax1.set_yticks(np.arange(0,190,10))  #y轴标签刻度位置
        ax1.set_yticklabels(np.arange(0,190,10),fontsize=16)  #y轴标签
        plt.legend(loc=2)
        
        ax2 = ax1.twinx()   # 使用第二个y轴
        y2.plot(label='PB',ax=ax2,style='ro-',alpha=0.8,kind='line')
        y3.plot(label='PS',ax=ax2,style='go-',alpha=0.8,kind='line')
        #右边x轴设置
        ax2.set_xlim([-0.5,len(x_ticks)-0.5])  #x轴范围，包含27.5
        #右边y轴设置
        ax2.set_ylabel('PB&PS values', fontsize=16)
        ax2.set_ylim([0,18])
        ax2.set_yticks(np.arange(0,19))  #y轴标签刻度位置
        ax2.set_yticklabels(np.arange(0,19),fontsize=16)  #y轴标签
        
        plt.title("沪深A股申万一级行业分类PE、PB、PS比较（{0})".format(tdate),fontsize=20) #标题
        plt.legend(loc=1)
        plt.savefig(add_pic+'pe_pb_ps_{0}.png'.format(tdate),dpi=400,bbox_inches='tight')
    
    #画行业因子均值的热力图
#    @staticmethod
#    def heat_map(df):
#        f, ax = plt.subplots(figsize=(32, 9))
##        plt.xticks(rotation='90')
#        sns.heatmap(df, square=True, linewidths=.5, annot=True)
#        plt.show()
           
    def main(self):
        # 仅画图
        pe = self.mean_industry_pe()
        pb = self.mean_industry_pb()
        ps = self.mean_industry_ps()
        pe_pb_ps = self.fomat_df(pe,pb,ps)
        
        reversed_tdate = self.trade_date[0,:][::-1]
        for num in range(1,len(reversed_tdate)+1):
            tdate = reversed_tdate[num-1]
            self.draw(pe_pb_ps,'PE_%s'%num,'PB_%s'%num,'PS_%s'%num,tdate)
        # 年末因子均值按行业排名
        self.ordinal_industry_pe()
        self.ordinal_industry_pb()
        self.ordinal_industry_ps()

if __name__=='__main__':
    drawing = Single_factors_draw()
    drawing.main()






