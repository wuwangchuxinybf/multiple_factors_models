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
from date_process_class import DateProcess
# 解决X轴名称不能显示中文的问题
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
#%matplotlib inlinex`

add_winddata = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/wind/'
add_ready = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/MF_data/prepared_data/del_cixin/'
add_pic = 'C:/Users/wuwangchuxin/Desktop/TF_SummerIntern/20190326pic/'

pe = np.load(add_ready+'windfactors_pe.npy')
pb = np.load(add_ready+'windfactors_pb.npy')
ps = np.load(add_ready+'windfactors_ps.npy')
#估值指标
ep = np.load(add_ready+'windfactors_ep.npy')
bp = np.load(add_ready+'windfactors_bp.npy')
sp = np.load(add_ready+'windfactors_sp.npy')
pcf_ocf = np.load(add_ready+'windfactors_pcf_ocf.npy') #市现率
dividendyield = np.load(add_ready+'windfactors_dividendyield.npy') #股息率
evtoebitda = np.load(add_ready+'windfactors_evtoebitda.npy')  #企业倍数

float_mv = np.load(add_ready+'wind_float_mv.npy')

#industry_sw1 = np.load(add_ready+'industry_sw1.npy')
#industry_sw1_name = np.load(add_ready+'industry_sw1_name.npy')
#hs300_sw_1class_weight = np.load(add_ready+'hs300_sw_1class_weight.npy')
#hs300_sw_1class_weight_industrynames = np.load(add_ready+'hs300_sw_1class_weight_industrynames.npy')

class Single_factors_draw:
    def __init__(self): #,*args
#        for fac in args:
#            setattr(self,fac,np.load(add_ready+'windfactors_%s.npy'%fac))
        self.stockcode = np.load(add_ready+'matind_stockscode.npy').reshape(-1,1)
        self.trade_date = np.load(add_ready+'matcol_month_end_tdate.npy').reshape(1,-1)
        self.float_mv = np.load(add_ready+'wind_float_mv.npy')
        industry_code = pd.read_excel(add_winddata+'industry_sw1_class.xlsx')
        industry_code = industry_code[industry_code['code'].isin(self.stockcode[:,0])]
        self.industry = industry_code.sort_values(by='code')
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
       
    def mean_industry_process(self,factor_arr):
        #去极值并按行业分类求均值
        mid_factor = Clean_Data(factor_arr).Median_deextremum()
        factor_df = pd.DataFrame(mid_factor,columns=self.trade_date[0,:],index=self.stockcode[:,0])
        factor_mean_industry = factor_df.groupby(np.array(self.industry.industry_1class)).mean()      
        return factor_mean_industry
    
    @staticmethod
    def ordinal_industry_process(factor_df):
        # 每年最后一个月的因子值按照行业分类排序
        factor_year = factor_df.groupby([x[:4] for x in factor_df.columns],axis=1).last()
        factor_year_arr = np.array(factor_year.values)
        factor_year_ord = Clean_Data(factor_year_arr).Ordinal_values()
        factor_year_ord =pd.DataFrame(factor_year_ord,index = factor_year.index,
                                      columns =factor_year.columns)
        return factor_year_ord
    
    def sub_fac_corr(self,fac_d):
        mid_fac = Clean_Data(fac_d).Median_deextremum()
        fac_df = pd.DataFrame(mid_fac,columns=self.trade_date[0,:],index=self.stockcode[:,0])
        fac_year = fac_df.groupby([x[:4] for x in fac_df.columns],axis=1).last()
        return fac_year

    def fac_corr(self,factor_df,fac_2):
        '''因子与市值或者因子之间的相关系数,float_mv'''
        mid = self.sub_fac_corr(factor_df)
        res = []
        for col in range(mid.shape[1]):
#            np.nancorrcoef(mid[:,col],self.sub_fac_corr(fac_2)[:,col])
            res.append(mid.iloc[:,col].corr(self.sub_fac_corr(fac_2).iloc[:,col]))
        return [round(x,4) for x in res]

    @staticmethod
    def draw(*args):
        # 单因子横截面数据按行业分类平均比较图,原始因子PE,PB,PS
        # 画图参数
        if len(args)!=3:
            raise ValueError('just 3 factors plot availible now!')
        font = {'rotation' : 30,   # 旋转30度
                 'fontsize' : 12,  # 字体大小
                 'color'   : 'r',  #字体颜色：红色
                }
        # 按照最后一个月的第一个因子值倒序排序
        factors_df_list =[]
        for tdate in args[0].columns[::-1]:
            if tdate == '2018-07-31':
                args[0].sort_values(by=tdate,ascending=False,inplace=True)
                factors_df_list.append(args[0])
                for n in range(1,len(args)):
                    factors_df_list.append(args[n].reindex(args[0].index))
            x_ticks = np.array(factors_df_list[0].index)
            for n in range(1,len(args)+1):
                names=locals()
                names['y%s'%n] = factors_df_list[n-1][tdate]
            
            ax1 = plt.figure(figsize=(16, 9)).add_subplot(1,1,1)
            #fig,ax1 = plt.subplots(figsize=(16, 9))
            eval('y1').plot(label='PE',ax=ax1,style='bo-',alpha=0.8,kind='bar',grid=True) #Series格式画图
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
            eval('y2').plot(label='PB',ax=ax2,style='ro-',alpha=0.8,kind='line')
            eval('y3').plot(label='PS',ax=ax2,style='go-',alpha=0.8,kind='line')
            #右边x轴设置
            ax2.set_xlim([-0.5,len(x_ticks)-0.5])  #x轴范围，包含27.5
            #右边y轴设置
            ax2.set_ylabel('PB&PS values', fontsize=16)
            ax2.set_ylim([0,18])
            ax2.set_yticks(np.arange(0,19))  #y轴标签刻度位置
            ax2.set_yticklabels(np.arange(0,19),fontsize=16)  #y轴标签
            szzs_tdate = DateProcess.tradedays.loc[DateProcess.tradedays['date'] ==tdate,'close'].values[0]
            plt.title("沪深A股申万一级行业分类PE、PB、PS比较（{0}(上证综指:{1}))".format(tdate,szzs_tdate),fontsize=20) #标题
            plt.legend(loc=1)
            plt.savefig(add_pic+'pe_pb_ps_{0}.png'.format(tdate),dpi=400,bbox_inches='tight')
    
    #画行业因子均值的热力图
#    @staticmethod
#    def heat_map(df):
#        f, ax = plt.subplots(figsize=(32, 9))
##        plt.xticks(rotation='90')
#        sns.heatmap(df, square=True, linewidths=.5, annot=True)
#        plt.show()
    
    def factor_mkt_value(self,factor):
        '''按流通市值进行行业中性分组求因子均值，得到同年度排序结果,流通市值倒序排列'''
        mid_factor = Clean_Data(factor).Median_deextremum()
        factor_df = pd.DataFrame(mid_factor,columns=self.trade_date[0],index=self.stockcode[:,0])
        factor_df = factor_df.groupby([x[:4] for x in factor_df.columns],axis=1).last()
        float_mv_df = pd.DataFrame(self.float_mv,columns=self.trade_date[0],index=self.stockcode[:,0])
        float_mv_df = float_mv_df.groupby([x[:4] for x in float_mv_df.columns],axis=1).last()
        
        fac_indus_res = pd.DataFrame(index = factor_df.columns,
                                   columns=['group1','group2','group3','group4','group5']) #资金变化结果df
        for n in range(factor_df.shape[1]):
            mid_column = float_mv_df.columns[n]
            mid_float_mv = float_mv_df[[mid_column]]
            mid_factor = factor_df[[mid_column]]
            mid_df = pd.merge(mid_factor,mid_float_mv,left_index=True,right_index=True,how='inner')
            mid_df = pd.merge(mid_df,self.industry,left_index=True,right_on='code',how='inner')
            mid_df.rename(columns={mid_df.columns[0]:'factor',
                                   mid_df.columns[1]:'float_mv'},inplace = True)
            grouped =mid_df.groupby(by='industry_1class') #按行业分组
            grouped_df = pd.DataFrame(columns=['factor','float_mv','group_NO'])
            for indus_name,value in grouped:
                value.sort_values(by='float_mv',ascending=False,inplace=True)
                value.dropna(inplace=True)
                value.reset_index(drop=True,inplace=True)
                group_amount = int(value.shape[0]/5) #每组股票数量
                for group_n in range(5):
                    if group_n==4:
                        mid_indus_group = value.loc[group_n*group_amount:,['factor','float_mv']]
                        mid_indus_group['group_NO'] = 5
                        grouped_df = grouped_df.append(mid_indus_group)
                    else:
                        mid_indus_group = value.loc[group_n*group_amount:(group_n+1)*group_amount-1,['factor','float_mv']]
                        mid_indus_group['group_NO'] = group_n+1
                        grouped_df = grouped_df.append(mid_indus_group)                          
            fac_indus_res.iloc[n,:] = grouped_df.groupby('group_NO')['factor'].mean().values
        return fac_indus_res
    
    @staticmethod
    def draw_mkt_value(fac_indus_res,fac_name):
        font = {'rotation' : 0,   # 旋转30度
                 'fontsize' : 12,  # 字体大小
                }
        ax1 = plt.figure(figsize=(16, 9)).add_subplot(1,1,1)
        fac_indus_res.plot(ax=ax1,kind='bar',grid=True)
        ax1.set_xlabel('年份', fontsize=16) #x轴名称
        ax1.set_xticklabels(range(2009,2019),font) #x轴标签
        ax1.set_ylabel('因子均值', fontsize=16) #x轴名称
        plt.title('按流通市值申万一级行业分类中性分组{0}因子均值'.format(fac_name),fontsize=20) #标题
        plt.legend(loc='best')
        plt.savefig(add_pic+'{0}.png'.format(fac_name),dpi=400,bbox_inches='tight')
        
        
if __name__=='__main__':
    # 申万一级行业因子值分布图;
    drawing = Single_factors_draw()
    pe_p = drawing.mean_industry_process(pe)
    pb_p = drawing.mean_industry_process(pb)
    ps_p = drawing.mean_industry_process(ps)

    # 画图
    drawing.draw(pe_p,pb_p,ps_p)

    # 年末因子均值按申万一级行业排名
    pe_ord = drawing.ordinal_industry_process(pe_p)
    pb_ord = drawing.ordinal_industry_process(pb_p)
    ps_ord = drawing.ordinal_industry_process(ps_p)

    #按每年最后一个月的流通市值对因子排序，然后分组取均值，做申万一级行业中性处理
    drawing.draw_mkt_value(drawing.factor_mkt_value(pe),'pe')
    drawing.draw_mkt_value(drawing.factor_mkt_value(pb),'pb')
    drawing.draw_mkt_value(drawing.factor_mkt_value(ps),'ps')

    #因子与流动市值的相关系数
    fac_l = ['ep','bp','roe','roic','arturn','tagr','debttoasset','revs60','rstr12']
    corr_df = pd.DataFrame(index = fac_l,columns = [*range(2009,2019),'mean_corr'])
    for f in fac_l:
        mid_corr = drawing.fac_corr(eval(f),float_mv)
        corr_df.loc[f,:] = [*mid_corr,round(sum(mid_corr)/len(mid_corr),4)]
    tmp = corr_df.copy()
    tmp.reset_index(inplace=True)
    
    #因子之间的相关系数矩阵
    fac_l2 = ['ep','bp','roe','roic','arturn','tagr','debttoasset','revs60','rstr12']
    res_dic = dict()
#    #每年最后一个交易日
#    date_df = pd.DataFrame(drawing.trade_date,index=['date']).T
#    date_ylast = date_df.groupby([x[:4] for x in date_df.date]).last()
    for tyear in range(2009,2019):
        corr_df2 = pd.DataFrame(index = fac_l2,columns = fac_l2)
        for f2 in fac_l2:
            fac2 = drawing.sub_fac_corr(eval(f2))
            for f3 in fac_l2:
                fac3 = drawing.sub_fac_corr(eval(f3))
                corr_df2.loc[f2,f3] = fac2.loc[:,str(tyear)].corr(fac3.loc[:,str(tyear)])
        res_dic[tyear] =  Clean_Data.Round_df(corr_df2)
        print (tyear,'done')













