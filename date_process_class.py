# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 10:40:36 2018

@author: wuwangchuxin
"""
import pandas as pd

class DateProcess():
    # 截止到2018-08-08
    tradedays = pd.read_excel(r'C:\Users\wuwangchuxin\Desktop\TF_SummerIntern\MF_data\market_tradedate.xlsx')
    def __init__(self,date):
        self.date=date
    # 将类似'2017/12/20'形式 转化为 '2017-12-20'形式
    def format_date(self):
        if len(self.date) == 10:
            return self.date[:4]+'-'+self.date[5:7]+'-'+self.date[8:]
        elif len(self.date) == 8:
            return self.date[:4]+'-0'+self.date[5]+'-0'+self.date[-1]
        elif self.date[-2] == r'/':
            return self.date[:4]+'-'+self.date[5:7]+'-0'+self.date[-1]
        else:
            return self.date[:4]+'-0'+self.date[5]+'-'+self.date[-2:]
    # 求某一交易日向前或者向后差若干交易日的交易日   
    def tdays_offest(self,num):
        td = self.format_date()
        return str(self.tradedays.loc[self.tradedays[self.tradedays['date']==td].index+num,\
                                      'date'].reset_index(drop=True)[0])[:10]        

if __name__=='__main__':
    d = DateProcess('2017/12/20')
    d.format_date()
    d.tdays_offest(-3)