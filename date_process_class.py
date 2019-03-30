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
    def format_date(self):
        '''将类似'2017/12/20'形式 转化为 '2017-12-20'形式'''
        if len(self.date) == 10:
            return self.date[:4]+'-'+self.date[5:7]+'-'+self.date[8:]
        elif len(self.date) == 8:
            return self.date[:4]+'-0'+self.date[5]+'-0'+self.date[-1]
        elif self.date[-2] == r'/':
            return self.date[:4]+'-'+self.date[5:7]+'-0'+self.date[-1]
        else:
            return self.date[:4]+'-0'+self.date[5]+'-'+self.date[-2:]
       
    def tdays_offest(self,num):
        '''求某一交易日向前或者向后差若干交易日的交易日
           num>0,朝后；num<0,朝前'''
        try:
            td = self.format_date()
            return str(self.tradedays.loc[self.tradedays[self.tradedays['date']==td].index+num,
                                          'date'].reset_index(drop=True)[0])[:10]
        except:
            raise ValueError('not tradeday')
    # 求某一交易日向后若干个月的交易日
    def tmonths_offset(self,num):
        '''得到当前交易日若干月之后的第一个交易日，
           因为tradedays最后一天为2018-09-19,如果超出则返回2099-12-31'''
        if not (isinstance(num,int) and num>0):
            raise ValueError('please input a positive integer')
        else:
            td = self.format_date()
            if int(td[5:7])<=12-num:
                mid = td[:5]+str(int(td[5:7])+num)+td[7:]
            else:
                mid = str(int(td[:4])+1)+'-'+str(int(td[5:7])+num-12)+td[7:]
            if len(mid)<10:
                mid = mid[:5]+'0'+mid[5:]
            if mid<=str(max(self.tradedays['date']))[:10]:
                return str(min(self.tradedays['date'][
                            pd.Series(str(x)[:10] for x in self.tradedays['date'])>=mid]))[:10]
            else:
                return '2099-12-31'
        
if __name__=='__main__':
    d = DateProcess('2017/12/1')
    d.format_date()
    d.tdays_offest(-3)
    d.tmonths_offset(3)
    
    d1 = DateProcess('1993/11/30')
    d1.format_date()
    d1.tdays_offest(-3)
    d1.tmonths_offset(3)    
    
    