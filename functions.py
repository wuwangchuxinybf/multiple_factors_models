# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:01:59 2018

@author: wuwangchuxin
"""
import pandas as pd

tradedays = pd.read_excel(r'C:\Users\wuwangchuxin\Desktop\TF_SummerIntern\MF_data\market_tradedate.xlsx')
def tdays(td,num):
    return str(tradedays.loc[tradedays[tradedays['date']==td].index+num,'date'].reset_index(drop=True)[0])[:10]


if __name__=='__main__':
    tdays('2018-07-27',-3)