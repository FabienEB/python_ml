#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 10:58:28 2018

@author: fabien.baker
"""

# Uncomment below line to install alpha_vantage
#!pip install alpha_vantage

#https://www.alphavantage.co/documentation/
import  datetime 
from dateutil.parser import parse
import pandas as pd 
import time  
import pandas as pd 
import numpy as np 


###########################################
# INPUTS 

exp_date =  '2018-09-20 17:59:00'
fmt = '%Y-%m-%d %H:%M:%S'
cur_date = datetime.datetime.now()

###########################################



import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries

ts = TimeSeries(key='32C7UUB73NUWCH4I', output_format='pandas')
data, meta_data = ts.get_intraday(symbol='WOWLI8.AX',interval='10min', outputsize='compact')
#print(data.head())

timeseries = data.index 

plt.plot(timeseries,data['4. close'])


###########################################

def call_price(st,sr,cur_date,exp_date,vol,rr=0.03):
    d1 = np.log(st/sr)+(rr+vol^2/2)*cur_date/(vol*np.sqrt(cur_date))
    return d1

    




