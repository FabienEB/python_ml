

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:21:21 2017

@author: fbaker
"""

import os
import sys
sys.path.append(
    os.path.normpath(os.path.expanduser('~/Documents/GitHub/business-intelligence/python'))
)
from datavirtuality.datavirtuality import DataVirtualityConnection
from utilities import utilities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr as pr
from datavirtuality.datavirtuality import DataVirtualityConnection

with DataVirtualityConnection('fabien_baker','Fa$1bien') as dv_cnxn:

            query_a = """
            SELECT
            visit_date_app,  
            sessions_app,  
            price_app,  
            location_search_app, 
            completed_app,
            sessions_web, 
            price_web, 
            completed_web
             
            FROM(
            SELECT  
            visit_date as visit_date_app,  
            COUNT(DISTINCT sid) as sessions_app,  
            COUNT(DISTINCT CASE WHEN screen_name = 'Vehicle Selection' then sid else null end) as price_app,  
            COUNT(DISTINCT CASE WHEN screen_name = 'Location Search' then sid else null end) as location_search_app, 
            COUNT(DISTINCT completed ) as completed_app
             
            FROM "mixpanel.native_app_visitor_page_view_order" s 
            where visit_date between '2017-11-01' and '2017-11-22'
            and customer_type = 'NC' 
            group by 1
            ) as app
            JOIN ( 
            SELECT 
            visit_date as visit_date_web, 
            COUNT(DISTINCT sid) as sessions_web, 
            COUNT(DISTINCT CASE WHEN price = 1 then sid else null end) as price_web, 
            COUNT(DISTINCT CASE WHEN completed = 1 then sid else null end) as completed_web
            
            FROM "snowplow.visitor_funnel_basic" v
            where visit_date between '2017-11-01' and '2017-11-22'
            and returning_customer_flag = 'NC'
            and dvce_type = 'Mobile'
            and channel <> 'direct_cleaned_cookies'
            group by 1
                ) AS web On web.visit_date_web = app.visit_date_app
            ORDER BY visit_date_app
            LIMIT %(limit)s
            """
            data = dv_cnxn.query(query_a, parameters={'limit': 10000}).get_as_dataframe()
            data.to_pickle('data')
            
    

conversions = data['visit_date_app']
conversions_df = conversions.to_frame()

  
conversions_df['app_lp_search'] = data['location_search_app']/data['sessions_app']
conversions_df['app_lp_price'] = data['price_app']/data['sessions_app']
conversions_df['app_cvr']  = data['completed_app']/data['sessions_app']
conversions_df['web_lp_price'] = data['price_web']/data['sessions_web']
conversions_df['web_cvr'] = data['completed_web']/data['sessions_web']


#print(pr(conversions_df['app_lp_search'],conversions_df['web_lp_price']))
#print(pr(conversions_df['app_lp_price'],conversions_df['web_lp_price']))
#print(pr(conversions_df['app_cvr'],conversions_df['web_cvr']))


x_in = conversions_df['app_lp_search']
y_in = conversions_df['web_lp_price']
  
def least_squares_error_l(x,y):
    Pearson = pr(x,y)
    r = Pearson[0]
    beta = r * np.std(y)/np.std(x)
    alpha = np.mean(y) - beta*np.mean(x)
    return alpha,beta
    
coeff = least_squares_error_l(x_in , y_in)  
alpha_calc = coeff[0]
beta_calc = coeff[1] 
   
print(alpha_calc,beta_calc)


            
"""    

   
price = data['price']
location_search = data['location_search']
sessions = data['sessions']
visit_date = data['visit_date']


d1 = data['d']
d2 = data['d2']
e1 = data['e']
e2 = data['e2']
f1 = data['f']
f2 = data['f2']
g1 = data['g']
g2 = data['g2']

y = y.astype(float)
b = b.astype(float)
c = c.astype(float)
d = d.astype(float)
d1 = d1.astype(float)
d2 = d2.astype(float)
e1 = e1.astype(float)
e2 = e2.astype(float)
f1 = f1.astype(float)
f2 = f2.astype(float)
g1 = g1.astype(float)
g2 = g2.astype(float)

x_i = data[['dummy','b']]

pr1 = pr(b,y)
pr2 = pr(c,y)
pr3 = pr(d,y)
pr4 = pr(d1,y)
pr5 = pr(d2,y)
pr6 = pr(e1,y)
pr7 = pr(e2,y)
pr8 = pr(f1,y)
pr9 = pr(f2,y)
pr10 = pr(g1,y)
pr11 = pr(g2,y)

print(pr1)
print(pr1)
print(pr2)
print(pr3)
print(pr4)
print(pr5)
print(pr6)
print(pr7)
print(pr8)
print(pr9)
print(pr10)
print(pr11)
""" 
