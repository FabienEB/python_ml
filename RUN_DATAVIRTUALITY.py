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



with DataVirtualityConnection(username,password) as dv_cnxn:

            query_a = """
            SELECT
            visit_date,
            1 as dummy,
            
            
            
            SUM(CASE WHEN source = 'direct' then price else null end )/
            CAST(count(distinct CASE WHEN source = 'direct' then sid else null end) AS float) as a, 

            SUM(CASE WHEN source <> 'direct' then price else null end )/
            CAST(count(distinct CASE WHEN source <> 'direct' then sid else null end) AS float) as b,
            
            SUM(CASE WHEN dvce_type <> 'direct' and dvce_type = 'Mobile' then price else null end )/
            CAST(count(distinct CASE WHEN source <> 'direct' and dvce_type = 'Mobile' then sid else null end) AS float) as c,
            
            SUM(CASE WHEN dvce_type <> 'direct' and dvce_type = 'Mobile' then price else null end )/
            CAST(count(distinct CASE WHEN source <> 'direct' and dvce_type = 'Mobile' then sid else null end) AS float) as d,
            
           SUM(CASE WHEN dvce_type <> 'direct' and dvce_type = 'Mobile' and visit_number = 1 then price else null end )/
            CAST(count(distinct CASE WHEN source <> 'direct' and dvce_type = 'Mobile' and visit_number = 1 then sid else null end) AS float) as d2,
            
            SUM(CASE WHEN dvce_type <> 'direct' and dvce_type = 'Desktop' then price else null end )/
            CAST(count(distinct CASE WHEN source <> 'direct' and dvce_type = 'Desktop' then sid else null end) AS float) as e1,
            
            SUM(CASE WHEN dvce_type <> 'direct' and dvce_type = 'Desktop' and visit_number = 1 then price else null end )/
            CAST(count(distinct CASE WHEN source <> 'direct' and dvce_type = 'Desktop' and visit_number = 1 then sid else null end) AS float) as e2,
            
            SUM(CASE WHEN dvce_type <> 'direct' and dvce_type = 'Desktop' and website_language = 'en' then price else null end )/
            CAST(count(distinct CASE WHEN source <> 'direct'  and dvce_type = 'Desktop' and website_language = 'en' then sid else null end) AS float) as f,
            
            SUM(CASE WHEN dvce_type <> 'direct' and dvce_type = 'Mobile' and website_language = 'en' then price else null end )/
            CAST(count(distinct CASE WHEN source <> 'direct'  and dvce_type = 'Mobile' and website_language = 'en' then sid else null end) AS float) as f2,
             
            SUM(CASE WHEN dvce_type <> 'direct' and website_language <> 'en' then price else null end )/
            CAST(count(distinct CASE WHEN source <> 'direct' and website_language <> 'en' then sid else null end) AS float) as g,
            
            SUM(CASE WHEN dvce_type <> 'direct' and dvce_type = 'Desktop' and website_language <> 'en' then price else null end )/
            CAST(count(distinct CASE WHEN source <> 'direct' and dvce_type = 'Desktop' and website_language <> 'en' then sid else null end) AS float) as g2
            
            from "snowplow.visitor_funnel_basic" 
            where visit_date between '2017-10-01' and '2017-11-06'
            
              --where month(visit_date) in (09,10,11) 
              --and year(visit_date) in (2016,2017)
              --and visit_date < '2017-11-07'
            group by 1
            order by visit_date asc
            LIMIT %(limit)s
            """
            data = dv_cnxn.query(query_a, parameters={'limit': 10000}).get_as_dataframe()
            data.to_pickle('data')
            
y = data['a']
b = data['b']
c = data['c']
d = data['d']
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

