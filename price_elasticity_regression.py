#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:26:13 2018

@author: fabien.baker
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy.optimize import curve_fit
import pandas as pd 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
np.random.seed(1729)
#df = read_csv('')
#hcm_cross = pd.read_csv('hcm_gb_poly_20180720.csv')


# CREATE ARRAY FOR THE DATA

x_cebu = np.array([0.95,0.975,1.0,1.05])

y_btr_cebu = np.array([0.3816,0.3786,0.370,0.367])

y_rpv_cebu = np.array([ 45.29,45.75,46.30,48.53])
y_gmv_cebu = np.array([ 12104269,12315767,12466316,12582735])

x_rain = np.array([0.00,0.12,1.0,3.66])

y_efta = np.array([0.72,0.50,0.49,0.45])
y_btr = np.array([0.38,0.37,0.36,0.35])



##############################################################


###############################################################


#FUNCTIONS 
def func_exp(x, a, b, c):
    return a * np.exp(-b * x) + c

def func_two(x, b, c, d):
    return b*x**2 +c*x + d

def func_three(x, b, c, d):
    return b*x**3 +c*x + d

def func_poly(x ,a ,b, c, d):
    return a*x**3 + b*x**2 +c*x + d


def func_log(x, a,b,c):
    return (b*np.log(x)+ c)

def func_log_two(x,a,b,c):
      return a * np.log(b * x) + c



############################################################

#FITS A CURVE TO THE FUNCTIONS 
popt_btr ,pcov_btr = curve_fit(func, x_rain, y_efta)
popt_rpv ,pcov_btr = curve_fit(func, x_rain, y_efta)

#popt_rpv ,pcov_rpv = curve_fit(func_log_two, x_rain, y_btr)
#popt_ar ,pcov_ar = curve_fit(func_two, x_hcm, y_ar_hcm)
#popt_dau ,pcov_dau = curve_fit(func_log, x_d, y_hanoiPriceElasticity)


# PRINTS THE FUNCTION HYPER PARAMETERS
print("Rain efta = %s , b = %s, c = %s" % (popt_btr[0], popt_btr[1], popt_btr[2]))
print("    BTR a  = %s , b = %s, c = %s" % (popt_rpv[0], popt_rpv[1], popt_rpv[2]))
                                                                                                               


# GENERATE FAKE 50 OF POTENTIAL MULTIPLIERS BETWEEN X0.5 AND X1.9
x = np.linspace(0.0, 3.9,50)
x_dau = np.linspace(2,90,88)

# SAVES THE REGRESSION FOR BTR AND RPV AS VARIABLE SO IT CAN BE EXTRACTED INTO EXCEL  
y_btr_func = func_two(x, *popt_btr)
y_rpv_func = func_three(x, *popt_rpv)
#y_ar_func = func_two(x, *popt_ar)
#y_dau_func = func_log(x_dau, *popt_dau)




fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()
par2 = host.twinx()

par1.set_ylim(0, 0.9)

host.set_xlabel("Precip. mm 5 minute")
host.set_ylabel("EFTA 1")
par1.set_ylabel("BTR 2")
#par2.set_ylabel("Allocation Rate")


color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.9)

xs = sym.Symbol('\lambda')    
tex = sym.latex(func_two(xs,*popt_btr)).replace('$', '')
plt.title('Rain fall impact EFTA & BTR 2wheel' ,fontsize=12)




p1, = host.plot(x, y_btr_func,alpha=0.5, color=color1)
p1, = host.plot(x, y_btr_func,'-', color='b',label="EFTA")
p2, = par1.plot(x, y_rpv_func, color='g', label="EFTA 2")
#p3, = par2.plot(x, y_ar_func, color='g',label="AR")

lns = [p1, p2]
host.legend(handles=lns, loc='upper right')


par2.spines['right'].set_position(('outward', 60))      
plt.savefig("pyplot_multiple_y-axis.png", bbox_inches='tight')

