#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:28:56 2018

@author: fabien.baker
"""

import webbrowser
import time 
    

'''

strg = input("How long do you want to focus?").lower()
y = "30"
for x in strg:
    if x ==  "n":
        print("Then why aren't you watching this ??")
        time.sleep(2) 
        webbrowser.open('https://www.google.com.sg/search?q=FOCUS!!&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjFl_2KxN3eAhXKP48KHYjPDY4Q_AUIDigB&biw=1680&bih=851#imgrc=iLmZtXFD9VWUGM:') 
    if x == y:
            print("GOOOOD LUCK")
            time.sleep(0.5) 
            print("GOOOOD LUCK")
            time.sleep(1) 
            print("GOOOOD LUCK")
            time.sleep(0.3) 
            print("GOOOOD LUCK")

    else :print('fabiens code doesnt work this is an error.... !') 
'''



def are_you_focused(y):
    print(" Hello Viva...")
    strg = input("How long do you want to focus??:  ").lower()
    start = time.time()
    time.sleep(5) 
    for x in strg: 
        end = time.time()
        x = (end - start)
        if x < y:
            time.sleep(2) 
            webbrowser.open('https://www.google.com.sg/search?q=FOCUS!!&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjFl_2KxN3eAhXKP48KHYjPDY4Q_AUIDigB&biw=1680&bih=851#imgrc=iLmZtXFD9VWUGM:') 
        else : print("1")
            
        