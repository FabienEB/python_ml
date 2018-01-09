# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:33:07 2018

@author: fbaker
"""

import random as rn


def steve(range_of_guess):
    winner = 0
    correct_number = rn.randrange(1,range_of_guess,1)
    guesstaken = 0 
    guess_quadrent = int(range_of_guess/2)

    last_high = "x"
    last_low = "x"
    guessarray = []
    
    
    
    
    while winner == 0: 
       guess = guess_quadrent
       guesstaken = guesstaken + 1 
       if guess < correct_number:
           if last_high == "x": 
               last_low = guess_quadrent            
               guess_quadrent = guess_quadrent + 25

           if last_high != "x": 
               last_low = guess_quadrent            
               guess_quadrent = guess_quadrent+int((last_high - guess_quadrent ) / 2) 
                            
           #print(guess)
           #print('too low')
        
       if guess > correct_number:  
           if last_low == "x": 
               last_high = guess_quadrent        
               guess_quadrent = guess_quadrent - 25
            
           if last_low != "x":   
               last_high = guess_quadrent            
               guess_quadrent = int((guess_quadrent- last_low) / 2) + last_low

           #print(guess)
           #print('too high')    
        
       if guess == correct_number: 
           winnner = 1
           #print('-------')
           #print(guess)
           print(guesstaken)
           #print('winner!')
           guessarray.append(guesstaken)
           break
    
    
