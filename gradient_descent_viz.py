#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:03:48 2018

@author: fabien.baker
"""

# GRADIENT DESCENT 


import numpy as np
import matplotlib.pyplot as plt

def func_y(x):
    y = x**2 - 4*x + 2

    return y

def gradient_descent(previous_x, learning_rate, epoch):
    x_gd = []
    y_gd = []
    
    x_gd.append(previous_x)
    y_gd.append(func_y(previous_x))

    # begin the loops to update x and y
    for i in range(epoch):
        current_x = previous_x - learning_rate*(2*previous_x - 4)
        x_gd.append(current_x)
        y_gd.append(func_y(current_x))

        # update previous_x
        previous_x = current_x

    return x_gd, y_gd

# Initialize x0 and learning rate
x0 = -0.7
learning_rate = 0.15
epoch = 10


# y = x^2 - 4x + 2
x = np.arange(-1, 5, 0.01)
y = func_y(x)

fig, ax = plt.subplots(1, 2, sharey = True)

ax.plot(x, y, lw = 0.9, color = 'k')
ax.set_xlim([min(x), max(x)])
ax.set_ylim([-3, max(y)+1])
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

x_gd, y_gd = gradient_descent(x0, learning_rate, epoch)


ax.scatter(x_gd, y_gd, c = 'b')

for i in range(1, epoch+1):
    ax.annotate('', xy=(x_gd[i], y_gd[i]), xytext=(x_gd[i-1], y_gd[i-1]),
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')

plt.show()