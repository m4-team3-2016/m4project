#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 19:44:43 2017

@author: group 3
"""

import numpy as np
import matplotlib.pyplot as plt


def computeLineEquation(pointLine, isPlotEnable=False):
    # Define the known points
    x = [pointLine1["Traffic"][0][0], pointLine1["Traffic"][1][0]]
    y = [pointLine1["Traffic"][0][1], pointLine1["Traffic"][1][1]]
    
    # Calculate the coefficients.
    coefficients = np.polyfit(x, y, 1)
    
    # Print the findings
    if isPlotEnable:
        print 'a =', coefficients[0]
        print 'b =', coefficients[1]
        
        # Compute the values of the line
        polynomial = np.poly1d(coefficients)
        x_axis = np.linspace(0,500,100)
        y_axis = polynomial(x_axis)
        
        # Plot the points and the line
        plt.plot(x_axis, y_axis)
        plt.plot( x[0], y[0], 'go' )
        plt.plot( x[1], y[1], 'go' )
        plt.grid('on')
        plt.show()
    
    return coefficients


if __name__ == "__main__":
    # Example of usage
    pointLine1 = {}
    pointLine2 = {}
    pointLine1["Traffic"] = [[78,194],[295,68]]
    pointLine2["Traffic"] = [[14,99],[172,8]]

    # Compute the coefficients    
    coef = computeLineEquation(pointLine2, True)
    
