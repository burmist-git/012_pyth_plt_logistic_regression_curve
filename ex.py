#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Date        : Wed Dec 16 12:31:50 CET 2020
Autor       : Leonid Burmistrov
Description : Simple reminder-training example.
'''

import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(1234567890)

def sigmoidFunction(k,x):
    pol=polynomialFunction(k,x)
    return 1/(1+np.exp(-pol))

def polynomialFunction(k,x):
    return np.dot(k,x)

def plot(x,y,y_s):
    # using the variable axs for multiple Axes
    fig, axs = plt.subplots( nrows=2, ncols=1,
                             sharex=False, sharey=False,
                             squeeze=True, subplot_kw=None,
                             gridspec_kw=None,
                             figsize=(10,12))
    #plt.tight_layout()

    # 
    axs[0].plot(x, y)
    #axs[0].set_xbound(-0.5,5.5)
    #axs[0].set_ybound(-0.5,5.5)
    #axs[0].set_title('The title 0')
    #axs[0].set_xlabel('The x lable 0')
    #axs[0].set_ylabel('The y lable 0')
    axs[0].set_position([0.05,0.55,0.9,0.40])
    #
    axs[1].scatter(x, y_s)
    #axs[1].set_xbound(-0.5,5.5)
    #axs[1].set_ybound(-0.5,5.5)
    #axs[1].set_title('The title 1')
    #axs[1].set_xlabel('The x lable 1')
    #axs[1].set_ylabel('The y lable 1')
    axs[1].set_position([0.05,0.05,0.9,0.40])

    plt.show()

def main():

    nPoints = 10000
    xMin = -5
    xMax = 15

    #k = np.expand_dims(np.array([0,1]),axis=0)
    #x_ones = np.expand_dims(np.ones(nPoints),axis=0)
    #x = np.expand_dims(np.linspace(xMin, xMax, nPoints),axis=0)
    #
    #k = np.array([10])
    k = np.array([10,10,-10.0,1.0])
    #k = np.array([100,10,-10,1.2])
    assert (len(k)>1);
    #
    x_ones = np.ones(nPoints)
    x = np.linspace(xMin, xMax, nPoints)
    
    x = np.row_stack((x_ones,x))

    if(len(k)>2):
        for i in range(len(k)-2):
            x = np.row_stack((x,x[:][-1]*x[:][1]))
            
    print('')
    print('k.ndim  = ', k.ndim)
    print('k.shape = ', k.shape)
    print('k       = ', k)
    print('')
    print('x.ndim  = ', x.ndim)
    print('x.shape = ', x.shape)
    #print('x       = ', x)
    print('')
    print('x_ones.ndim  = ', x_ones.ndim)
    print('x_ones.shape = ', x_ones.shape)
    #print('x_ones       = ', x_ones)
    #
    try:
        y_dot = np.dot(k,x)
        print('')
        print('y_dot.ndim  = ', y_dot.ndim)
        print('y_dot.shape = ', y_dot.shape)
        #print('y_dot       = ', y_dot)
    except ValueError:
        print('')
        print('np.dot(k,x) --> ValueError')
    #
    try:
        y_star = k * x
        print('')
        print('y_star.ndim  = ', y_star.ndim)
        print('y_star.shape = ', y_star.shape)
        #print('y_star       = ', y_star)
    except ValueError:
        print('')
        print('k * x --> ValueError')
    #
    try:
        y_at = k @ x
        print('')
        print('y_at.ndim  = ', y_at.ndim)
        print('y_at.shape = ', y_at.shape)
        #print('y_at       = ', y_at)
    except ValueError:
        print('')
        print('k @ x --> ValueError')

    y = polynomialFunction(k,x)
    y_s = sigmoidFunction(k,x)
    print('')
    print('y.ndim  = ', y.ndim)
    print('y.shape = ', y.shape)
    #print('y       = ', y)        

    plot(x[1][:],y,y_s)
    
if __name__ == "__main__":
    main()
