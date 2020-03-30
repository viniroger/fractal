#!/usr/bin/env python3.7.6
# -*- Coding: UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Variable name
var1 = 'glo_avg'

def plot_data(df, var1):
    '''
    Plot input data
    '''
    # Convert from string to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    fig, ax = plt.subplots()
    ax.plot(df['Timestamp'], df[var1], marker='.')
    hours = mdates.HourLocator(interval = 1)
    h_fmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(h_fmt)

    fig.autofmt_xdate()
    plt.savefig('data.png')
    plt.close()

def calc_frac(df, i, var1, dtau, frac_day):
	'''
	Calculate area from 1 rectangle
	'''
	p_k0 = df.iloc[i-1][var1]
	p_k1 = df.iloc[i+dtau][var1]
	rect_area = np.abs(p_k1 - p_k0) * dtau
	series = pd.Series([dtau, rect_area], index=frac_day.columns)
	frac_day = frac_day.append(series , ignore_index=True)
	return frac_day

def calc_Sdtau(df, var1):
    '''
    Calculate area S(dtau)
    '''
    N = df.shape[0]
    dtau_lim = N//2
    frac_day = pd.DataFrame(columns=['dtau','frac'])
    # dtau loop
    for dtau in range(1,dtau_lim):
        #print(dtau)
        # Time series loop
        for i in range(0, df.shape[0], dtau):
            if (i+dtau) < df.shape[0]:
                #print(df.iloc[i]['Timestamp'])
                frac_day = calc_frac(df, i, var1, dtau, frac_day)
    return frac_day

def calc_dim(frac_day):
    '''
    Sum S areas for each dtau
    Calculate fractal dimension using linear regression
    Plot fracs x dtau
    '''
    frac_day = frac_day.groupby('dtau').sum()
    y = np.log(frac_day['frac']/(frac_day.index**2))
    x = np.log(1/frac_day.index)
    coef = np.polyfit(x, y, 1)
    y_fit = np.poly1d(coef)
    y2 = y_fit(x)

    rmse = sqrt(mean_squared_error(y, y2))
    #qui2 = stats.chisquare(y, y2)
    str_coef = 'D=%5.3f\nC=%5.3f' %tuple(coef)
    #str_coef = '%s' %y_fit
    str_rmse = '\nRMSE=%5.3f' %rmse
    textstr = str_coef + str_rmse

    fig, ax = plt.subplots()
    plt.plot(x, y, "*")
    plt.plot(x, y2, "-")
    #plt.show()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top')
    plt.savefig('fit.png')
    plt.close()

    return coef[0]

# MAIN
df = pd.read_csv('input.csv')
plot_data(df, var1)
frac = calc_Sdtau(df, var1)
D = calc_dim(frac)
print('Dimension: %s' %D)
