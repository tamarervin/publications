'''
Tamar Ervin
Compute power spectra 

'''

# imports

import glob
import pyspedas
from pyspedas import time_string, time_double
from pytplot import tplot, get_data, cdf_to_tplot, store_data
import astrospice
import sunpy 
import sunpy.coordinates as scoords
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from scipy import stats


import sys, os
import datetime
import numpy as np
sys.path.append(os.path.realpath(''))
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt

from tools.plotting import plot_pfss
from tools.settings import CsvDir, ImgDir, PlotDir, DataDir
import matplotlib.ticker as ticker
from scipy.signal.windows import tukey

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from datetime import timedelta

### read in dataframe
turb = pd.read_csv('dataframe.csv')

### choose time frame of interest
use1 = np.logical_and(turb.Time >= pd.Timestamp('2023-03-16 12:00'), turb.Time <= pd.Timestamp('2023-03-16 18:00:00'))
t1 = turb[use1].copy()

### create multiple dataframes
ss, ii = 1400,3
tZp = [t1.Zp[i*ss:i*ss+ss] for i in range(0, ii)]
tZm = [t1.Zm[i*ss:i*ss+ss] for i in range(0, ii)]
tdv = [t1.deltav[i*ss:i*ss+ss] for i in range(0, ii)]
tdb = [t1.deltab[i*ss:i*ss+ss] for i in range(0, ii)]

### calculate PSD with Tukey window
def windowed_psd(series):
    dff, dfp = pd.DataFrame(), pd.DataFrame()
    plt.figure()
    col=['red', 'green', 'blue', 'orange', 'purple', 'black']
    for i, time_series in enumerate(series):
        ### RESAMPLE DATA
        sample_rate = 1/5  # sample rate in seconds
        sample_freq = 1/sample_rate # sample frequency in Hz
        N = len(time_series)

        ### Create a Tukey window
        alpha = 0.5 
        tukey_window = tukey(N, alpha)
        windowed_time_series = time_series * tukey_window
        
        # Compute the FFT of the windowed time series
        fft_result = np.fft.fft(windowed_time_series) / N ## calculate FFT and normalize
        fft_freq = np.fft.fftfreq(len(fft_result), sample_freq)
        
        # Calculate the power spectral density
        power_spectrum = (2*N/sample_freq) * np.abs(fft_result)**2

        # normalize by the tukey window
        ww = 1/2 * (1 - np.cos((2 * np.pi * np.arange(0, N * alpha / 2)) / (alpha * N)))
        ww1 = 1/2 * (1 - np.cos((2 * np.pi * np.arange(N / 2 + N * alpha / 2, N)) / (alpha * N)))

        # Create the piecewise function
        y1 = ww
        x2 = np.arange(N * alpha / 2, N / 2 + N * alpha / 2)
        y2 = np.ones(len(x2))
        y3 = ww1

        ### CALCULATE Wss
        Wss = np.sum(np.concatenate((y1, y2, y3)) ** 2) / N
        power_spectrum /= Wss

        # Add to average
        dff[str(i)] = np.abs(fft_freq)
        dfp[str(i)] = power_spectrum
        
        plt.loglog(dff[str(i)], dfp[str(i)] , color=col[i])

    dfp['freq'] = dff.mean(axis=1)
    dfp['power'] = dfp.mean(axis=1)

    return dfp


### SPECTRAL FITTING
import numpy as np
from scipy.optimize import curve_fit
import tabulate

# Define your equations
def equation_p(x, Cp, Cinf):
    return Cp * (x**(-3/2)) + Cinf * (x**(-5/3))

def equation_m(x, Cm, Cinf, kt):
    return Cm * (x**(-3/2)) * np.sqrt(1 + np.sqrt(x/kt)) + Cinf * (x**(-5/3))

# Define the fit functions with a shared Cinf parameter
def fit_function_p(x, Cp, shared_Cinf):
    return equation_p(x, Cp, shared_Cinf)

def fit_function_m(x, Cm, shared_Cinf, kt):
    return equation_m(x, Cm, shared_Cinf, kt)

def var_p(x_values, covariance1):
    # Calculate the partial derivatives of Y with respect to Cp and Cinf
    partial_derivative_Cp = x_values**(-3/2)
    partial_derivative_Cinf = x_values**(-5/3)

    # Calculate the variance of Y
    variance_Y = (partial_derivative_Cp**2 * covariance1[0, 0] +
                partial_derivative_Cinf**2 * covariance1[1, 1] +
                2 * partial_derivative_Cp * partial_derivative_Cinf * covariance1[0, 1])

    # Calculate the standard error of Y
    standard_error_p = np.sqrt(variance_Y)
    return standard_error_p

def var_m(x_values, covariance2, kt_fit):
    partial_derivative_Cm = x_values**(-3/2) * np.sqrt(1 + np.sqrt(x_values/kt_fit))
    partial_derivative_Cinf = x_values**(-5/3)

    # Calculate the variance of Y without considering the error in kt
    variance_Y = (partial_derivative_Cm**2 * covariance2[0, 0] +
                  partial_derivative_Cinf**2 * covariance2[1, 1] +
                  2 * partial_derivative_Cm * partial_derivative_Cinf * covariance2[0, 1])

    # Calculate the standard error of Y
    standard_error_m = np.sqrt(variance_Y)
    return standard_error_m
