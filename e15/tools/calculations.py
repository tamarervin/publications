"""
Tamar Ervin
September 15, 2023

Calculation functions
"""

import numpy as np
import scipy.constants as con


##### ---------- PLASMA BETA CALCULATION  ---------- ######

def calc_beta(B_RTN, T, n):
    """
    plasma beta calculation
    
    B_RTN: [Br, Bt, Bn] 
    T: temperature in eV
    n: number density in cm^-3
    """

    Br, Bt, Bn = B_RTN[0], B_RTN[1], B_RTN[2]
    B = np.sqrt(Br**2 + Bt**2 + Bn**2) ## B magnitude
    B = (B*(10**(-9))) ## nT to T
    Tt = T/11605 ## eV to Kelvin
    nn = n/(1e-6) ##cm^-3 to m^-3

    beta = 2*con.mu_0*nn*con.Boltzmann*Tt/(B**2) ## beta = 2 mu0 n k_B T /B^2

    return beta


##### ---------- PSD OF TIMESERIES  ---------- ######
def get_psd(time_series, sampling_frequency=3.0):
    # Compute the FFT
    fft_result = np.fft.fft(time_series)

    # Calculate the power spectrum
    power_spectrum = np.abs(fft_result)**2

    # Determine the sampling frequency and time step
    time_step = 1.0 / sampling_frequency

    # Calculate the frequencies
    frequencies = np.fft.fftfreq(len(time_series), time_step)

    # Calculate the Power Spectral Density (PSD)
    psd = power_spectrum / sampling_frequency

    return frequencies, psd
