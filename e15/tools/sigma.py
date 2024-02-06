"""
Calculation of Cross Helicity and Residual Energy
Tamar Ervin
August 3, 2023
"""

import numpy as np

def smooth(data, num):
    return np.convolve(data, np.ones(num)/num, mode='same')

def calculate_delta(data, smoothed_data):
    return data - smoothed_data

def calculate_velocity_change(delta_b, nppin):
    return delta_b * 1e-9 / np.sqrt(1.25e-6 * nppin * 1e6 * 1.67e-27) / 1000.0

def calc_vA(B, nppin):
    return B * 1e-9 / np.sqrt(1.25e-6 * nppin * 1e6 * 1.67e-27) / 1000.0

def calculate_sigma_c(zp_square, zm_square):
    return (zp_square - zm_square) / (zp_square + zm_square)

# def calculate_sigma_r(Zpr, Zmr, Zpt, Zmt, Zpzp_square, zm_square):
#     sigmar = 2*(Zpr*Zmr + Zpt*Zmt + Zpn*Zmn)/(zp_square+zm_square)
#     return 2 * (zp * zm) / (zp + zm)

def calc_sigma(dataframe, num=5):

    # smooth and calculate velocity fluctuation
    Vr_smo = smooth(dataframe.vr, num)
    Vt_smo = smooth(dataframe.vt, num)
    Vn_smo = smooth(dataframe.vn, num)
    delta_Vr = calculate_delta(dataframe.vr, Vr_smo)
    delta_Vt = calculate_delta(dataframe.vt, Vt_smo)
    delta_Vn = calculate_delta(dataframe.vn, Vn_smo)

    # smooth and calculate magnetic field fluctuation
    Br_smo = smooth(dataframe.Br, num)
    Bt_smo = smooth(dataframe.Bt, num)
    Bn_smo = smooth(dataframe.Bn, num)
    delta_Br = calculate_delta(dataframe.Br, Br_smo)
    delta_Bt = calculate_delta(dataframe.Bt, Bt_smo)
    delta_Bn = calculate_delta(dataframe.Bn, Bn_smo)
    
    # calculate Alfven speed
    Vra = calc_vA(dataframe.Br, dataframe.use_dens.values)
    Vta = calc_vA(dataframe.Bt, dataframe.use_dens.values)
    Vna = calc_vA(dataframe.Bn, dataframe.use_dens.values)
    vA = np.linalg.norm([Vra, Vta, Vna], axis=0)

    # calculate magnetic field in velocity units
    delta_Vrb = calculate_velocity_change(delta_Br, dataframe.use_dens.values)
    delta_Vtb = calculate_velocity_change(delta_Bt, dataframe.use_dens.values)
    delta_Vnb = calculate_velocity_change(delta_Bn, dataframe.use_dens.values)

    Zpr = delta_Vr.values + delta_Vrb.values
    Zpt = delta_Vt.values + delta_Vtb.values
    Zpn = delta_Vn.values + delta_Vnb.values
    Zmr = delta_Vr.values - delta_Vrb.values
    Zmt = delta_Vt.values - delta_Vtb.values
    Zmn = delta_Vn.values - delta_Vnb.values

    Zpsquare = Zpr**2 + Zpt**2 + Zpn**2
    Zmsquare = Zmr**2 + Zmt**2 + Zmn**2

    deltav = np.sqrt(delta_Vr**2 + delta_Vt**2 + delta_Vn**2)
    deltab = np.sqrt(delta_Vrb**2 + delta_Vtb**2 + delta_Vnb**2)

    sigmac = calculate_sigma_c(Zpsquare, Zmsquare)
    sigmar = 2*(Zpr*Zmr + Zpt*Zmt + Zpn*Zmn)/(Zpsquare+Zmsquare)

    return sigmac, sigmar, vA, np.sqrt(Zpsquare), np.sqrt(Zmsquare), deltav, deltab