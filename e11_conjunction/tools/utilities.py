'''
Utilities for plotting and generating paper stats
Tamar Ervin
June 27, 2023
'''

import os
import pandas as pd
import datetime
from pyspedas import time_string
import numpy as np

def read_csv(DF_DIR):
    his_df = pd.read_csv(os.path.join(DF_DIR, 'his_df_full.csv'))
    pas_df = pd.read_csv(os.path.join(DF_DIR, 'pas_df.csv'))
    spi_df = pd.read_csv(os.path.join(DF_DIR, 'spi_df.csv'))
    solo_df = pd.read_csv(os.path.join(DF_DIR, 'solo_df.csv'))
    fields_df = pd.read_csv(os.path.join(DF_DIR, 'fields_df1.csv'))
    pfss_df = pd.read_csv(os.path.join(DF_DIR, 'pfss.csv'))
    temp_df = pd.read_csv(os.path.join(DF_DIR, 'temp_df.csv'))
    alpha_temp_df = pd.read_csv(os.path.join(DF_DIR, 'alpha_temp_df.csv'))
    alpha_density_df = pd.read_csv(os.path.join(DF_DIR, 'alpha_density_df.csv'))
    mldf = pd.read_csv(os.path.join(DF_DIR, 'mldf.csv'))
    sodf = pd.read_csv(os.path.join(DF_DIR, 'mldf_so.csv'))

    return his_df, pas_df, spi_df, solo_df, fields_df, pfss_df, temp_df, alpha_temp_df, alpha_density_df, mldf, sodf


### ------- PLOTTING SETUP ------- ###
# plotting colors
c = ['#ae017e','#085A99',  '#c98000']
sc = ['#FCA4C4',  '#8FD3F4', '#FFCC70']

# plotting arguments
ssize = 0.5
pkwargs = {'color': sc[0], 's': ssize, 'alpha':0.5}
akwargs = {'color': sc[1], 's': ssize, 'alpha':0.5}
pdkwargs = {'color': c[0]}
adkwargs = {'color': c[1]}
pspkwargs = {'color': sc[0], 's': ssize, 'alpha':0.5}
sokwargs = {'color': sc[1], 's': ssize, 'alpha':0.5}
pspdkwargs = {'color': c[0]}
sodkwargs = {'color': c[1]}

# plot protons
def pplot(ax, var, parker, parkerdown):
    ax.scatter(parker.lon, parker[var], **pkwargs, zorder=-1)
    ax.step(parkerdown.lon, parkerdown[var].values, where='post', **pdkwargs, zorder=1)

# plot alphas
def aplot(ax, var, parker, parkerdown):
    ax.scatter(parker.lon, parker[var], **akwargs, zorder=-1)
    ax.step(parkerdown.lon, parkerdown[var].values, where='post', **adkwargs, zorder=1)

# plot psp particles
def partplot(ax, var, part, partdown, alpha=False):
    if alpha:
        ax.scatter(part.lon, part[var], **akwargs, zorder=-1)
        ax.step(partdown.lon, partdown[var].values, where='post', **adkwargs, zorder=1)
    else:
        ax.scatter(part.lon, part[var], **pkwargs, zorder=-1)
        ax.step(partdown.lon, partdown[var].values, where='post', **pdkwargs, zorder=1)

# plot psp
def pmagplot(ax, var, parker, parkerdownl):
    ax.scatter(parker.lon, parker[var], **pspkwargs, zorder=-1)
    ax.step(parkerdownl.lon, parkerdownl[var].values, where='post', **pspdkwargs, zorder=1)

# plot so
def smagplot(ax, var, smag, smagdown):
    ax.scatter(smag.lon, smag[var], **sokwargs, zorder=-1)
    ax.step(smagdown.lon, smagdown[var].values, where='post', **sodkwargs, zorder=1)

# plot solo pas
def spartplot(ax, var, spart, spartdown):
    ax.scatter(spart.lon, spart[var], **sokwargs, zorder=-1)
    ax.step(spartdown.lon, spartdown[var].values, where='post', **sodkwargs, zorder=1)

# plot solo particles
def abunplot(ax, var, solo, solodown):
    if var == 'iron': 
        ax.scatter(solo.lon, solo[var]/0.0589, color=sc[2], alpha=0.5, s=ssize, zorder=-1)
        ax.step(solodown.lon, solodown[var].values/0.0589, where='post', color=c[2], zorder=1)
    elif var == 'oxy':
        ax.scatter(solo.lon, solo[var], color=sc[1], alpha=0.5, s=ssize, zorder=-1)
        ax.step(solodown.lon, solodown[var].values, where='post', color=c[1], zorder=1) 
    else:
        ax.scatter(solo.lon, solo[var], color=sc[0], alpha=0.5, s=ssize, zorder=-1)
        ax.step(solodown.lon, solodown[var].values, where='post', color=c[0], zorder=1)


### ------ LONGITUDE BINNING ----- ###

def lon_bin(df):
    lon_step = 1
    bin_edges = np.arange(50, 201, step=lon_step)
    bin_labels = np.arange(50, 201, step=lon_step)[:-1]
    df['bins'] = pd.cut(df['lon'], bins=bin_edges, labels=bin_labels)
    avg_columns = []
    for column in df.columns:
        if column != 'bins':
            avg_col = df.groupby('bins')[column].mean()
            avg_columns.append(avg_col)

    # Concatenate the average columns into a single DataFrame
    df_avg = pd.concat(avg_columns, axis=1)
    return df_avg


### ------ TIMESERIES RESAMPLING ------ ###

def time_resample(dfs):
    bin_size = pd.Timedelta(hours=3) # 3 hour cadence
    dds = []
    for df in dfs:
        dd = df.resample(bin_size, closed='left').mean()
        dds.append(dd)
    return dds


### ------ RENORMALIZE DATA ------ ###
def renormalize_data(data, new_min, new_max):
    old_min = np.min(data)
    old_max = np.max(data)
    normalized_data = (data - old_min) / (old_max - old_min)  # Scale to range [0, 1]
    renormalized_data = (normalized_data * (new_max - new_min)) + new_min  # Scale to new range
    return renormalized_data


### ------ GENERATE DATETIME FRAME ------ ###
def gen_dt_arr(dt_init,dt_final,cadence_days=1) :
    """
    Get array of datetime.datetime from {dt_init} to {dt_final} every 
    {cadence_days} days
    """
    dt_list = []
    while dt_init < dt_final :
        dt_list.append(dt_init)
        dt_init += datetime.timedelta(days=cadence_days)
    return np.array(dt_list)


### ------ CALCULATE CROSS HELICITY ------ ###

def smooth(data, num):
    return np.convolve(data, np.ones(num)/num, mode='same')

def calculate_delta(data, smoothed_data):
    return data - smoothed_data

def calculate_velocity_change(delta_b, nppin):
    return delta_b * 1e-9 / np.sqrt(1.25e-6 * nppin * 1e6 * 1.67e-27) / 1000.0

def calculate_sigma_c(zp_square, zm_square):
    return (zp_square - zm_square) / (zp_square + zm_square)

def calculate_sigma_r(zp, zm):
    return 2 * (zp * zm) / (zp + zm)

def calc_sigma(dataframe, mag, num=5):

    # Assuming the data for Vrin, Vtin, Vnin, Brin, Btin, Bnin, and Nppin are already defined.

    Vr_smo = smooth(dataframe.vr, num)
    Vt_smo = smooth(dataframe.vt, num)
    Vn_smo = smooth(dataframe.vn, num)
    delta_Vr = calculate_delta(dataframe.vr, Vr_smo)
    delta_Vt = calculate_delta(dataframe.vt, Vt_smo)
    delta_Vn = calculate_delta(dataframe.vn, Vn_smo)

    Br_smo = smooth(mag.Br, num)
    Bt_smo = smooth(mag.Bt, num)
    Bn_smo = smooth(mag.Bn, num)
    delta_Br = calculate_delta(mag.Br, Br_smo)
    delta_Bt = calculate_delta(mag.Bt, Bt_smo)
    delta_Bn = calculate_delta(mag.Bn, Bn_smo)
        

    delta_Vrb = calculate_velocity_change(delta_Br, dataframe.use_dens)
    delta_Vtb = calculate_velocity_change(delta_Bt, dataframe.use_dens)
    delta_Vnb = calculate_velocity_change(delta_Bn, dataframe.use_dens)

    Zpr = delta_Vr + delta_Vrb
    Zpt = delta_Vt + delta_Vtb
    Zpn = delta_Vn + delta_Vnb
    Zmr = delta_Vr - delta_Vrb
    Zmt = delta_Vt - delta_Vtb
    Zmn = delta_Vn - delta_Vnb

    Zpsquare = Zpr**2 + Zpt**2 + Zpn**2
    Zmsquare = Zmr**2 + Zmt**2 + Zmn**2

    sigmac = calculate_sigma_c(Zpsquare, Zmsquare)
    sigmar = calculate_sigma_r(Zpr, Zmr)

    return sigmac, sigmar