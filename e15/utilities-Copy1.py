'''
Utilities for plotting and generating paper stats
Tamar Ervin
June 27, 2023
'''

import os, glob
import pandas as pd
import datetime
# from pyspedas import time_string
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

# def lon_bin(df):
#     lon_step = 1
#     bin_edges = np.arange(50, 201, step=lon_step)
#     bin_labels = np.arange(50, 201, step=lon_step)[:-1]
#     df['bins'] = pd.cut(df['lon'], bins=bin_edges, labels=bin_labels)
#     avg_columns = []
#     for column in df.columns:
#         if column != 'bins':
#             avg_col = df.groupby('bins')[column].mean()
#             avg_columns.append(avg_col)

#     # Concatenate the average columns into a single DataFrame
#     df_avg = pd.concat(avg_columns, axis=1)
#     return df_avg

def lon_bin(df, counts=1, vv='sslon'):
    lon_step = 1
    bin_edges = np.arange(50, 201, step=lon_step)
    bin_labels = np.arange(50, 201, step=lon_step)[:-1]
    df['bins'] = pd.cut(df[vv], bins=bin_edges, labels=bin_labels)
    
    avg_columns = []
    for column in df.columns:
        if column != 'bins':
            # Calculate the count of points in each bin
            bin_counts = df['bins'].value_counts()
            
            # Filter bins with at least 5 points
            valid_bins = bin_counts[bin_counts >= counts].index
            
            # Calculate the average only for valid bins
            avg_col = df[df['bins'].isin(valid_bins)].groupby('bins')[column].mean()
            avg_columns.append(avg_col)
    
    # Concatenate the average columns into a single DataFrame
    df_avg = pd.concat(avg_columns, axis=1)
    return df_avg



### ------ TIMESERIES RESAMPLING ------ ###

def time_resample(dfs, bin_size):
    # bin_size = pd.Timedelta(hours=3) # 3 hour cadence
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


### ------ READ IN DATA ------ ###
def read_data(RES_DIR, sigma_time=20, pp='parker20_gamma.csv', pflag=True):
    # BIN SIZES
    bin_size = pd.Timedelta(minutes=30)
    sigma_bin = pd.Timedelta(minutes=sigma_time)

    # READ IN PSP DATA
    file = glob.glob(os.path.join(RES_DIR, pp))
    parker = pd.read_csv(file[0])
    parker['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in parker.Time]
    if pflag:
        time_mask = np.logical_and(pd.Timestamp('2023-03-16 18:00:00') <= parker['Time'],
                           parker['Time'] <= pd.Timestamp('2023-03-17 00:00:00'))
        flag_mask = parker['flag'] == 0
        parker = parker[np.logical_or(flag_mask, time_mask)].copy()
        # flagNe = np.logical_and(pd.Timestamp('2023-03-16 18:00:00')<=parker.Time, parker.Time<=pd.Timestamp('2023-03-17 00:00:00'))
        # parker = parker[parker['flag'] == 0].copy()
    parker = parker.set_index(parker.Time)
    
    pss = parker.resample(sigma_bin, closed='left', label='left', loffset=sigma_bin / 2).mean()
    pss['Time'] = pss.index

    # READ IN ORBITER DATA
    file = glob.glob(os.path.realpath(os.path.join(RES_DIR, 'orbiter.csv')))
    orbiter = pd.read_csv(file[0])
    orbiter['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in orbiter.Time]
    orbiter = orbiter.set_index(orbiter.Time)

    oss = orbiter.resample(sigma_bin, closed='left', label='left', loffset=sigma_bin / 2).mean()
    oss['Time'] = oss.index

    # # READ IN WIND DATA
    # file = glob.glob(os.path.realpath(os.path.join(RES_DIR, 'orbiter_his.csv')))
    # his_orbiter = pd.read_csv(file[0])
    # his_orbiter['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in his_orbiter.Time]
    # his_orbiter = his_orbiter.set_index(his_orbiter.Time)
    file = glob.glob(os.path.realpath(os.path.join(RES_DIR, 'wind.csv')))
    wind = pd.read_csv(file[0])
    wind['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in wind.Time]
    wind = wind.set_index(wind.Time)
    
    wss = wind.resample(sigma_bin, closed='left', label='left', loffset=sigma_bin / 2).mean()
    wss['Time'] = wss.index


    #### RESAMPLE DATA ####
    parkerdownt = parker.resample(bin_size, closed='left', label='left', loffset=bin_size / 2).mean()
    parkerdownt['Time'] = parkerdownt.index
    parkerdownl = lon_bin(parker, vv='sslon')

    #### RESAMPLE DATA ####
    orbiterdownt = orbiter.resample(bin_size, closed='left', label='left', loffset=bin_size / 2).mean()
    orbiterdownt['Time'] = orbiterdownt.index
    orbiterdownl = lon_bin(orbiter, vv='sslon')

    #### RESAMPLE DATA ####
    winddownt = wind.resample(bin_size, closed='left', label='left', loffset=bin_size / 2).mean()
    winddownt['Time'] = winddownt.index
    winddownl = lon_bin(wind, vv='sslon')

    # return parker, parkerdownt, parkerdownl, pss, orbiter, orbiterdownl, orbiterdownt, oss, his_orbiter, his_orbiterdownt, his_orbiterdownl
    return parker, parkerdownt, parkerdownl, pss, orbiter, orbiterdownl, orbiterdownt, oss, wind, winddownt, winddownl, wss