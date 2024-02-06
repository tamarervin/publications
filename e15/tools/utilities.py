"""
Tamar Ervin
Date: September 19, 2022
Utility functions for reading and writing data
"""

import os
import csv
import datetime

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.time import Time

import glob

def read_csv(csv_file_path):
    """
    function to read csv file and return list of dates
    Parameters
    ----------
    csv_file_path : str
        path to csv file
    Returns
    -------
    csv_list : str, list
        list of elements in csv file
    """

    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f)
        csv_list = list(reader)

    return csv_list


def get_dates(date):
    """
    function to convert dates from either JD, string, or datetime
    to a Sunpy usable date form
    Parameters
    ----------
    date
        date in any form (JD, string, datetime)
    Returns
    -------
    date_str : str
        UT datetime as string
    date_obj : datetime
        UT datetime object
    date_jd : float
        UT datetime as float (JD)
    """
    if isinstance(date, str):
        date_str = date
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
        date_jd = Time(date_str)
        date_jd.format = 'jd'
    elif isinstance(date, float):
        t = Time(date, format='jd')
        date_obj = t.datetime
        date_str = date_obj.strftime('%Y-%m-%dT%H:%M:%S.%s')
        date_jd = date
    else:
        date_obj = date
        date_str = date_obj.strftime('%Y-%m-%dT%H:%M:%S.%s')
        date_jd = Time(date_str)
        date_jd.format = 'jd'

    return date_str, date_obj, date_jd


def append_list_as_row(file_name, list_of_elem):
    """
    function to add row to csv file
    Parameters
    ----------
    file_name: path to csv file
    list_of_elem: elements as a list to add to file
    Returns
    -------
    """
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

    return None


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

### ------ LONGITUDE BINNING ----- ###

def lon_bin(df, vv='lon'):
    lon_step = 1
    bin_edges = np.arange(0, 361, step=lon_step)
    bin_labels = np.arange(0, 361, step=lon_step)[:-1]
    df['bins'] = pd.cut(df[vv], bins=bin_edges, labels=bin_labels)
    avg_columns = []
    for column in df.columns:
        if column != 'bins':
            avg_col = df.groupby('bins')[column].mean()
            avg_columns.append(avg_col)

    # Concatenate the average columns into a single DataFrame
    df_avg = pd.concat(avg_columns, axis=1)
    return df_avg


### ------ TIMESERIES RESAMPLING ------ ###

def time_resample(dfs, bin_size= pd.Timedelta(hours=1)): # 1 hour cadence
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


### ------ ROLL TO 180 DEG ------ ###
@u.quantity_input
def rollto180(arr:u.deg) : 
    """
    Cast an array of longitudes in the range [0,360] deg to the range
    [-180,180] deg. Useful when examining stuff that crosses through
    Carrington L0.
    """
    return (((arr + 180*u.deg).to("deg").value % 360) - 180)*u.deg


### ------ READ IN DATA ------ ###
def read_data(RES_DIR, sigma_time=20, pp='parker.csv', pflag=True):
    # BIN SIZES
    bin_size = pd.Timedelta(minutes=30)
    sigma_bin = pd.Timedelta(minutes=sigma_time)

    # READ IN PSP DATA
    file = glob.glob(os.path.join(RES_DIR, pp))
    parker = pd.read_csv(file[0])
    parker['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in parker.Time]
    if pflag:
        sa = np.logical_and(pd.Timestamp('2023-03-16 18:00:00')<=parker.Time, parker.Time<=pd.Timestamp('2023-03-17 00:00:00'))
        parker['flag'][sa] = np.zeros(len(parker['flag'][sa]))
        parker = parker[parker['flag'] == 0].copy()
    use = np.logical_and(pd.Timestamp('2023-03-15 00:00:00')<=parker.Time, parker.Time<=pd.Timestamp('2023-03-20 12:00:00'))
    parker = parker[use].copy()
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

    #### RESAMPLE DATA ####
    parkerdownt = parker.resample(bin_size, closed='left', label='left', loffset=bin_size / 2).mean()
    parkerdownt['Time'] = parkerdownt.index
    parkerdownl = lon_bin(parker, vv='sslon')

    #### RESAMPLE DATA ####
    orbiterdownt = orbiter.resample(bin_size, closed='left', label='left', loffset=bin_size / 2).mean()
    orbiterdownt['Time'] = orbiterdownt.index
    orbiterdownl = lon_bin(orbiter, vv='sslon')

    # READ IN WIND DATA
    file = glob.glob(os.path.realpath(os.path.join(RES_DIR, 'wind.csv')))
    wind = pd.read_csv(file[0])
    wind['Time'] = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S.%f') for d in wind.Time]
    wind = wind.set_index(wind.Time)
    
    wss = wind.resample(sigma_bin, closed='left', label='left', loffset=sigma_bin / 2).mean()
    wss['Time'] = wss.index

    #### RESAMPLE DATA ####
    winddownt = wind.resample(bin_size, closed='left', label='left', loffset=bin_size / 2).mean()
    winddownt['Time'] = winddownt.index
    winddownl = lon_bin(wind, vv='sslon')



    return parker, parkerdownt, parkerdownl, pss, orbiter, orbiterdownl, orbiterdownt, oss, wind, winddownt, winddownl, wss
