"""
Tamar Ervin
Date: September 19, 2022
Functions for looking at PSP data and building coordinates
"""

from telnetlib import RSP
import sunpy
import astropy
import numpy as np
import astrospice
import datetime
import astropy.units as u

from pyspedas import time_string
from astropy.coordinates import SkyCoord


def get_psp_coords(timestamps):
    """
    Function to get the coordinates of PSP at a list of timestamps

    Args:
        timestamps (list): list of timestamps for which we have PSP data

    Returns:
        sunpy coordinates: coordinates of PSP's trajectory at given timestamps
    """
    # Load in PSP Spice Kernels (download happens automatically)
    kernels = astrospice.registry.get_kernels('psp','predict') 

    # Create the coordinates. We need the spice string "SOLAR PROBE PLUS"
    # This produces 
    psp_coords_inertial = astrospice.generate_coords('SOLAR PROBE PLUS',timestamps)

    # Transform to Heliographic Carrington, i.e. the frame that co-rotates with the Sun.
    psp_coords_carr = psp_coords_inertial.transform_to(
        sunpy.coordinates.HeliographicCarrington(observer="self")
    )

    return psp_coords_carr

def parker_streamline(phi_0=0.0,r0=1.0,sampling=100, w_s=(2.*np.pi)/(24.47*(24*3600)), v_sw=360e3,rmax=215):
    """
    function to generate the parker spiral 

    Parameters
    ----------
    phi_0 : float, optional
        starting angular value, by default 0.0
    r0 : float, optional
        starting radial value, by default 1.0
    sampling : int, optional
        sampling rate for radius, by default 100
    w_s : tuple, optional
        rotation of the Sun, by default (2.*np.pi)/(24.47*(24*3600))
    v_sw : float, optional
        solar wind velocity, by default 360e3
    rmax : int, optional
        maximum solar radius to project out to, by default 215

    Returns
    -------
    list 
        list of radial values
    degrees
        phi value
    """
    
    # Output r :: Rs, phi :: deg
    r = np.linspace(r0,rmax,sampling)*u.R_sun
    r0 = r0*u.R_sun
    phi = np.degrees(-w_s/v_sw*(r.to("m").value-r0.to("m").value)) + phi_0

    return r.value, phi


def delta_long(r, lat, vsw=360.*u.km/u.s, a=[14.713*u.deg/u.d, -2.396*u.deg/u.d, -1.787*u.deg/u.d], rss=2.5):
    # omega_sun = a[0] + a[1] * (np.sin(lat))**2 + a[2] * (np.sin(lat))**4
    omega_sun = a[0]
    return omega_sun * (r - rss * u.R_sun) / vsw


def coord_projection(coords_carr, rs, timestamps):
    rss = rs*u.R_sun
    coords_carr.representation_type = "spherical"
    source_surface = astropy.coordinates.SkyCoord(
        radius = rss * np.ones(len(coords_carr)),
        ## The projection can also take a varying solar wind speed as an input
        lat = coords_carr.lat,
        lon = coords_carr.lon + delta_long(coords_carr.radius, coords_carr.lat, rss=rss.value),
        frame = sunpy.coordinates.HeliographicCarrington(observer="self", obstime=timestamps)
    )

    return source_surface


def time_to_lon(timestamps, so=False, rss=2.5):

    # convert times to timestamps
    use_times = time_string(timestamps)

    if so:
        # get coordinates
        kernels = astrospice.registry.get_kernels('solar orbiter', 'predict')
        coords_inertial = astrospice.generate_coords('SOLAR ORBITER', use_times)
    
    else:
        # get coordinates
        kernels = astrospice.registry.get_kernels('psp', 'predict')
        coords_inertial = astrospice.generate_coords('SOLAR PROBE PLUS', use_times)

    # Transform to Heliographic Carrington, i.e. the frame that co-rotates with the Sun.
    coords_carr = coords_inertial.transform_to(
    sunpy.coordinates.HeliographicCarrington(observer="self"))

    # get coordinates at source surface
    source_surface = coord_projection(coords_carr, rss, use_times)

    return source_surface, coords_carr


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

@u.quantity_input
def ballistic_delta_long(r:u.R_sun,
               r_inner=2.0*u.R_sun,
               vsw=360.*u.km/u.s,
               omega_sun=14.713*u.deg/u.d,
               ):
    """ 
    Ballistic longitudinal shift of a Parker spiral connecting two
    points at radius r and r_inner, for a solar wind speed vsw. Solar
    rotation rate is also tunable
    """
    return (omega_sun * (r - r_inner) / vsw)

def ballistically_project(skycoord,r_inner = 2.0*u.R_sun, vr_arr=None) :
    """
    Given a `SkyCoord` of a spacecraft trajectory in the Carrington frame,
    with `representation_type="spherical"`, and optionally an array of
    measured solar wind speeds at the same time intervals of the trajectory,
    return a SkyCoord for the trajectory ballistically projected down to 
    `r_inner` via a Parker spiral of the appropriate curvature. When `vr_arr`
    is not supplied, assumes wind speed is everywhere 360 km/s
    """
    if vr_arr is None : vr_arr = np.ones(len(skycoord))*360*u.km/u.s
    lons_shifted = skycoord.lon + ballistic_delta_long(skycoord.radius,
                                             r_inner=r_inner,
                                             vsw=vr_arr
                                            )
    return SkyCoord(
        lon = lons_shifted, 
        lat = skycoord.lat,
        radius = r_inner * np.ones(len(skycoord)),
        representation_type="spherical",
        frame = skycoord.frame
    )