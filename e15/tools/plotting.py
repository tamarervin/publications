# Plotting Functions

# Tamar Ervin
# May 18, 2023


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import scipy.stats as stats
import sunpy
import matplotlib as mpl

def correlation_plot(data_list, labels):
    data = np.zeros((len(data_list[0]), len(data_list)))
    for i, d in enumerate(data_list):
        data[:, i] = d

    hist2d_kwargs={'quiet': True, 'plot_datapoints': True, 'plot_density': False, 'plot_contours': False, 'no_fill_contours': True}
    corn = corner.corner(data, labels=labels, labelpad=0.1, **hist2d_kwargs)


    # remove the 1D histograms
    axes = np.array(corn.axes).reshape((len(data_list), len(data_list)))
    for a in axes[np.triu_indices(len(data_list))]:
        a.remove()

    for a in axes[np.tril_indices(len(data_list))]:
        line = a.lines
        if line:
            # plot stuff
            a.scatter(line[0].get_xdata(), line[0].get_ydata(), color='lavender', edgecolor='black', linewidth=1.0)
            # a.scatter(line[0].get_xdata()[spot_days], line[0].get_ydata()[spot_days], color='lightcoral', edgecolor='black', linewidth=1.0)

            # set text box with correlation
            corr = stats.spearmanr(line[0].get_xdata(), line[0].get_ydata())
            # spot_corr = stats.spearmanr(line[0].get_xdata()[spot_days], line[0].get_ydata()[spot_days])
            # no_spot = stats.spearmanr(line[0].get_xdata()[~spot_days], line[0].get_ydata()[~spot_days])
            props = dict(boxstyle='round', facecolor='lavender', alpha=0.5)
            #  + '\n' + str(np.round(spot_corr[0], 3)) + '\n' + str(np.round(no_spot[0], 3)
            a.text(0.925, 0.925, str(np.round(corr[0], 3)),
                transform=a.transAxes, fontsize=14, color='k', alpha=1.0,
            ha='right', va='top', fontweight='bold', bbox=props)

            # set axes grid tick marks
            r = np.max(line[0].get_xdata()) - np.min(line[0].get_xdata())
            a.set(xlim=(np.min(line[0].get_xdata()) - r/6, np.max(line[0].get_xdata()) + r/6), xticks=np.arange(np.min(line[0].get_xdata()), np.max(line[0].get_xdata()) + r/6, step=r/2))
            r = np.max(line[0].get_ydata()) - np.min(line[0].get_ydata())
            a.set(ylim=(np.min(line[0].get_ydata()) - r/6, np.max(line[0].get_ydata()) + r/6), yticks=np.arange(np.min(line[0].get_ydata()), np.max(line[0].get_ydata()) + r/6, step=r/2))



### PFSS PLOT####
def plot_pfss(smap, hcs, source_surface, flines, datetimes, ax, BrR2, nf=8, dim=[0, 360, -90, 90], tm=7, dd=True, yl=True, full=True):

    # plot on axes
    plt.sca(ax)
    
    # color dictionary
    color_dict = {-1:"blue", 0:"black", 1:"red"}
    
    # plot euv map
    if type(smap) == sunpy.map.mapbase.GenericMap:
        lons = np.linspace(0, 360, 721)
        lats = np.linspace(-90, 90, 361)
        lognorm = mpl.colors.LogNorm(vmin=np.nanpercentile(smap.data.flatten(),10
                                                           ), 
                                vmax=np.nanpercentile(smap.data.flatten(),99.9))
        ax.pcolormesh(lons, lats, smap.data, cmap='sdoaia193', norm=lognorm, zorder=-1)
    else:
        lons = np.linspace(0, 360, 361)
        lats = np.linspace(-90, 90, 181)
        ax.pcolormesh(lons,lats,smap,cmap="coolwarm", zorder=-1)
        
    # plot HCS
    ax.plot(hcs.lon, hcs.lat, color='white', label='HCS', zorder=0)

    if full:
        # plot field lines
        for f in flines[::nf] :
            fcoords = f.coords
            fcoords.representation_type="spherical"
            ax.plot(fcoords.lon,
                    fcoords.lat,
                    # fcoords.z.to("R_sun"),
                    color = color_dict.get(f.polarity), 
                    linewidth=0.5, alpha=0.5, zorder=1
                )

        # plot trajectory
        polarity = np.sign(BrR2)
        pos = np.where(polarity == 1)
        neg = np.where(polarity == -1)
        ax.scatter(source_surface.lon[neg], source_surface.lat[neg], color='navy', label='Negative Polarity', zorder=2, s=3)
        ax.scatter(source_surface.lon[pos], source_surface.lat[pos], color='darkred',label='Positive Polarity', zorder=3, s=3)

    # plot dates
    if dd:
        dates = [i.date() for i in datetimes]
        dates_str = [d.strftime('%m-%d-%Y') for d in dates]
        psp_inds = [np.where(np.array(dates_str) == d)[0][0] for d in np.unique(dates_str)[1:]]
        labels = np.unique(dates_str)[1:]
        for i, x in enumerate(list(zip(source_surface.lon[psp_inds], source_surface.lat[psp_inds]))):
            label = labels[i]
            ax.text(x[0].value - 2, x[1].value + 2, label[:-5], ha="center", va="bottom", color='white', rotation=45, size='large', zorder=5)

    # title and labels
    ax.set_xlim((dim[0], dim[1]))
    ax.set_ylim((dim[2], dim[3]))
    ax.set_xticks(np.linspace(dim[0], dim[1], tm))
    ax.set_yticks(np.linspace(dim[2], dim[3], tm))
    ax.set_xlabel(r"$\rm Carrington \; Longitude \; [deg]$")
    if yl:
        ax.set_ylabel(r"$\rm Carrington \; Latitude \; [deg]$")

    return ax
        