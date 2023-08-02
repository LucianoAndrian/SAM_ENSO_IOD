"""
SAM vs IOD preliminary
"""
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
import xarray as xr
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os
from ENSO_IOD_Funciones import DMI
from ENSO_IOD_Funciones import is_months

import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

w_dir = '/home/luciano.andrian/doc/salidas/'
out_dir = '/home/luciano.andrian/doc/salidas/SAM_IOD/Scatter/'
file_dir = '/datos/luciano.andrian/ncfiles/'
pwd = '/datos/luciano.andrian/ncfiles/'



def SamDetrend(data):
    p40 = data.sel(lat=-40, method='nearest').mean(dim='lon')
    trend = np.polyfit(range(0, len(p40['var'])), p40['var'], deg=1)
    p40 = p40['var'] - trend[0] * range(0, len(p40['var']))

    p65 = data.sel(lat=-65, method='nearest').mean(dim='lon')
    trend = np.polyfit(range(0, len(p65['var'])), p65['var'], deg=1)
    p65 = p65['var'] - trend[0] * range(0, len(p65['var']))

    # index
    sam = (p40 - p40.mean(dim='time')) / p40.std(dim='time') - (p65 - p65.mean(dim='time')) / p65.std(dim='time')

    return sam

def PlotEnso_Iod(dmi, sam, dmi_or, sam_or, title, fig_name = 'fig_enso_iod', out_dir=out_dir, save=False, alpha=0.3):
    from numpy import ma
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(x=dmi_or, y=sam_or, marker='o', s=20, edgecolor='k', color='gray', alpha=alpha)
    im = plt.scatter(x=dmi, y=sam, marker='o', s=40, edgecolor='k', color='orangered')


    plt.ylim((-4, 4));
    plt.xlim((-4, 4))
    plt.axhspan(-0.31, 0.31, alpha=0.2, color='black', zorder=0)
    plt.axvspan(-0.5, 0.5, alpha=0.05, color='black', zorder=0)
    # ax.grid(True)
    fig.set_size_inches(6, 6)
    plt.xlabel('IOD', size=15)
    plt.ylabel('SAM', size=15)

    plt.text(-3.8, 3.4, 'SAM+/IOD-', dict(size=15))
    plt.text(-.4, 3.4, 'SAM+', dict(size=15))
    plt.text(+2, 3.4, 'SAM+/IOD+', dict(size=15))
    plt.text(+2.6, -.1, 'IOD+', dict(size=15))
    plt.text(+2, -3.4, 'SAM-/IOD+', dict(size=15))
    plt.text(-.4, -3.4, 'SAM-', dict(size=15))
    plt.text(-3.8, -3.4, ' SAM-/IOD-', dict(size=15))
    plt.text(-3.8, -.1, 'IOD-', dict(size=15))
    plt.title(title,size=16)
    plt.tight_layout()
    if save:
        plt.savefig(out_dir + 'ENSO_IOD'+ fig_name + '.jpg')
    else:
        plt.show()


end = 2020
i = 1920
# indices: ----------------------------------------------------------------------------------------------------#
dmi = DMI(filter_bwa=False, start_per=str(i), end_per=str(end))[2]
dmi = dmi.sel(time=slice(str(i) + '-01-01', str(end) + '-12-01'))
aux = xr.open_dataset('/datos/luciano.andrian/ncfiles/psl.nc')
sam = SamDetrend(aux)
sam = sam.sel(time=slice(str(i) + '-01-01', str(end) + '-12-01'))

#y_dmi = dmi.where((dmi.Mes==10)).Años.dropna().values


dmi = dmi/dmi.std('time')
sam = sam/sam.std('time')
dmi_or=None
sam_or=None

# SON
aux = dmi.sel(time=is_months(month=dmi['time.month'], mmin=9, mmax=11))
aux2 = sam.sel(time=is_months(month=sam['time.month'], mmin=9, mmax=11))

PlotEnso_Iod(None, None, aux, aux2, 'SON', fig_name='SAM_IOD_SON_Scatter', alpha=0.8, save=True)

aux = aux.groupby('time.year').mean()
aux2 = aux2.groupby('time.year').mean()

PlotEnso_Iod(None, None, aux, aux2, 'SON average', fig_name='SAM_IOD_SON-AV_Scatter', alpha=0.8, save=True)



# JJA
aux = dmi.sel(time=is_months(month=dmi['time.month'], mmin=6, mmax=8))
aux2 = sam.sel(time=is_months(month=sam['time.month'], mmin=6, mmax=8))

PlotEnso_Iod(None, None, aux, aux2, 'JJA', fig_name='SAM_IOD_JJA_Scatter', alpha=0.8, save=True)

aux = aux.groupby('time.year').mean()
aux2 = aux2.groupby('time.year').mean()

PlotEnso_Iod(None, None, aux, aux2, 'JJA average', fig_name='SAM_IOD_JJA-AV_Scatter', alpha=0.8, save=True)


# JASON
aux = dmi.sel(time=is_months(month=dmi['time.month'], mmin=7, mmax=11))
aux2 = sam.sel(time=is_months(month=sam['time.month'], mmin=7, mmax=11))

PlotEnso_Iod(None, None, aux, aux2, 'JASON', fig_name='SAM_IOD_JASON_Scatter', alpha=0.8, save=True)

aux = aux.groupby('time.year').mean()
aux2 = aux2.groupby('time.year').mean()

PlotEnso_Iod(None, None, aux, aux2, 'JASON average', fig_name='SAM_IOD_JASON-AV_Scatter', alpha=0.8, save=True)