"""
Comparacion con Marshall
uno es es sfc y el otro en 1000hpa... ok! es sólo para ver el comportamiento
"""
################################################################################
dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/sam_comparison/'
################################################################################
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import sys
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
from ENSO_IOD_Funciones import SameDateAs
################################################################################
save = True
################################################################################
def PlotScatterSam(x, y, nom_x, nom_y, corr, nombre_fig, save):
    fig, ax = plt.subplots()
    im = plt.scatter(x=x, y=y, color='gray', edgecolors='black')
    plt.title(nom_x + ' vs ' + nom_y + '   r = ' + str(np.round(corr, 3)))
    plt.xlabel(nom_x)
    plt.ylabel(nom_y)
    plt.ylim((-6, 6))
    plt.xlim((-6, 6))
    plt.grid(True)
    fig.set_size_inches(6,6)
    if save:
        plt.savefig(out_dir + nombre_fig + '.jpg')
        plt.close()
    else:
        plt.show()
################################################################################
# Marshall #####################################################################
try:
    sam_marshall = pd.read_fwf(dir + 'sam_marshall.txt', skiprows = 1,
                               header=None)
except:
    print('-----------------------')
    print('el directorio no existe')
    sys.exit(1)

sam_marshall = sam_marshall.loc[sam_marshall[0]<=2020]

year = sam_marshall.iloc[:, 0]
dates = xr.cftime_range(start=str(sam_marshall[0][0]) + '-01-01',
                        end='2020-12-01', freq='MS')
values = sam_marshall.iloc[:, 1:].values.flatten()

sam_marshall = xr.Dataset(
    {'sam': (['time'], values)},
    coords={'time': dates}
)
# sam_marshall = sam_marshall.rolling(time=3, center=True).mean()
# sam_marshall = sam_marshall.sel(time=sam_marshall.time.dt.month.isin(10))

# NOAA #########################################################################
try:
    sam_noaa = pd.read_fwf(dir + 'sam_noaa.txt', skiprows = 1, header=None)
except:
    print('-----------------------')
    print('el directorio no existe')
    sys.exit(1)

sam_noaa = sam_noaa.loc[sam_noaa[0]<=2020]

year = sam_noaa.iloc[:, 0]
dates = xr.cftime_range(start=str(sam_noaa[0][0]) + '-01-01',
                        end='2020-12-01', freq='MS')
values = sam_noaa.iloc[:, 1:].values.flatten()

sam_noaa = xr.Dataset(
    {'sam': (['time'], values)},
    coords={'time': dates}
)

# sam_noaa = sam_noaa.rolling(time=3, center=True).mean()
# sam_noaa = sam_noaa.sel(time=sam_noaa.time.dt.month.isin(10))
# EC ###########################################################################
try:
    sam_1000 = xr.open_dataset(dir + 'sam_1000.nc')
    sam_700 = xr.open_dataset(dir + 'sam_700.nc')
except:
    print('-----------------------')
    print('el directorio no existe')
    sys.exit(1)
# sam_1000 = SameDateAs(sam_1000, sam_marshall)
# sam_1000 = sam_1000.rolling(time=3, center=True).mean()
# sam_1000 = sam_1000.sel(time=sam_1000.time.dt.month.isin(10))

################################################################################
sam_marshall = SameDateAs(sam_marshall, sam_noaa)
sam_1000 = SameDateAs(sam_1000, sam_noaa)
sam_700 = SameDateAs(sam_700, sam_noaa)

corr_mar_noaa = pearsonr(sam_marshall.sam.values,
                         sam_noaa.sam.values)[0]

corr_mar_ec_1000 = pearsonr(sam_marshall.sam.values,
                            sam_1000.mean_estimate.values)[0]

corr_mar_ec_700 = pearsonr(sam_marshall.sam.values,
                           sam_700.mean_estimate.values)[0]

corr_ec_ec = pearsonr(sam_700.mean_estimate.values,
                      sam_1000.mean_estimate.values)[0]

corr_noaa_ec_700 = pearsonr(sam_noaa.sam.values,
                            sam_700.mean_estimate.values)[0]
#------------------------------------------------------------------------------#
PlotScatterSam(x=sam_marshall.sam.values/sam_marshall.std('time').sam.values,
               y=sam_noaa.sam.values/sam_noaa.std('time').sam.values,
               nom_x='Marshall (slp)', nom_y='NOAA 700hPa'
               , corr=corr_mar_noaa, nombre_fig='SAM_full_Marshall_NOAA',
               save=save)

PlotScatterSam(x=sam_marshall.sam.values/sam_marshall.std('time').sam.values,
               y=sam_1000.mean_estimate.values/
                 sam_1000.std('time').mean_estimate.values,
               nom_x='Marshall (slp)', nom_y='EC 1000hPa'
               , corr=corr_mar_ec_1000, nombre_fig='SAM_full_Marshall_EC1000',
               save=save)

PlotScatterSam(x=sam_marshall.sam.values/sam_marshall.std('time').sam.values,
               y=sam_700.mean_estimate.values/
                 sam_700.std('time').mean_estimate.values,
               nom_x='Marshall (slp)', nom_y='EC 700hPa'
               , corr=corr_mar_ec_700, nombre_fig='SAM_full_Marshall_EC700', 
               save=save)

PlotScatterSam(x=sam_700.mean_estimate.values/
                 sam_700.std('time').mean_estimate.values,
               y=sam_1000.mean_estimate.values/
                 sam_1000.std('time').mean_estimate.values,
               nom_x='EC 700hPa', nom_y='EC 1000hPa'
               , corr=corr_ec_ec, nombre_fig='SAM_full_EC1000_EC700', 
               save=save)

PlotScatterSam(x=sam_noaa.sam.values/sam_noaa.std('time').sam.values,
               y=sam_700.mean_estimate.values/
                 sam_700.std('time').mean_estimate.values,
               nom_x='NOAA 700hPa', nom_y='EC 700hPa'
               , corr=corr_noaa_ec_700, nombre_fig='SAM_full_NOAA_EC700', 
               save=save)
################################################################################
print('done')
################################################################################
