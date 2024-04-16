"""
Testeos conceptuales de CN a partir de modelos de regresión
"""
################################################################################
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import xarray as xr
import pandas as pd
pd.options.mode.chained_assignment = None

from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2
#from PCMCI import PCMCI

import statsmodels.api as sm

import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
################################################################################
save=False
use_sam=False
################################################################################
# ---------------------------------------------------------------------------- #
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
hgt_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
#out_dir = ''
# ---------------------------------------------------------------------------- #
hgt = xr.open_dataset(hgt_dir + 'ERA5_HGT200_40-20.nc')
hgt = hgt.rename({'longitude': 'lon', 'latitude': 'lat', 'z': 'var'})
hgt = hgt.interp(lon=np.arange(0,360,2), lat=np.arange(-90, 90, 2))

hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
hgt_anom = hgt.groupby('time.month') - \
           hgt_clim.groupby('time.month').mean('time')

if use_sam:
    weights = np.sqrt(np.abs(np.cos(np.radians(hgt_anom.lat))))
    hgt_anom = hgt_anom * weights

#hgt_anom2 = hgt_anom.sel(lat=slice(-80, 0), lon=slice(60, 70))
hgt_anom = hgt_anom.rolling(time=3, center=True).mean()
hgt_anom = hgt_anom.sel(time=slice('1940-02-01', '2020-11-01'))
hgt_anom = hgt_anom.sel(time=hgt_anom.time.dt.month.isin([8,9,10,11]))
hgt_anom = hgt_anom.sel(time=hgt_anom.time.dt.month.isin([10]))

# ---------------------------------------------------------------------------- #
def OpenObsDataSet(name, sa=True,
                   dir= '/pikachu/datos/luciano.andrian/observado/ncfiles/'
                        'data_no_detrend/'):

    aux = xr.open_dataset(dir + name + '.nc')
    if sa:
        aux2 = aux.sel(lon=slice(270, 330), lat=slice(15, -60))
        if len(aux2.lat) > 0:
            return aux2
        else:
            aux2 = aux.sel(lon=slice(270, 330), lat=slice(-60, 15))
            return aux2
    else:
        return aux

def Detrend(xrda, dim):
    aux = xrda.polyfit(dim=dim, deg=1)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients)
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients)
    dt = xrda - trend
    return dt

def Weights(data):
    weights = np.transpose(np.tile(np.cos(data.lat * np.pi / 180),
                                   (len(data.lon), 1)))
    data_w = data * weights
    return data_w


data = OpenObsDataSet(name='pp_pgcc_v2020_1891-2023_1', sa=True)
data = data.rename({'precip':'var'})
data_40_20 = data.sel(time=slice('1940-01-16', '2020-12-16'))
del data

data_40_20 = Weights(data_40_20)
data_40_20 = data_40_20.sel(lat=slice(20, -80)) # HS
data_40_20 = data_40_20.rolling(time=3, center=True).mean()
aux = data_40_20.sel(time=data_40_20.time.dt.month.isin([8,9,10,11]))
pp = Detrend(aux, 'time')

pp_serie = pp.sel(lat=slice(-15,-30), lon=slice(295,315)).mean(['lon', 'lat'])
pp_serie['var'][-1]=0
################################################################################
# indices
# ---------------------------------------------------------------------------- #
sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam = sam.rolling(time=3, center=True).mean()

dmi = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/"
                      "sst.mnmean.nc")
aux = aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34 = Nino34CPC(aux, start=1920, end=2020)[0]
# ---------------------------------------------------------------------------- #
dmi2 = SameDateAs(dmi, hgt_anom)
n342 = SameDateAs(n34, hgt_anom)
sam2 = SameDateAs(sam, hgt_anom)
pp = SameDateAs(pp, hgt_anom)
pp_serie = SameDateAs(pp_serie, hgt_anom)

#sam3 = sam2
#c = c/c.std()
dmi3 = dmi2/dmi2.std()
n343 = n342/n342.std()
sam3 = sam2/sam2.std()

amd = hgt_anom.sel(lon=slice(210,290), lat=slice(-90,-50)).mean(['lon', 'lat'])
amd = amd/amd.std()
hgt_anom = hgt_anom/hgt_anom.std()

pp_serie = pp_serie/pp_serie.std()
pp = pp/pp.std()
# Funciones ####################################################################
def regre(series, intercept, coef=0):
    df = pd.DataFrame(series)
    if intercept:
        X = np.column_stack((np.ones_like(df[df.columns[1]]),
                             df[df.columns[1:]]))
    else:
        X = df[df.columns[1:]].values
    y = df[df.columns[0]]

    coefs = np.linalg.lstsq(X, y, rcond=None)[0]

    coefs_results = {}
    for ec, e in enumerate(series.keys()):
        if intercept and ec == 0:
            e = 'constant'
        if e != df.columns[0]:
            if intercept:
                coefs_results[e] = coefs[ec]
            else:
                coefs_results[e] = coefs[ec-1]

    if isinstance(coef, str):
        return coefs_results[coef]
    else:
        return coefs_results
# ---------------------------------------------------------------------------- #
# idea, a partir de un modelo de causalidad:
# modelo a dmi --> sam ------------------------------------------------------- #
series_full = {'c': pp_serie['var'].values, 'dmi': dmi3.values,
          'n34': n343.values, 'sam': sam3.values}
series_dmi_n34 = {'c': pp_serie['var'].values, 'dmi': dmi3.values,
          'n34': n343.values}
series_n34 = {'c': pp_serie['var'].values, 'n34': n343.values}
series_n34_sam = {'c': pp_serie['var'].values, 'n34': n343.values,
                  'sam': sam3.values}

regre(series_dmi_n34, True, 'dmi') # Efecto total DMI sobre pp sesa
regre(series_full, True, 'dmi') # Efecto total DMI sobre pp sesa

regre(series_n34, True, 'n34') # Efecto total N34 sobre pp sesa
regre(series_full, True, 'n34') # Efecto total N34 sobre pp sesa

regre(series_n34_sam, True, 'sam') # Efecto total SAM sobre pp sesa
regre(series_full, True, 'sam') # Efecto total SAM sobre pp sesa
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
