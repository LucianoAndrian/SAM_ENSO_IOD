"""
Intento simple de usar pcmci en dmi-enso por punto de grilla
"""

################################################################################
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import xarray as xr
import pandas as pd
pd.options.mode.chained_assignment = None

from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2
from PCMCI import PCMCI

import statsmodels.api as sm
import concurrent.futures
from datetime import datetime
import time

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
#hgt_anom = hgt_anom.sel(time=hgt_anom.time.dt.month.isin([10]))

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
aux = Detrend(aux, 'time')

pp = aux.sel(lat=slice(-15,-30), lon=slice(295,315)).mean(['lon', 'lat'])
pp['var'][-1]=0
################################################################################
# indices
# ---------------------------------------------------------------------------- #
sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam = sam.rolling(time=3, center=True).mean()

dmi = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
aux = aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34 = Nino34CPC(aux, start=1920, end=2020)[0]
# ---------------------------------------------------------------------------- #
dmi2 = SameDateAs(dmi, hgt_anom)
n342 = SameDateAs(n34, hgt_anom)
sam2 = SameDateAs(sam, hgt_anom)
pp = SameDateAs(pp, hgt_anom)

#sam3 = sam2
#c = c/c.std()
dmi3 = dmi2/dmi2.std()
n343 = n342/n342.std()
sam3 = sam2/sam2.std()

amd = hgt_anom.sel(lon=slice(210,290), lat=slice(-90,-50)).mean(['lon', 'lat'])
amd = amd/amd.std()

pp = pp/pp.std()
# Funciones ####################################################################
def recursive_cut(s1, s2):
    i=1
    while(len(s1)!=len(s2)):
        i += 1
        if len(s1)>len(s2):
            s1 = s1[1:]
        else:
            s2 = s2[1:]

        if i > 100:
            print('Error recursive_cut +100 iter.')
            return

    return s1, s2


def min_len(st):
    min_len = float('inf')
    min_st = None

    for s in st:
        cu_len = len(s)
        if cu_len < min_len:
            min_len = cu_len
            min_st = s

    return min_st


def SetLags(x, y, ty, series, parents):

    z_series = []
    for i, p  in enumerate(parents):

        zn = series[p.split('_lag_')[0]]
        tz = np.int(p.split('_lag_')[1])
        len_series = len(zn)

        if i==0:
            ty_aux = max(0, tz - ty)
            tz_aux = max(0, ty - tz)

            x = x[tz+tz_aux:]
            y = y[ty_aux:len_series-ty]
            z_series.append(zn[tz_aux:len_series-tz])
            tz_prev = tz

        if i>0:
            if i>1:
                t_aux = 0
            else:
                t_aux = max(ty, tz_prev)

            z_aux = zn[t_aux:len_series - tz]
            z_series.append(z_aux)

            aux = min_len([x, y, z_aux])

            x,_ = recursive_cut(x, aux)
            y,_ = recursive_cut(y, aux)

            for iz, zs in enumerate(z_series):
                z_series[iz] ,_ = recursive_cut(zs, aux)

    z_columns = [f'parent{i}' for i in range(1, len(z_series) + 1)]
    z_data = {column: z_series[i] for i, column in enumerate(z_columns)}

    return pd.DataFrame({'x':x, 'y':y, **z_data})


# def regre(df):
#
#     x_model = sm.OLS(df['y'], sm.add_constant(df[df.columns[2:]])).fit()
#     x_res = x_model.resid
#
#     return x_model.predict()


def regre(df):
    X = np.column_stack((np.ones_like(df[df.columns[1:2]]), df[df.columns[2:]]))
    y = df['y']

    # Calcular los coeficientes de la regresión lineal
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Calcular los residuos
    x_res = y - np.dot(X, beta)

    #return x_res
    return  np.dot(X, beta)


def regre_res(df):
    X = np.column_stack((np.ones_like(df[df.columns[1:2]]), df[df.columns[2:]]))
    y = df['x']

    # Calcular los coeficientes de la regresión lineal
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Calcular los residuos
    x_res = y - np.dot(X, beta)
    return x_res


def resize_serie(s1, len_min):
    i=1
    while(len(s1)!=len_min):
        i += 1
        s1 = s1[1:]
        if i > 100:
            print('Error recursive_cut +100 iter.')
            return
    return s1


def Compute_PCMCI_MLR(x, target, pcmci_results=False):
    #target='dmi'
    dc2020=False
    lag=0
    # PENDIENTE
    # SETEAR LAAAG!
    x_values = x
    series = {'c': x_values, 'dmi': dmi3.values, 'n34': n343.values,
              'sam':sam3.values}

    result_df = PCMCI(series=series, tau_max=2, pc_alpha=0.2, mci_alpha=0.05)
    if pcmci_results:
        print(result_df)
    actors_parents=[]

    nombres = ['c', 'dmi', 'n34', 'sam']
    actors = [nombres[0], target] + [x for x in nombres if
                                               x != nombres[0] and x != target]

    for actor in actors:
        aux_df = result_df.loc[result_df['Target'] == actor]
        links = list(aux_df['Actor'])

        # Solo autocorrelacion
        own_links = []
        for element in links:
            if actor in element:
                own_links.append(element)

        # own_links=links
        df = SetLags(series[actor], series[actor], ty=lag,
                     series=series, parents=own_links)

        if actor == 'c':
            # regre(df) estima 'c' a partir de sus own_links c_t
            actors_parents.append(regre(df)) #0

        elif actor == target:
            actors_parents.append(regre(df)) #1 target_t
            actors_parents.append(regre_res(df)) #2 residuo de target_t
        else:
            actors_parents.append(df['x'].values) #3, 6 series orignales
            actors_parents.append(regre(df)) # 4, 7 series_t
            actors_parents.append(regre_res(df)) #5, 8 residuos


    # longitud de las series =
    len_min = min(len(serie) for serie in actors_parents)
    for i, a_res in enumerate(actors_parents):
        actors_parents[i] = resize_serie(a_res, len_min)

    # CEN
    if dc2020:
        aux_df = pd.DataFrame({'c_or': resize_serie(series['c'], len_min),
                               target: resize_serie(series[target], len_min),
                               target + '_t': actors_parents[1],
                               'c_t': actors_parents[0],
                               actors[2]: actors_parents[3],
                               actors[3]: actors_parents[6]
                               })
    else:

        aux_df = pd.DataFrame({'c_or': resize_serie(series['c'], len_min),
                               'c_t': actors_parents[0],
                               target: resize_serie(series[target], len_min),
                               target + '_t': actors_parents[1],
                               actors[2]: resize_serie(series[actors[2]], len_min),
                               actors[2] + '_t': actors_parents[4],
                               actors[3]: resize_serie(series[actors[3]], len_min),
                               actors[3] + '_t': actors_parents[7]
                               })

        # aux_df = pd.DataFrame({'c_or': resize_serie(series['c'], len_min),
        #                        # target:actors_parents[2],
        #                        target: resize_serie(series[target], len_min),
        #                        'n34': resize_serie(series['n34'], len_min),
        #                        'c_t': actors_parents[0],
        #                        target + '_t': actors_parents[1],
        #                        # '2actor': actors_parents[3],
        #                        '2actor_t': actors_parents[4],
        #                        })

        # 'c_l': actors_parents[0].values,
        # target +'_l': actors_parents[1],
        # 'actor_WO': actors_parents[3]
        # 'n34_res':actors_parents[2].values})

    model = sm.OLS(aux_df['c_or'], aux_df[aux_df.columns[1:]]).fit()

    # aca se selecciona el beta asociado al target!
    return model.params
    # if model.pvalues[0]<1.05:
    #     return model.params
    # else:
    #     return np.nan


def ComputeByXrUf(c):
    hgt_anom2 = hgt_anom.sel(lat=slice(c[0], c[1]), lon=slice(c[2], c[3]))
    reg_array = xr.apply_ufunc(
        Compute_PCMCI_MLR,
        hgt_anom2['var'],
        input_core_dims=[['time']],
        # dask='parallelized',
        vectorize=True,
        output_dtypes=[float])
    return reg_array


Compute_PCMCI_MLR(amd['var'].values, 'dmi', True)
Compute_PCMCI_MLR(pp['var'].values, 'dmi', True)
