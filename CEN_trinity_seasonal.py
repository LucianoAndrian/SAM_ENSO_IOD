"""
CEN
ENSO-IOD-U 50hPa
"""
# ---------------------------------------------------------------------------- #
save = False
out_dir = ''

# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import xarray as xr
import pandas as pd
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

from funciones.indices import Nino34CPC, DMI2
from cen.cen_funciones import Detrend, Weights
from funciones.utils import SameDateAs
# ---------------------------------------------------------------------------- #
# set data

sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
hgt_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
dir_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_no_detrend/'

v = 'ERA5_HGT200_40-20.nc'
dir_file = hgt_dir + v

def change_name_dim(data, dim_to_check, dim_to_rename):
    if dim_to_check in list(data.dims):
        return data.rename({dim_to_check: dim_to_rename})

def change_name_variable(data, new_name_var):
    return data.rename({list(data.data_vars)[0]:new_name_var})

def set_data_to_cen(dir_file, interp_2x2=True,
                    select_lat=None, select_lon=None,
                    rolling=False, rl_win=3,
                    purge_extra_dims=False):
    data = xr.open_dataset(dir_file)
    data = change_name_dim(data, 'latitude', 'lat')
    data = change_name_dim(data, 'longitude', 'lon')
    data = change_name_variable(data, 'var')

    if interp_2x2:
        try:
            data = data.interp(lon=np.arange(data.lon[0], data.lon[-1],2),
                               lat=np.arange(data.lat[0], data.lat[-1],2))
        except:
            data = data.interp(lon=np.arange(data.lon[0], data.lon[-1],2),
                               lat=np.arange(data.lat[0], data.lat[-1],-2))

    if select_lon is not None:
        data = data.sel(lon=slice(select_lon[0], select_lon[-1]))

    if select_lat is not None:
        if len(select_lat)>1:
            data = data.sel(lat=slice(select_lat[0], select_lat[-1]))
        else:
            data = data.sel(lat=select_lat[0])

    data_clim = data.sel(time=slice('1979-01-01', '2000-12-01'))

    data_anom_mon = data.groupby('time.month') - \
                    data_clim.groupby('time.month').mean('time')

    if rolling:
        data_anom_mon = data_anom_mon.rolling(time=rl_win, center=True).mean()

    # pesos de esta forma para normalizar para el analisis
    #weights = np.sqrt(np.abs(np.cos(np.radians(data_anom_mon.lat))))
    # data_anom_mon = data_anom_mon * weights

    data_anom_mon = Weights(data_anom_mon)

    if purge_extra_dims:
        extra_dim = [d for d in data_anom_mon['var'].dims
                     if d not in ['lon', 'lat', 'time']]
        for dim in extra_dim:
            first_val = data_anom_mon[dim].values[0]
            data_anom_mon = data_anom_mon.sel({dim: first_val}).drop(dim)
            print(f'drop_dim: {dim}')

    return data_anom_mon

hgt_anom_mon = set_data_to_cen(dir_file, interp_2x2=True,
                               rolling=True, rl_win=3)

# ---------------------------------------------------------------------------- #
# index
dmi_or = DMI2(filter_bwa=False, start_per='1959', end_per='2020',
              sst_anom_sd=False, opposite_signs_criteria=False)[2]

sst_aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/"
                      "sst.mnmean.nc")
sst_aux = sst_aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34_or = Nino34CPC(sst_aux, start=1920, end=2020)[0]

u50_dir_file = '/pikachu/datos/luciano.andrian/observado/' \
          'ncfiles/ERA5/downloaded/ERA5_U50hpa_40-20.mon.nc'

u50_or = set_data_to_cen(u50_dir_file, interp_2x2=False, select_lat=[50],
                         rolling=True, rl_win=3, purge_extra_dims=True)

u50_or = Detrend(u50_or, 'time')
u50_or = u50_or.mean('lon')
u50_or = xr.DataArray(u50_or['var'].drop('lat'))
# ---------------------------------------------------------------------------- #
lags = {'SON': [10, 10, 10, 10],
        'ASO': [9, 9, 9, 9],
        'ASO--SON': [10, 10, 9, 9],
        'JAS_ASO--SON': [10, 10, 8, 9],
        'JAS--SON': [10, 10, 8, 8]}

# CEN ------------------------------------------------------------------------ #
hgt_anom_mon = hgt_anom_mon.sel(
    time=hgt_anom_mon.time.dt.year.isin(range(1959, 2021)))


def identify_lags(lags):
    lag_target = lags[0]
    indices_lags = lags[1:]
    return lag_target, indices_lags

def Setlag(indice, lag, variable_target, years_to_remove):
    indice = indice.sel(time=indice.time.dt.month.isin(lag))
    indice = indice.sel(
        time=indice.time.dt.year.isin(variable_target.time.dt.year))
    indice = indice.sel(time=~indice.time.dt.year.isin(years_to_remove))
    indice = indice / indice.std()

    return indice

def SetLag_to_ActorList(variable_target, month_target, indices, lags,
                     years_to_remove, verbose=1):

    # Seteo de variable target
    variable_target = variable_target.sel(
        time=variable_target.time.dt.month.isin([month_target]))
    if verbose > 1: print('month_target seteado')
    variable_target = variable_target / variable_target.std()
    if verbose > 1: print('normalizado')
    variable_target = variable_target.sel(
        time=~variable_target.time.dt.month.isin(years_to_remove))
    if verbose > 1: print('years_to_remove ok')

    actor_list = {}
    for (indice_name, indice), lag in zip(indices.items(), lags):
        if verbose > 0: print(f'indice: {indice_name} - lag: {lag}')
        actor_list[indice_name] = Setlag(indice, lag, variable_target,
                                         years_to_remove)
    return variable_target, actor_list


indices = {'n34':n34_or, 'dmi':dmi_or, 'u50':u50_or}
for l_count, lag_key in enumerate(lags.keys()):
    lag_target, indices_lags = identify_lags(lags[lag_key])

    print(f'{lag_key}')

    hgt_target, actor_list = SetLag_to_ActorList(
        variable_target=hgt_anom_mon,
        month_target=lag_target,
        indices=indices,
        lags=indices_lags,
        years_to_remove=[2002, 2019])

# Seguir compute CEN --------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #