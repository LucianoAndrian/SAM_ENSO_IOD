"""
CEN
ENSO-IOD-U 50hPa
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas_cen/'
plots = False

# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import xarray as xr
import pandas as pd
pd.options.mode.chained_assignment = None
from funciones.indices import Nino34CPC, DMI2
from cen.cen_funciones import Detrend, Weights, set_actor_effect_dict, \
    apply_CEN_effect, identify_lags, SetLag_to_ActorList
from funciones.utils import change_name_dim, change_name_variable

# aux funciones -------------------------------------------------------------- #
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

def df_linea_lag(lag_key):
    data = {
        'v_efecto': f'Lag {lag_key}',
        'b': '',
        'alpha_0.15': '',
        'alpha_0.1': '',
        'alpha_0.05': ''
    }
    return pd.DataFrame([data])

def concat_df(df1=None, df2=None):
    if df1 is None:
        return df2
    else:
        return pd.concat([df1, df2], ignore_index=True)

# ---------------------------------------------------------------------------- #
# set data
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
hgt_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
dir_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_no_detrend/'

hgt_anom_mon = set_data_to_cen(dir_file = f'{hgt_dir}ERA5_HGT200_40-20.nc',
                               interp_2x2=True, rolling=True, rl_win=3)

# indices -------------------------------------------------------------------- #
dmi_or = DMI2(filter_bwa=False, start_per='1959', end_per='2020',
              sst_anom_sd=False, opposite_signs_criteria=False)[2]

sst_aux = xr.open_dataset(
    '/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc')
sst_aux = sst_aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34_or = Nino34CPC(sst_aux, start=1920, end=2020)[0]

u50_dir_file = '/pikachu/datos/luciano.andrian/observado/' \
          'ncfiles/ERA5/downloaded/ERA5_U50hpa_40-20.mon.nc'
u50_or = set_data_to_cen(u50_dir_file, interp_2x2=False, select_lat=[-60],
                         rolling=True, rl_win=3, purge_extra_dims=True)
u50_or = Detrend(u50_or, 'time')
u50_or = u50_or.mean('lon')
u50_or = xr.DataArray(u50_or['var'].drop('lat'))

# CEN ------------------------------------------------------------------------ #
indices = {'n34':n34_or, 'dmi':dmi_or, 'u50':u50_or}

# lags, lag_variable + lags orden como indices
lags = {'SON': [10, 10, 10, 10],
        #'ASO--SON': [10, 9, 9, 9],
        'ASO': [9, 9, 9, 9],
        'JAS_ASO--SON': [10, 8, 8, 9],
        'JAS--SON': [10, 8, 8, 8]}
lags = {'SON': [10, 10, 10, 10]}
effects_dict = set_actor_effect_dict(target='u50',
                                     totales=['dmi:dmi+n34','n34:n34',],
                                     directos=['dmi', 'n34'])

effects_dict_n34_dmi = set_actor_effect_dict(target='dmi',
                                     totales=['n34:n34'],
                                     directos=['n34'])

# El periodo de la variable ordena el resto
hgt_anom_mon = hgt_anom_mon.sel(
    time=hgt_anom_mon.time.dt.year.isin(range(1959, 2021)))

df_n34_dmi = None
df = None
for l_count, lag_key in enumerate(lags.keys()):
    lag_target, indices_lags = identify_lags(lags[lag_key])

    print(f'{lag_key}')
    df_linea = df_linea_lag(lag_key)

    hgt_target, actor_list = SetLag_to_ActorList(
        variable_target=hgt_anom_mon,
        month_target=lag_target,
        indices=indices,
        lags=indices_lags,
        years_to_remove=[2002, 2019],
        verbose=0)

    aux_df_n34_dmi = apply_CEN_effect(actor_list,
                                      effects_dict_n34_dmi,
                                      sig=True,
                                      alpha_sig=[0.15, 0.1, 0.05])
    aux_df_n34_dmi = concat_df(df_linea, aux_df_n34_dmi)
    df_n34_dmi = concat_df(df_n34_dmi, aux_df_n34_dmi)

    aux_df = apply_CEN_effect(actor_list,
                              effects_dict,
                              sig=True,
                              alpha_sig=[0.15, 0.1, 0.05])
    aux_df = concat_df(df_linea, aux_df)
    df = concat_df(df, aux_df)

print('Done lags')

if save:
    df_n34_dmi.to_csv(f'{out_dir}cen_n34-dmi_trinity_son.csv', index=False)
    df.to_csv(f'{out_dir}cen_trinity_son.csv', index=False)
    print('Saved')
    print(f'{out_dir}cen_n34-dmi_trinity_son.csv')
    print(f'{out_dir}cen_trinity_son.csv')

if plots:
    from cen.cen import CEN_ufunc

    effects_dict = set_actor_effect_dict(target='u50',
                                         totales=['dmi:dmi+n34',
                                                  'n34:n34',
                                                  'u50:dmi+n34+u50'],
                                         directos=['dmi', 'n34', 'u50'],
                                         to_parallel_run=True)

    cen = CEN_ufunc(actor_list)

    regre_efectos_totales, regre_efectos_directos = \
        cen.run_ufunc_cen(variable_target=hgt_target.sel(lat=slice(20, -80)),
                          effects=effects_dict)

