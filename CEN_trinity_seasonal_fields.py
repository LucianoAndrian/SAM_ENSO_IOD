"""
CEN
ENSO-IOD-U 50hPa
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas_cen/'

# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import pandas as pd
pd.options.mode.chained_assignment = None
import xarray as xr
from cen.cen_funciones import set_actor_effect_dict, set_data_to_cen, \
    apply_cen_2d

# ---------------------------------------------------------------------------- #
# set data
hgt_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
hgt_anom_mon = set_data_to_cen(dir_file = f'{hgt_dir}ERA5_HGT200_40-20.nc',
                               interp_2x2=True, rolling=True, rl_win=3)
hgt_anom_mon = hgt_anom_mon.sel(
    time=hgt_anom_mon.time.dt.year.isin(range(1959, 2021)))
# indices -------------------------------------------------------------------- #
from CEN_set_actors import n34_or, dmi_or, u50_or

# CEN set -------------------------------------------------------------------- #
indices = {'n34':n34_or, 'dmi':dmi_or, 'u50':u50_or}

# lags, lag_variable + lags orden como indices
lags = {'SON': [10, 10, 10, 10],
        'ASO--SON': [10, 9, 9, 9],
        'ASO': [9, 9, 9, 9],
        'JAS_ASO--SON': [10, 8, 8, 9],
        'JAS--SON': [10, 8, 8, 8]}

lags = {'SON': [10, 10, 10, 10]}

# CEN ------------------------------------------------------------------------ #
# El periodo de la variable ordena el resto
effects_dict = set_actor_effect_dict(target='u50',
                                     totales=['dmi:dmi+n34', 'n34:n34',
                                              'u50:dmi+n34+u50'],
                                     directos=['dmi', 'n34', 'u50'],
                                     to_parallel_run=True)

efectos_totales, efectos_directos = apply_cen_2d(
    variable_target=hgt_anom_mon.sel(lat=slice(20, -80)),
    effects_dict=effects_dict,
    indices=indices,
    lags=lags,
    alpha=0.05,
    years_to_remove=[2002, 2019],
    log_level='info',
    verbose=0)

# ---------------------------------------------------------------------------- #
if save:
    print('Saving..')
    for k, v in efectos_directos.items():
        aux_v = v

        # lista de subkeys y valores
        sk_list = list(aux_v.keys())
        v_list = [aux_v[sk] for sk in sk_list]

        # concatenar con labels en la nueva dimensión
        da_out = xr.concat(v_list,
                           dim=xr.DataArray(sk_list, dims="actor", name="actor"))
        da_out.to_netcdf(f"{out_dir}cen_directo_2d_hgt200_{k.lower()}_seasonal.nc")

    print('Done directos')

    for k, v in efectos_totales.items():
        aux_v = v

        # lista de subkeys y valores
        sk_list = list(aux_v.keys())
        v_list = [aux_v[sk] for sk in sk_list]

        # concatenar con labels en la nueva dimensión
        da_out = xr.concat(v_list,
                           dim=xr.DataArray(sk_list, dims="actor", name="actor"))
        da_out.to_netcdf(f"{out_dir}cen_totales_2d_hgt200_{k.lower()}_seasonal.nc")

    print('Done totales')
# ---------------------------------------------------------------------------- #