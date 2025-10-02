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
    apply_cen_2d#, OpenObsDataSet

# aux finciones -------------------------------------------------------------- #
def aux_save_as_nc(dict_to_save, efecto_name_file, name_variable_file, out_dir):
    print('Saving..')
    for k, v in dict_to_save.items():
        aux_v = v

        sk_list = list(aux_v.keys())
        v_list = [aux_v[sk] for sk in sk_list]

        da_out = xr.concat(v_list,
                           dim=xr.DataArray(sk_list, dims="actor",
                                            name="actor"))

        da_out.to_netcdf(f"{out_dir}cen_{efecto_name_file}_2d_"
                         f"{name_variable_file}_{k.lower()}_seasonal.nc")

    print('Done')

# ---------------------------------------------------------------------------- #
# set data
hgt_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
hgt_anom_mon = set_data_to_cen(dir_file = f'{hgt_dir}ERA5_HGT200_40-20.nc',
                               interp_2x2=True, rolling=True, rl_win=3,
                               select_lat=[20, -80])
hgt_anom_mon = hgt_anom_mon.sel(
    time=hgt_anom_mon.time.dt.year.isin(range(1959, 2021)))

pp_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_no_detrend/'
prec_anom_mon = set_data_to_cen(dir_file = f'{pp_dir}pp_pgcc_v2020_1891-2023_1.nc',
                               interp_2x2=False, rolling=True, rl_win=3,
                                select_lon=[270, 330], select_lat=[15, -60])
prec_anom_mon = prec_anom_mon.sel(
    time=prec_anom_mon.time.dt.year.isin(range(1959, 2021)))
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

# CEN ------------------------------------------------------------------------ #
# El periodo de la variable ordena el resto
effects_dict = set_actor_effect_dict(target='u50',
                                     totales=['dmi:dmi+n34', 'n34:n34',
                                              'u50:dmi+n34+u50'],
                                     directos=['dmi', 'n34', 'u50'],
                                     to_parallel_run=True)

# hgt200 --------------------------------------------------------------------- #
print('hgt200')
efectos_totales, efectos_directos = apply_cen_2d(
    variable_target=hgt_anom_mon,
    effects_dict=effects_dict,
    indices=indices,
    lags=lags,
    alpha=0.05,
    years_to_remove=[2002, 2019],
    log_level='info',
    verbose=0)

if save:
    aux_save_as_nc(dict_to_save=efectos_directos,
                   efecto_name_file='directo',
                   name_variable_file='hgt200',
                   out_dir=out_dir)

    aux_save_as_nc(dict_to_save=efectos_totales,
                   efecto_name_file='totales',
                   name_variable_file='hgt200',
                   out_dir=out_dir)

# ---------------------------------------------------------------------------- #
# prec ----------------------------------------------------------------------- #
print('prec')
efectos_totales, efectos_directos = apply_cen_2d(
    variable_target=prec_anom_mon,
    effects_dict=effects_dict,
    indices=indices,
    lags=lags,
    alpha=0.05,
    years_to_remove=[2002, 2019],
    log_level='info',
    verbose=0)

if save:
    aux_save_as_nc(dict_to_save=efectos_directos,
                   efecto_name_file='directo',
                   name_variable_file='prec',
                   out_dir=out_dir)

    aux_save_as_nc(dict_to_save=efectos_totales,
                   efecto_name_file='totales',
                   name_variable_file='prec',
                   out_dir=out_dir)
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #