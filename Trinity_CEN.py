"""
Red causal ENSO-IOD-U50hpa
"""
################################################################################
# Seteos generales ----------------------------------------------------------- #
save = False
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cn_effect/'
modname='Trinity'

# Caja de PP
pp_lons = [295, 310]
pp_lats = [-30, -40]
nombre_caja_pp = 's_sesa'

# Caja mar de Amundsen
amd_lons = [210, 270]
amd_lats = [-80, -50]
nombre_caja_amd = 'amd'
################################################################################
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
from ENSO_IOD_Funciones import Nino34CPC, DMI2
from cen_funciones import OpenObsDataSet, Detrend, Weights, \
    auxSetLags_ActorList, aux_alpha_CN_Effect_2
from CEN_ufunc import CEN_ufunc

################################################################################
if save:
    dpi = 200
else:
    dpi = 70

# if use_strato_index:
#     per = '1979_2020'
# else:
#     per = '1940_2020'
################################################################################
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
hgt_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
dir_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_no_detrend/'
################################################################################
################################################################################
# HGT ------------------------------------------------------------------------ #
hgt = xr.open_dataset(hgt_dir + 'ERA5_HGT200_40-20.nc')
hgt = hgt.rename({'longitude': 'lon', 'latitude': 'lat', 'z': 'var'})
hgt = hgt.interp(lon=np.arange(0,360,2), lat=np.arange(-90, 90, 2))

hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
hgt200_anom_or = hgt.groupby('time.month') - \
           hgt_clim.groupby('time.month').mean('time')

weights = np.sqrt(np.abs(np.cos(np.radians(hgt200_anom_or.lat))))
hgt200_anom_or = hgt200_anom_or * weights

hgt200_anom_or = hgt200_anom_or.rolling(time=3, center=True).mean()
hgt200_anom_or = hgt200_anom_or.sel(time=slice('1940-02-01', '2020-11-01'))

# 750hpa
# hgt = xr.open_dataset(hgt_dir + 'ERA5_HGT750_40-20.nc')
# hgt = hgt.rename({'longitude': 'lon', 'latitude': 'lat', 'z': 'var'})
# hgt = hgt.interp(lon=np.arange(0,360,2), lat=np.arange(-90, 90, 2))
#
# hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
# hgt750_anom_or = hgt.groupby('time.month') - \
#            hgt_clim.groupby('time.month').mean('time')
#
# weights = np.sqrt(np.abs(np.cos(np.radians(hgt750_anom_or.lat))))
# hgt750_anom_or = hgt750_anom_or * weights
#
# hgt750_anom_or = hgt750_anom_or.rolling(time=3, center=True).mean()
# hgt750_anom_or = hgt750_anom_or.sel(time=slice('1940-02-01', '2020-11-01'))

# PP ------------------------------------------------------------------------- #
pp_or = OpenObsDataSet(name='pp_pgcc_v2020_1891-2023_1', sa=True, dir=dir_pp)
pp_or = pp_or.rename({'precip':'var'})
pp_or = pp_or.sel(time=slice('1940-01-16', '2020-12-16'))

pp_or = Weights(pp_or)
pp_or = pp_or.sel(lat=slice(20, -60), lon=slice(270,330)) # SA
pp_or = pp_or.rolling(time=3, center=True).mean()
pp_or = pp_or.sel(time=pp_or.time.dt.month.isin([8,9,10,11]))
pp_or = Detrend(pp_or, 'time')

# Caja PP
pp_caja_or = pp_or.sel(lat=slice(pp_lats[0], pp_lats[1]),
                  lon=slice(pp_lons[0],pp_lons[1])).mean(['lon', 'lat'])
pp_caja_or['var'][-1]=0 # aca nse que pasa.

# ---------------------------------------------------------------------------- #
# indices
# ---------------------------------------------------------------------------- #
# sam_or = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
# sam_or = sam_or.rolling(time=3, center=True).mean()
#
# asam_or = xr.open_dataset(sam_dir + 'asam_700.nc')['mean_estimate']
# asam_or = asam_or.rolling(time=3, center=True).mean()
#
# ssam_or = xr.open_dataset(sam_dir + 'ssam_700.nc')['mean_estimate']
# ssam_or = ssam_or.rolling(time=3, center=True).mean()

dmi_or = DMI2(filter_bwa=False, start_per='1959', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

sst_aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/"
                      "sst.mnmean.nc")
sst_aux = sst_aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34_or = Nino34CPC(sst_aux, start=1920, end=2020)[0]

u50_or = xr.open_dataset('/pikachu/datos/luciano.andrian/observado/'
                         'ncfiles/ERA5/downloaded/ERA5_U50hpa_40-20.mon.nc')
u50_or = u50_or.rename({'u': 'var'})
u50_or = u50_or.rename({'longitude': 'lon'})
u50_or = u50_or.rename({'latitude': 'lat'})
u50_or = Weights(u50_or)
u50_or = u50_or.sel(lat=-60)
u50_or = u50_or - u50_or.mean('time')
u50_or = u50_or.rolling(time=3, center=True).mean()
u50_or = Detrend(u50_or, 'time')
u50_or = u50_or.sel(expver=1).drop('expver')
u50_or = u50_or.mean('lon')
u50_or = xr.DataArray(u50_or['var'].drop('lat'))
################################################################################
hgt200_anom_or =\
    hgt200_anom_or.sel(time=hgt200_anom_or.time.dt.year.isin(range(1959,2021)))

# QUE LAGS USAR ?????
lags = {'SON':[10,10,10],
        'ASO--SON':[10, 9, 10],
        'JAS_ASO--SON':[10, 8, 9],
        'JAS--SON':[10, 8, 8],
        #'JAS_ASO--OND':[11, 8, 9],
        'JAS--SON2':[10, 8, 10],
        'ASO':[9, 9, 9]}

coefs_dmi_u50 = [-0.01,-0.03, -0.11, -0.11, -0.06]
coefs_n34_u50 = [0.08, 0.11, 0.16, 0.16, 0.14]
coefs_dmi_asam = [-0.28, -0.27, -0.12, -0.009, -0.15]
coefs_n34_asam = [-0.40, -0.40, -0.53, -0.57, -0.49]
coefs_dmi_ssam = [-0.12, - 0.09, -0.04, 0.19, -0.009]
coefs_n34_ssam = [0, -0.03, 0.12, -0.38, -0.10]
coefs_u50_asam = [0.236, 0.23, 0.42, 0.53, 0.25]
coefs_u50_ssam = [0.636, 0.636, 0.689, 0.683, 0.644]

lags = {'SON':[10,10,10]}

for l_count, lag_key in enumerate(lags.keys()):
    seasons_lags = lags[lag_key]
    print(f"{lag_key} ########################################################")

    hgt200_anom, pp, asam, ssam, u50, strato_indice2, dmi, n34, actor_list, \
    dmi_aux, n34_aux, u50_aux, sam_aux, aux_ssam, aux_asam  = \
        auxSetLags_ActorList(lag_target=seasons_lags[0],
                             lag_dmin34=seasons_lags[1],
                             lag_strato=seasons_lags[2],
                             hgt200_anom_or=hgt200_anom_or, pp_or=pp_or,
                             dmi_or=dmi_or, n34_or=n34_or, u50_or=u50_or,
                             strato_indice=None,
                             years_to_remove=[2002, 2019])


    print(f"# {modname} CEN --------------------------------------------------")
    aux_alpha_CN_Effect_2(actor_list,
                          set_series_directo=['dmi', 'n34'],
                          set_series_totales={'dmi': ['dmi', 'n34'],
                                              'n34': ['n34']},
                          variables={'u50': u50},
                          sig=True, alpha_sig=[0.05, 0.1, 0.15, 1])


    print(f"# Plot -----------------------------------------------------------")
    cen = CEN_ufunc(actor_list)
    hgt200_anom2 = hgt200_anom.sel(lat=slice(-80, 20))

    # Los factores_sp sirven para los 3 modelos ya las relaciones no cambian
    # entre los actores xq estamos asumiendo
    # la función va tomar los sp que tenga como actores nomas

    factores_sp_u50 = {'u50': {'dmi->u50': coefs_dmi_u50[l_count],
                               'n34->u50': coefs_n34_u50[l_count]}}

    actors_and_sets_total = {'dmi': 'dmi:n34',
                             'n34': 'n34',
                             'u50': 'dmi:n34:u50'}

    actors_and_sets_direc = {'dmi': 'dmi:n34:u50',
                             'n34': 'dmi:n34:u50',
                             'u50': 'dmi:n34:u50'}

    cen.Compute_CEN_and_Plot([hgt200_anom2], ['hgt200'], ['hs'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp_u50,
                             aux_name=f"Mod_{modname}_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir,
                             actors_to_plot=['dmi', 'n34', 'u50'])

    cen.Compute_CEN_and_Plot([pp], ['pp'], ['sa'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp_u50,
                             aux_name=f"Mod_{modname}_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir)

print('#######################################################################')
print('# Done ################################################################')
print('#######################################################################')