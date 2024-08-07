"""
Redes cusales ENSO, IOD, SAM, ASAM, STRATO

pasado en limpio sólo para la red mas grande que incluye tdo
"""
################################################################################
# Seteos generales ----------------------------------------------------------- #
save = False
use_strato_index = False
use_u50 = True
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cn_effect/'

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
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2
from cen_funciones import CN_Effect_2
from cen_funciones import OpenObsDataSet, Detrend, Weights, aux2_Setlag, \
    auxSetLags_ActorList, aux_alpha_CN_Effect_2, CN_Effect_2
# import matplotlib.pyplot as plt
# import cartopy.feature
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import cartopy.crs as ccrs
# from Scales_Cbars import get_cbars
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

hgt = xr.open_dataset(hgt_dir + 'ERA5_HGT750_40-20.nc')
hgt = hgt.rename({'longitude': 'lon', 'latitude': 'lat', 'z': 'var'})
hgt = hgt.interp(lon=np.arange(0,360,2), lat=np.arange(-90, 90, 2))

hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
hgt750_anom_or = hgt.groupby('time.month') - \
           hgt_clim.groupby('time.month').mean('time')

weights = np.sqrt(np.abs(np.cos(np.radians(hgt750_anom_or.lat))))
hgt750_anom_or = hgt750_anom_or * weights

hgt750_anom_or = hgt750_anom_or.rolling(time=3, center=True).mean()
hgt750_anom_or = hgt750_anom_or.sel(time=slice('1940-02-01', '2020-11-01'))

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
sam_or = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam_or = sam_or.rolling(time=3, center=True).mean()

asam_or = xr.open_dataset(sam_dir + 'asam_700.nc')['mean_estimate']
asam_or = asam_or.rolling(time=3, center=True).mean()

ssam_or = xr.open_dataset(sam_dir + 'ssam_700.nc')['mean_estimate']
ssam_or = ssam_or.rolling(time=3, center=True).mean()

dmi_or = DMI2(filter_bwa=False, start_per='1959', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

sst_aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/"
                      "sst.mnmean.nc")
sst_aux = sst_aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34_or = Nino34CPC(sst_aux, start=1920, end=2020)[0]

if use_strato_index:
    strato_indice = xr.open_dataset('strato_index.nc').rename({'year':'time'})
    strato_indice = strato_indice.rename(
        {'__xarray_dataarray_variable__':'var'})
    hgt200_anom_or2 = hgt200_anom_or.sel(time =
                            hgt200_anom_or.time.dt.year.isin(
                                strato_indice['time']))
    strato_indice = strato_indice.sel(time = range(1979,2021))
else:
    strato_indice = None

if use_u50:
    u50_or = xr.open_dataset('/pikachu/datos/luciano.andrian/observado/'
                           'ncfiles/ERA5/downloaded/ERA5_U50hpa_40-20.mon.nc')
    u50_or = u50_or.rename({'u': 'var'})
    u50_or = u50_or.rename({'longitude': 'lon'})
    u50_or = u50_or.rename({'latitude': 'lat'})
    u50_or = Weights(u50_or)
    u50_or = u50_or.sel(lat=-60)
    u50_or = u50_or - u50_or.mean('time')
    u50_or = u50_or.rolling(time=3, center=True).mean()
    #u50 = u50.sel(time=u50.time.dt.month.isin(mm))
    u50_or = Detrend(u50_or, 'time')
    u50_or = u50_or.sel(expver=1).drop('expver')
    u50_or = u50_or.mean('lon')
    u50_or = xr.DataArray(u50_or['var'].drop('lat'))

################################################################################
# Comparación u50 vs strato caja
# u50 en SON
# YA ESTA
################################################################################
# (hgt200_anom, pp, asam, ssam, u50, strato_indice2, dmi, n34, actor_list,
#  dmi_aux, n34_aux, u50_aux, sam_aux, asam_aux, ssam_aux) = (
#     auxSetLags_ActorList(lag_target=10,
#                          lag_dmin34=10,
#                          lag_strato=10,
#                          hgt200_anom_or=hgt200_anom_or, pp_or=pp_or,
#                          dmi_or=dmi_or, n34_or=n34_or, asam_or=asam_or,
#                          ssam_or=ssam_or, sam_or=sam_or, u50_or=u50_or,
#                          strato_indice=strato_indice,
#                          years_to_remove=[2002, 2019]))
#
# print('DMI, N34 - STRATO -----------------------------------------------------')
# aux_alpha_CN_Effect_2(actor_list,
#                       set_series_directo=['dmi', 'n34'],
#                       set_series_totales={'dmi': ['dmi', 'n34'],
#                                           'n34': ['n34']},
#                       variables={'strato' : strato_indice2['var']},
#                       sig=True, alpha_sig=[0.05, 0.1, 0.15, 1])
#
# print('DMI, N34 - U50 --------------------------------------------------------')
# aux_alpha_CN_Effect_2(actor_list,
#                       set_series_directo=['dmi', 'n34'],
#                       set_series_totales={'dmi': ['dmi', 'n34'],
#                                           'n34': ['n34']},
#                       variables={'u50' : u50},
#                       sig=True, alpha_sig=[0.05, 0.1, 0.15, 1])

################################################################################
# Usando U, no STRATO.
# Periodo 1959-2020
hgt200_anom_or =\
    hgt200_anom_or.sel(time=hgt200_anom_or.time.dt.year.isin(range(1959,2021)))

lags = {'SON':[10,10,10],
        'ASO--SON':[10, 9, 10],
        'JAS_ASO--SON':[10, 8, 9],
        'JAS--SON':[10, 8, 8],
        #'JAS_ASO--OND':[11, 8, 9],
        'JAS--SON2':[10, 8, 10],
        'ASO':[9, 9, 9]}

for lag_key in lags.keys():
    seasons_lags = lags[lag_key]
    print(f"{lag_key} ########################################################")

    hgt200_anom, pp, asam, ssam, u50, strato_indice2, dmi, n34, actor_list, \
    dmi_aux, n34_aux, u50_aux, sam_aux, aux_ssam, aux_asam  = \
        auxSetLags_ActorList(lag_target=seasons_lags[0],
                             lag_dmin34=seasons_lags[1],
                             lag_strato=seasons_lags[2],
                             hgt200_anom_or=hgt200_anom_or, pp_or=pp_or,
                             dmi_or=dmi_or, n34_or=n34_or, asam_or=asam_or,
                             ssam_or=ssam_or, sam_or=sam_or, u50_or=u50_or,
                             strato_indice=None, #auxssam_lag=9, auxasam_lag=9,
                             years_to_remove=[2002, 2019])
    # print('DMI, N34 - U50 ----------------------------------------------------')
    # aux_alpha_CN_Effect_2(actor_list,
    #                       set_series_directo=['u50', 'n34'],
    #                       set_series_totales={'u50': ['u50', 'n34'],
    #                                           'n34': ['n34']},
    #                       variables={'dmi': dmi},
    #                       sig=True, alpha_sig=[0.05, 0.1, 0.15, 1])



    print('DMI, N34 - U50 ----------------------------------------------------')
    aux_alpha_CN_Effect_2(actor_list,
                          set_series_directo=['dmi', 'n34'],
                          set_series_totales={'dmi': ['dmi', 'n34'],
                                              'n34': ['n34']},
                          variables={'u50': u50},
                          sig=True, alpha_sig=[0.05, 0.1, 0.15, 1])

    print('DMI, N34 ----------------------------------------------------------')
    aux_alpha_CN_Effect_2(actor_list,
                          set_series_directo=['n34'],
                          set_series_totales={'n34': ['n34']},
                          variables={'dmi': dmi},
                          sig=True, alpha_sig=[0.05, 0.1, 0.15, 1])

    # # ------------------------------------------------------------------------ #
    # print('DMI, N34 - SSAM ---------------------------------------------------')
    # aux_alpha_CN_Effect_2(actor_list,
    #                       set_series_directo=['dmi', 'n34'],
    #                       set_series_totales={'dmi': ['dmi', 'n34'],
    #                                           'n34': ['n34']},
    #                       variables={'ssam': ssam},
    #                       sig=True, alpha_sig=[0.05, 0.1, 0.15, 1])

    print('DMI, N34, U50 - SSAM ----------------------------------------------')
    aux_alpha_CN_Effect_2(actor_list,
                          set_series_directo=['dmi', 'n34', 'u50'],
                          set_series_totales={'dmi': ['dmi', 'n34'],
                                              'n34': ['n34'],
                                              'u50': ['dmi', 'n34', 'u50']},
                          variables={'aux_ssam': aux_ssam},
                          sig=True, alpha_sig=[0.05, 0.1, 0.15, 1])

    # # ------------------------------------------------------------------------ #
    # print('DMI, N34 - ASAM ---------------------------------------------------')
    # aux_alpha_CN_Effect_2(actor_list,
    #                       set_series_directo=['dmi', 'n34'],
    #                       set_series_totales={'dmi': ['dmi', 'n34'],
    #                                           'n34': ['n34']},
    #                       variables={'asam': asam},
    #                       sig=True, alpha_sig=[0.05, 0.1, 0.15, 1])

    print('DMI, N34, U50 - ASAM ----------------------------------------------')
    aux_alpha_CN_Effect_2(actor_list,
                          set_series_directo=['dmi', 'n34', 'u50'],
                          set_series_totales={'dmi': ['dmi', 'n34'],
                                              'n34': ['n34'],
                                              'u50': ['dmi', 'n34', 'u50']},
                          variables={'aux_asam': aux_asam},
                          sig=True, alpha_sig=[0.05, 0.1, 0.15, 1])


# ---------------------------------------------------------------------------- #
