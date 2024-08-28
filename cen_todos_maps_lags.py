"""
"Mapas causales" (regre coef) Redes cusales ENSO, IOD, SAM, ASAM, U50
con lags
"""
from test_regre_cn_w_lags import hgt_anom_lag1

################################################################################
# Seteos generales ----------------------------------------------------------- #
save = False
use_u50 = True
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cn_effect2/'

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
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2, ChangeLons
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

from CEN_ufunc import CEN_ufunc
################################################################################
if save:
    dpi = 200
else:
    dpi = 70
################################################################################
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
hgt_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
dir_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_no_detrend/'
################################################################################
# Funciones ------------------------------------------------------------------ #
def OpenObsDataSet(name, sa=True, dir=''):
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

def aux2_Setlag(serie_or, serie_lag, serie_set, years_to_remove):
    if serie_lag is not None:
        serie_f = serie_or.sel(
            time=serie_or.time.dt.month.isin([serie_lag]))
        serie_f = serie_f.sel(
            time=serie_f.time.dt.year.isin(serie_set.time.dt.year))
    else:
        serie_f = SameDateAs(serie_or, serie_set)

    serie_f = serie_f / serie_f.std()

    serie_f = serie_f.sel(time=~serie_f.time.dt.year.isin(years_to_remove))

    return serie_f

def auxSetLags_ActorList(lag_target, lag_dmin34, lag_strato, hgt200_anom_or,
                         pp_or, dmi_or, n34_or, asam_or, ssam_or, sam_or,
                         u50_or, strato_indice, years_to_remove=None,
                         asam_lag=None, ssam_lag=None, sam_lag=None,
                         auxdmi_lag=None, auxn34_lag=None, auxstrato_lag=None,
                         auxsam_lag=None, auxssam_lag=None, auxasam_lag=None,
                         auxhgt_lag=None, auxpp_lag=None):

    # lag_target
    if auxhgt_lag is None:
        hgt200_anom = hgt200_anom_or.sel(
            time=hgt200_anom_or.time.dt.month.isin([lag_target]))
    else:
        hgt200_anom = hgt200_anom_or.sel(
            time=hgt200_anom_or.time.dt.month.isin([auxhgt_lag]))

    if strato_indice is not None:
        hgt200_anom = hgt200_anom.sel(
            time=hgt200_anom.time.dt.year.isin([strato_indice.time]))

    hgt200_anom = hgt200_anom / hgt200_anom.std()
    hgt200_anom = hgt200_anom.sel(
        time=~hgt200_anom.time.dt.year.isin(years_to_remove))

    if auxpp_lag is None:
        pp = SameDateAs(pp_or, hgt200_anom)
    else:
        pp = pp_or.sel(
            time=pp_or.time.dt.month.isin([auxpp_lag]))
    pp = pp / pp.std()
    pp = pp.sel(time=~pp.time.dt.year.isin(years_to_remove))

    sam = aux2_Setlag(sam_or, sam_lag, hgt200_anom, years_to_remove)
    asam = aux2_Setlag(asam_or, asam_lag, hgt200_anom, years_to_remove)
    ssam = aux2_Setlag(ssam_or, ssam_lag, hgt200_anom, years_to_remove)

    dmi = aux2_Setlag(dmi_or, lag_dmin34, hgt200_anom, years_to_remove)
    n34 = aux2_Setlag(n34_or, lag_dmin34, hgt200_anom, years_to_remove)

    u50 = aux2_Setlag(u50_or, lag_strato, hgt200_anom, years_to_remove)

    dmi_aux = aux2_Setlag(dmi_or, auxdmi_lag, hgt200_anom, years_to_remove)
    n34_aux = aux2_Setlag(n34_or, auxn34_lag, hgt200_anom, years_to_remove)
    u50_aux = aux2_Setlag(u50_or, auxstrato_lag, hgt200_anom, years_to_remove)

    aux_sam = aux2_Setlag(sam_or, auxsam_lag, hgt200_anom, years_to_remove)
    aux_asam = aux2_Setlag(asam_or, auxasam_lag, hgt200_anom, years_to_remove)
    aux_ssam = aux2_Setlag(ssam_or, auxssam_lag, hgt200_anom, years_to_remove)


    if strato_indice is not None:
        strato_indice = strato_indice.sel(
            time=~strato_indice.time.isin(years_to_remove))
        actor_list = {'dmi': dmi.values, 'n34': n34.values, 'ssam': ssam.values,
                      'asam': asam.values,
                      'strato': strato_indice['var'].values,
                      'sam': sam.values, 'u50': u50.values,
                      'dmi_aux': dmi_aux.values, 'n34_aux':n34_aux.values,
                      'u50_aux':u50_aux.values, 'aux_sam':aux_sam.values,
                      'aux_ssam':aux_ssam.values, 'aux_asam':aux_asam.values}
    else:
        actor_list = {'dmi': dmi.values, 'n34': n34.values, 'ssam': ssam.values,
                      'asam': asam.values,
                      'strato': None,
                      'sam': sam.values, 'u50': u50.values,
                      'dmi_aux': dmi_aux.values, 'n34_aux':n34_aux.values,
                      'u50_aux':u50_aux.values, 'aux_sam':aux_sam.values,
                      'aux_ssam':aux_ssam.values, 'aux_asam':aux_asam.values}

    return (hgt200_anom, pp, asam, ssam, u50, strato_indice, dmi, n34,\
           actor_list, dmi_aux, n34_aux, u50_aux, aux_sam, aux_ssam,
            aux_asam)

def convertdates(dataarray, dimension, rename=None):
    fechas = pd.to_datetime(dataarray[dimension].values.astype(str),
                            format='%Y%m%d')
    dataarray[dimension] = fechas
    if rename is not None:
        dataarray = dataarray.rename({dimension: rename})
    return dataarray

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

dmi_or = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

sst_aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/"
                      "sst.mnmean.nc")
sst_aux = sst_aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34_or = Nino34CPC(sst_aux, start=1920, end=2020)[0]

strato_indice = None

hgt_lvls = xr.open_dataset(
    '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/ERA5_HGT500-10_79-20.mon.nc')

hgt_lvls = convertdates(hgt_lvls, 'date', 'time')
hgt_lvls = ChangeLons(hgt_lvls,'longitude')
hgt_lvls = hgt_lvls.rename({'latitude':'lat'})
hgt_lvls = hgt_lvls.sel(lat=-60)
hgt_lvls = (hgt_lvls.groupby('time.month') -
            hgt_lvls.groupby('time.month').mean('time'))
hgt_lvls = hgt_lvls.drop('expver')
hgt_lvls = hgt_lvls.drop('number')

first = True
for l in hgt_lvls.pressure_level.values:
    aux = hgt_lvls.sel(pressure_level=l)
    aux = aux / aux.std('time')

    if first:
        first = False
        hgt_lvls_nrm = aux
    else:
        hgt_lvls_nrm = xr.concat([hgt_lvls_nrm, aux], dim='pressure_level')

#hgt_lvls = hgt_lvls/hgt_lvls.std('time')


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
# nombre y lag_target, lag_dmin34 y lag_strato
lags = {'SON':[10,10,10],
        'ASO--SON':[10, 9, 10],
        'JAS-ASO--SON':[10, 8, 9],
        'JAS--SON':[10, 8, 8],
        #'JAS-ASO--OND':[11, 8, 9],
        'JAS--SON2':[10, 8, 10]}

coefs_dmi_u50 = [-0.01,-0.03, -0.11, -0.11, -0.06]
coefs_n34_u50 = [0.08, 0.11, 0.16, 0.16, 0.14]
coefs_dmi_asam = [-0.28, -0.27, -0.12, -0.009, -0.15]
coefs_n34_asam = [-0.40, -0.40, -0.53, -0.57, -0.49]
coefs_dmi_ssam = [-0.12, - 0.09, -0.04, 0.19, -0.009]
coefs_n34_ssam = [0, -0.03, 0.12, -0.38, -0.10]
coefs_u50_asam = [0.236, 0.23, 0.42, 0.53, 0.25]
coefs_u50_ssam = [0.636, 0.636, 0.689, 0.683, 0.644]

for l_count, lag_key in enumerate(lags.keys()):
    seasons_lags = lags[lag_key]

    hgt200_anom, pp, asam, ssam, u50, strato_indice2, dmi, n34, actor_list, \
    dmi_aux, n34_aux, u50_aux, sam_aux, aux_ssam, aux_asam  = \
        auxSetLags_ActorList(lag_target=seasons_lags[0],
                             lag_dmin34=seasons_lags[1],
                             lag_strato=seasons_lags[2],
                             hgt200_anom_or=hgt200_anom_or, pp_or=pp_or,
                             dmi_or=dmi_or, n34_or=n34_or, asam_or=asam_or,
                             ssam_or=ssam_or, sam_or=sam_or, u50_or=u50_or,
                             strato_indice=None,
                             years_to_remove=[2002, 2019])

    # test ------------------------------------------------------------------- #
    hgtlvls_anom, pp, asam, ssam, u50, strato_indice2, dmi, n34, actor_list, \
    dmi_aux, n34_aux, u50_aux, sam_aux, aux_ssam, aux_asam  = \
        auxSetLags_ActorList(lag_target=seasons_lags[0],
                             lag_dmin34=seasons_lags[1],
                             lag_strato=seasons_lags[2],
                             hgt200_anom_or=hgt_lvls_nrm, pp_or=pp_or,
                             dmi_or=dmi_or, n34_or=n34_or, asam_or=asam_or,
                             ssam_or=ssam_or, sam_or=sam_or, u50_or=u50_or,
                             strato_indice=None,
                             years_to_remove=[2002, 2019])

    cen = CEN_ufunc(actor_list)
    coef, pval = cen.compute_regression(hgtlvls_anom.z,
                                        actors_and_sets_direc['n34'],
                                        'n34', 0.5)

    Plot_vsP(pval, 'RdBu_r', save, dpi, 'titulo', 'name', out_dir)
    # test ------------------------------------------------------------------- #

    cen = CEN_ufunc(actor_list)
    hgt200_anom2 = hgt200_anom.sel(lat=slice(-80, 20))

    # Los factores_sp sirven para los 3 modelos ya las relaciones no cambian
    # entre los actores xq estamos asumiendo asam -x- ssam
    # la función va tomar los sp que tenga como actores nomas

    factores_sp = {'u50': {'dmi->u50': coefs_dmi_u50[l_count],
                           'n34->u50': coefs_n34_u50[l_count]},
                   'ssam': {'dmi->ssam': coefs_dmi_ssam[l_count],
                            'n34->ssam': coefs_n34_ssam[l_count],
                            'u50->ssam': coefs_u50_ssam[l_count]},
                   'asam': {'dmi->asam': coefs_dmi_asam[l_count],
                            'n34->asam': coefs_n34_asam[l_count],
                            'u50->asam': coefs_u50_asam[l_count]}}
    factores_sp = {}

    # Mod Full ASAM -x- SSAM - u50 ---------------------------------------------
    actors_and_sets_total = {'dmi': 'dmi:n34',
                             'n34': 'n34',
                             'u50': 'dmi:n34:u50',
                             'asam': 'dmi:n34:u50:asam',
                             'ssam': 'dmi:n34:ssam:u50'}

    actors_and_sets_direc = {'dmi': 'dmi:n34:u50:asam:ssam',
                             'n34': 'dmi:n34:u50:asam:ssam',
                             'u50': 'dmi:n34:u50:asam:ssam',
                             'asam': 'dmi:n34:u50:asam',
                             'ssam': 'dmi:n34:ssam:u50'}

    cen.Compute_CEN_and_Plot([hgt200_anom2], ['hgt200'], ['hs'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp,
                             aux_name=f"Mod_ASAMxSSAM_u50_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir,
                             actors_to_plot = ['n34', 'dmi', 'u50'])

    cen.Compute_CEN_and_Plot([pp], ['pp'], ['sa'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp,
                             aux_name=f"Mod_ASAMxSSAM_u50_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir)

    # Mod Full ASAM - u50 ------------------------------------------------------
    actors_and_sets_total = {'dmi': 'dmi:n34',
                             'n34': 'n34',
                             'u50': 'dmi:n34:u50',
                             'asam': 'dmi:n34:u50:asam'}

    actors_and_sets_direc = {'dmi': 'dmi:n34:u50:asam',
                             'n34': 'dmi:n34:u50:asam',
                             'u50': 'dmi:n34:u50:asam',
                             'asam': 'dmi:n34:u50:asam'}

    cen.Compute_CEN_and_Plot([hgt200_anom2], ['hgt200'], ['hs'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp,
                             aux_name=f"Mod_ASAM_u50_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir,
                             actors_to_plot=['dmi', 'n34', 'u50'])

    cen.Compute_CEN_and_Plot([pp], ['pp'], ['sa'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp,
                             aux_name=f"Mod_ASAM_u50_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir)

    # Mod Full SSAM - u50 ------------------------------------------------------
    actors_and_sets_total = {'dmi': 'dmi:n34',
                             'n34': 'n34',
                             'u50': 'dmi:n34:u50',
                             'ssam': 'dmi:n34:u50:ssam'}

    actors_and_sets_direc = {'dmi': 'dmi:n34:u50:ssam',
                             'n34': 'dmi:n34:u50:ssam',
                             'u50': 'dmi:n34:u50:ssam',
                             'ssam': 'dmi:n34:u50:ssam'}

    cen.Compute_CEN_and_Plot([hgt200_anom2], ['hgt200'], ['hs'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp,
                             aux_name=f"Mod_SSAM_u50_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir,
                             actors_to_plot=['dmi', 'n34', 'u50'])

    cen.Compute_CEN_and_Plot([pp], ['pp'], ['sa'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp,
                             aux_name=f"Mod_SSAM_u50_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir)


    # Los mapas de regresión de estos modelos simples con lags pueden
    # ser relevantes

    # Mod dmi, n34 - u50 -------------------------------------------------------
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
                             aux_name=f"Mod_u50_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir,
                             actors_to_plot=['dmi', 'n34', 'u50'])

    cen.Compute_CEN_and_Plot([pp], ['pp'], ['sa'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp_u50,
                             aux_name=f"Mod_u50_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir)

    # Mod dmi, n34 - asam ------------------------------------------------------
    factores_sp_asam = {'asam': {'dmi->asam': coefs_dmi_asam[l_count],
                                 'n34->asam': coefs_n34_asam[l_count]}}

    actors_and_sets_total = {'dmi': 'dmi:n34',
                             'n34': 'n34',
                             'asam': 'dmi:n34:asam'}

    actors_and_sets_direc = {'dmi': 'dmi:n34:asam',
                             'n34': 'dmi:n34:asam',
                             'asam': 'dmi:n34:asam'}

    cen.Compute_CEN_and_Plot([hgt200_anom2], ['hgt200'], ['hs'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp_asam,
                             aux_name=f"Mod_ASAM_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir,
                             actors_to_plot=['dmi', 'n34', 'u50'])

    cen.Compute_CEN_and_Plot([pp], ['pp'], ['sa'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp_asam,
                             aux_name=f"Mod_ASAM_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir)

    # Mod dmi, n34 - u50 -------------------------------------------------------

    factores_sp_ssam = {'ssam': {'dmi->ssam': coefs_dmi_ssam[l_count],
                            'n34->ssam': coefs_n34_ssam[l_count]}}

    actors_and_sets_total = {'dmi': 'dmi:n34',
                             'n34': 'n34',
                             'ssam': 'dmi:n34:ssam'}

    actors_and_sets_direc = {'dmi': 'dmi:n34:ssam',
                             'n34': 'dmi:n34:ssam',
                             'ssam': 'dmi:n34:ssam'}

    cen.Compute_CEN_and_Plot([hgt200_anom2], ['hgt200'], ['hs'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp_ssam,
                             aux_name=f"Mod_SSAM_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir)

    cen.Compute_CEN_and_Plot([pp], ['pp'], ['sa'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp_ssam,
                             aux_name=f"Mod_SSAM_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir,
                             actors_to_plot=['dmi', 'n34', 'u50'])
################################################################################


lags = {'ASO':[9,9,9]}

coefs_dmi_u50 = [-0.16]
coefs_n34_u50 = [0.22]
coefs_dmi_asam = [-0.23]
coefs_n34_asam = [-0.44]
coefs_dmi_ssam = [-0.035]
coefs_n34_ssam = [-0.06]
coefs_u50_asam = [0.26]
coefs_u50_ssam = [0.512]

for l_count, lag_key in enumerate(lags.keys()):
    seasons_lags = lags[lag_key]

    hgt200_anom, pp, asam, ssam, u50, strato_indice2, dmi, n34, actor_list, \
    dmi_aux, n34_aux, u50_aux, sam_aux, aux_ssam, aux_asam  = \
        auxSetLags_ActorList(lag_target=seasons_lags[0],
                             lag_dmin34=seasons_lags[1],
                             lag_strato=seasons_lags[2],
                             hgt200_anom_or=hgt200_anom_or, pp_or=pp_or,
                             dmi_or=dmi_or, n34_or=n34_or, asam_or=asam_or,
                             ssam_or=ssam_or, sam_or=sam_or, u50_or=u50_or,
                             strato_indice=None, auxssam_lag=10, auxasam_lag=10,
                             auxhgt_lag=10, auxpp_lag=10,
                             years_to_remove=[2002, 2019])

    cen = CEN_ufunc(actor_list)
    hgt200_anom2 = hgt200_anom.sel(lat=slice(-80, 20))

    factores_sp = {'u50': {'dmi->u50': coefs_dmi_u50[l_count],
                           'n34->u50': coefs_n34_u50[l_count]},
                   'ssam': {'dmi->ssam': coefs_dmi_ssam[l_count],
                            'n34->ssam': coefs_n34_ssam[l_count],
                            'u50->ssam': coefs_u50_ssam[l_count]},
                   'asam': {'dmi->asam': coefs_dmi_asam[l_count],
                            'n34->asam': coefs_n34_asam[l_count],
                            'u50->asam': coefs_u50_asam[l_count]}}

    # Mod Full ASAM -x- SSAM - u50 ---------------------------------------------
    actors_and_sets_total = {'dmi': 'dmi:n34',
                             'n34': 'n34',
                             'u50': 'dmi:n34:u50',
                             'asam': 'dmi:n34:u50:asam',
                             'ssam': 'dmi:n34:ssam:u50'}

    actors_and_sets_direc = {'dmi': 'dmi:n34:u50:asam:ssam',
                             'n34': 'dmi:n34:u50:asam:ssam',
                             'u50': 'dmi:n34:u50:asam:ssam',
                             'asam': 'dmi:n34:u50:asam',
                             'ssam': 'dmi:n34:ssam:u50'}

    cen.Compute_CEN_and_Plot([hgt200_anom2], ['hgt200'], ['hs'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp,
                             aux_name=f"Mod_ASAMxSSAM_u50_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir,
                             #actors_to_plot=['dmi', 'n34', 'u50'])
                             actors_to_plot = ['asam', 'ssam'])

    cen.Compute_CEN_and_Plot([pp], ['pp'], ['sa'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp,
                             aux_name=f"Mod_ASAMxSSAM_u50_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir)

    hgt200_anom, pp, asam, ssam, u50, strato_indice2, dmi, n34, actor_list, \
    dmi_aux, n34_aux, u50_aux, sam_aux, aux_ssam, aux_asam  = \
        auxSetLags_ActorList(lag_target=seasons_lags[0],
                             lag_dmin34=seasons_lags[1],
                             lag_strato=seasons_lags[2],
                             hgt200_anom_or=hgt200_anom_or, pp_or=pp_or,
                             dmi_or=dmi_or, n34_or=n34_or, asam_or=asam_or,
                             ssam_or=ssam_or, sam_or=sam_or, u50_or=u50_or,
                             strato_indice=None, auxssam_lag=10, auxasam_lag=10,
                             auxhgt_lag=10, auxpp_lag=10,
                             years_to_remove=[2002, 2019])

    cen = CEN_ufunc(actor_list)
    hgt200_anom2 = hgt200_anom.sel(lat=slice(-80, 20))

    factores_sp = {'u50': {'dmi->u50': coefs_dmi_u50[l_count],
                           'n34->u50': coefs_n34_u50[l_count]},
                   'ssam': {'dmi->ssam': coefs_dmi_ssam[l_count],
                            'n34->ssam': coefs_n34_ssam[l_count],
                            'u50->ssam': coefs_u50_ssam[l_count]},
                   'asam': {'dmi->asam': coefs_dmi_asam[l_count],
                            'n34->asam': coefs_n34_asam[l_count],
                            'u50->asam': coefs_u50_asam[l_count]}}

    # Mod Full DMI, N34 - u50 --------------------------------------------------
    actors_and_sets_total = {'dmi': 'dmi:n34',
                             'n34': 'n34',
                             'u50': 'dmi:n34:u50'}

    actors_and_sets_direc = {'dmi': 'dmi:n34:u50',
                             'n34': 'dmi:n34:u50',
                             'u50': 'dmi:n34:u50'}

    cen.Compute_CEN_and_Plot([hgt200_anom2], ['hgt200'], ['hs'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp,
                             aux_name=f"Mod_ASAMxSSAM_NOSAMs_u50_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir,
                             #actors_to_plot=['dmi', 'n34', 'u50'])
                             actors_to_plot = ['n34', 'dmi', 'u50'])

    cen.Compute_CEN_and_Plot([pp], ['pp'], ['sa'],
                             actors_and_sets_total, actors_and_sets_direc,
                             save=save, factores_sp=factores_sp,
                             aux_name=f"Mod_ASAMxSSAM_NOSAMs_u50_LAG-{lag_key}",
                             alpha=0.10, out_dir=out_dir)




import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib
def Plot_vsP(data, cmap, save, dpi, titulo, name_fig, out_dir,
         step=1, data_ctn=None):

    fig_size = (5, 5)

    xticks = np.arange(0, 360, 30)
    yticks = np.arange(-60, 40, 20)
    contour = False

    levels = [-1, -.8, -.6, -.4, -.2, -.1, 0, .1, .2, .4, .6, .8, 1]

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.axes()

    if data_ctn is not None:
        levels_ctn = levels.copy()
        try:
            if isinstance(levels_ctn, np.ndarray):
                levels_ctn = levels_ctn[levels_ctn != 0]
            else:
                levels_ctn.remove(0)
        except:
            pass

        ax.contour(data.lon.values[::step], data.lat.values[::step],
                   data_ctn[::step, ::step], linewidths=0.8,
                   levels=levels_ctn, colors='black')


    im = ax.contourf(data.lon.values[::step], data.pressure_level.values[::step],
                     data[::step, ::step],
                     levels=levels, cmap=cmap, extend='both')

    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)

    ax.set_xticks(xticks)
    #ax.set_yticks(yticks, crs=crs_latlon)

    lon_formatter = LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.tick_params(labelsize=7)

    plt.yscale('log')
    ax.set_ylabel("Pressure [hPa]")
    ax.set_yscale('log')
    ax.set_ylim(10.*np.ceil(coef.pressure_level.values.max()/10.), 30)
    subs = [1,2,5]
    if coef.pressure_level.values.max()/100 < 30.:
        subs = [1,2,3,4,5,6,7,8,9]
    y1loc = matplotlib.ticker.LogLocator(base=10., subs=subs)
    ax.yaxis.set_major_locator(y1loc)
    fmt = matplotlib.ticker.FormatStrFormatter("%g")
    ax.yaxis.set_major_formatter(fmt)

    plt.title(titulo, fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig + '.jpg', dpi = dpi)
        plt.close()
    else:
        plt.show()

