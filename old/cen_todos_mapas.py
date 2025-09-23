"""
Redes cusales ENSO, IOD, SAM, ASAM, STRATO

pasado en limpio sólo para la red mas grande que incluye tdo
"""
################################################################################
# Seteos generales ----------------------------------------------------------- #
save = True
use_strato_index = True
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
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2
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

if use_strato_index:
    per = '1979_2020'
else:
    per = '1940_2020'
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

def auxSetLags_ActorList(lag_target, lag_dmin34, lag_strato, hgt200_anom_or,
                         pp_or, dmi_or, n34_or, asam_or, ssam_or, sam_or,
                         u50_or, strato_indice, years_to_remove=None):

    # lag_target
    hgt200_anom = hgt200_anom_or.sel(
        time=hgt200_anom_or.time.dt.month.isin([lag_target]))
    if strato_indice is not None:
        hgt200_anom = hgt200_anom.sel(
            time=hgt200_anom.time.dt.year.isin([strato_indice.time]))

    pp = SameDateAs(pp_or, hgt200_anom)
    sam = SameDateAs(sam_or, hgt200_anom)
    asam = SameDateAs(asam_or, hgt200_anom)
    ssam = SameDateAs(ssam_or, hgt200_anom)

    # lag_dmin34
    dmi = dmi_or.sel(time=dmi_or.time.dt.month.isin([lag_dmin34]))
    dmi = dmi.sel(time=dmi.time.dt.year.isin(hgt200_anom.time.dt.year))
    n34 = SameDateAs(n34_or, dmi)

    # lag_strato
    u50 = u50_or.sel(time=u50_or.time.dt.month.isin([lag_strato]))
    u50 = u50.sel(time=u50.time.dt.year.isin(hgt200_anom.time.dt.year))

    dmi = dmi / dmi.std()
    n34 = n34 / n34.std()
    sam = sam / sam.std()
    asam = asam / asam.std()
    ssam = ssam / ssam.std()
    u50 = u50 / u50.std()
    hgt200_anom = hgt200_anom / hgt200_anom.std()
    pp = pp / pp.std()

    hgt200_anom = hgt200_anom.sel(
        time=~hgt200_anom.time.dt.year.isin(years_to_remove))
    pp = pp.sel(time=~pp.time.dt.year.isin(years_to_remove))
    sam = sam.sel(time=~sam.time.dt.year.isin(years_to_remove))
    asam = asam.sel(time=~asam.time.dt.year.isin(years_to_remove))
    ssam = ssam.sel(time=~ssam.time.dt.year.isin(years_to_remove))
    n34 = n34.sel(time=~n34.time.dt.year.isin(years_to_remove))
    dmi = dmi.sel(time=~dmi.time.dt.year.isin(years_to_remove))
    u50 = u50.sel(time=~u50.time.dt.year.isin(years_to_remove))

    if strato_indice is not None:
        strato_indice = strato_indice.sel(
            time=~strato_indice.time.isin(years_to_remove))
        actor_list = {'dmi': dmi.values, 'n34': n34.values, 'ssam': ssam.values,
                      'asam': asam.values,
                      'strato': strato_indice['var'].values,
                      'sam': sam.values, 'u50': u50.values}

    else:
        actor_list = {'dmi': dmi.values, 'n34': n34.values, 'ssam': ssam.values,
                      'asam': asam.values,
                      'strato': None,
                      'sam': sam.values, 'u50': u50.values}

    return hgt200_anom, pp, asam, ssam, u50, strato_indice, dmi, n34, actor_list


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
# ---------------------------------------------------------------------------- #
# comparación strato vs u50 - en SON
# ---------------------------------------------------------------------------- #
hgt200_anom, pp, asam, ssam, u50, strato_indice2, dmi, n34, actor_list = \
    auxSetLags_ActorList(lag_target=10,
                         lag_dmin34=10,
                         lag_strato=10,
                         hgt200_anom_or=hgt200_anom_or,  pp_or=pp_or,
                         dmi_or=dmi_or, n34_or=n34_or, asam_or=asam_or,
                         ssam_or=ssam_or, sam_or=sam_or,
                         u50_or=u50_or,
                         strato_indice=strato_indice,
                         years_to_remove=[2002, 2019])

cen = CEN_ufunc(actor_list)
hgt200_anom2 = hgt200_anom.sel(lat=slice(-80, 20))

print('DMI, N34 - STRATO -----------------------------------------------------')
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'strato':'dmi:n34:strato'}

actors_and_sets_direc = {'dmi':'dmi:n34:strato',
                         'n34':'dmi:n34:strato',
                         'strato':'dmi:n34:strato'}

factores_sp = None # es irrelevante al quitar los SSW

cen.Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                         actors_and_sets_total, actors_and_sets_direc,
                         save=save, factores_sp=None, aux_name='Mod_strato',
                         alpha=0.10, out_dir=out_dir, actors_to_plot=['dmi'])

print('DMI, N34 - U50 --------------------------------------------------------')
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'u50':'dmi:n34:u50'}

actors_and_sets_direc = {'dmi':'dmi:n34:u50',
                         'n34':'dmi:n34:u50',
                         'u50':'dmi:n34:u50'}

factores_sp = None # es irrelevante al quitar los SSW

cen.Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                         actors_and_sets_total, actors_and_sets_direc,
                         save=save, factores_sp=None, aux_name='Mod_u50',
                         alpha=0.10, out_dir=out_dir)


 # ????
# ---------------------------------------------------------------------------- #
# Solo con U
# ---------------------------------------------------------------------------- #
hgt200_anom, pp, asam, ssam, u50, strato_indice2, dmi, n34, actor_list = \
    auxSetLags_ActorList(lag_target=10,
                         lag_dmin34=10,
                         lag_strato=10,
                         hgt200_anom_or=hgt200_anom_or,  pp_or=pp_or,
                         dmi_or=dmi_or, n34_or=n34_or, asam_or=asam_or,
                         ssam_or=ssam_or, sam_or=sam_or,
                         u50_or=u50_or,
                         strato_indice=None,
                         years_to_remove=[2002, 2019])

cen = CEN_ufunc(actor_list)



# ---------------------------------------------------------------------------- #
print('DMI, N34 - SSAM -------------------------------------------------------')
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'ssam':'dmi:n34:ssam'}

actors_and_sets_direc = {'dmi':'dmi:n34:ssam',
                         'n34':'dmi:n34:ssam',
                         'ssam':'dmi:n34:ssam'}

# factores_sp = {'ssam' : {'dmi-ssam':-0.28,
#                          'n34-dmi-ssam':0.635*-0.28},
#                'dmi':{'n34-dmi':0.635}}

factores_sp = None

cen.Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                         actors_and_sets_total, actors_and_sets_direc,
                         save=save, factores_sp=factores_sp,
                         aux_name='Mod_DMI-N34-SSAM',  alpha=0.10,
                         out_dir=out_dir, actors_to_plot=['dmi'])

print('DMI, N34, STRATO - SSAM -----------------------------------------------')
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'ssam':'dmi:n34:ssam:strato',
                         'strato':'dmi:n34:strato'}

actors_and_sets_direc = {'dmi':'dmi:n34:ssam:strato',
                         'n34':'dmi:n34:ssam:strato',
                         'ssam':'dmi:n34:ssam:strato',
                         'strato':'dmi:n34:ssam:strato'}

# factores_sp = {'ssam' : {'strato-ssam':-0.658,
#                          'dmi-strato-ssam':0.29 * -0.658,
#                          'n34-dmi-strato-ssam':0.29 * -0.658*0.635},
#                'dmi':{'n34-dmi':0.635}}
factores_sp = None

cen.Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                         actors_and_sets_total, actors_and_sets_direc,
                         save=save, factores_sp=factores_sp,
                         aux_name='Mod_DMI-N34-SSAM_STRATO', alpha=0.10,
                         out_dir=out_dir)

print('DMI, N34, U50 - SSAM --------------------------------------------------')
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'ssam':'dmi:n34:ssam:u50',
                         'u50':'dmi:n34:u50'}

actors_and_sets_direc = {'dmi':'dmi:n34:ssam:u50',
                         'n34':'dmi:n34:ssam:u50',
                         'ssam':'dmi:n34:ssam:u50',
                         'u50':'dmi:n34:ssam:u50'}

# factores_sp = {'ssam' : {'u50-ssam':-0.658,
#                          'dmi-u50-ssam':0.29 * -0.658,
#                          'n34-dmi-u50-ssam':0.29 * -0.658*0.635},
#                'dmi':{'n34-dmi':0.635}}
factores_sp = None

cen.Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                         actors_and_sets_total, actors_and_sets_direc,
                         save=save, factores_sp=factores_sp,
                         aux_name='Mod_DMI-N34-SSAM_U50', alpha=0.10,
                         out_dir=out_dir)

# ---------------------------------------------------------------------------- #
print('DMI, N34 - asam -------------------------------------------------------')
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'asam':'dmi:n34:asam'}

actors_and_sets_direc = {'dmi':'dmi:n34:asam',
                         'n34':'dmi:n34:asam',
                         'asam':'dmi:n34:asam'}

# factores_sp = {'asam' : {'dmi-asam':-0.28,
#                          'n34-dmi-asam':0.635*-0.28},
#                'dmi':{'n34-dmi':0.635}}
factores_sp = None

cen.Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                         actors_and_sets_total, actors_and_sets_direc,
                         save=save, factores_sp=factores_sp,
                         aux_name='Mod_DMI-N34-asam',  alpha=0.10,
                         out_dir=out_dir)

print('DMI, N34, STRATO - asam -----------------------------------------------')
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'asam':'dmi:n34:asam:strato',
                         'strato':'dmi:n34:strato'}

actors_and_sets_direc = {'dmi':'dmi:n34:asam:strato',
                         'n34':'dmi:n34:asam:strato',
                         'asam':'dmi:n34:asam:strato',
                         'strato':'dmi:n34:asam:strato'}

# factores_sp = {'asam' : {'strato-asam':-0.658,
#                          'dmi-strato-asam':0.29 * -0.658,
#                          'n34-dmi-strato-asam':0.29 * -0.658*0.635},
#                'dmi':{'n34-dmi':0.635}}
factores_sp = None

cen.Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                         actors_and_sets_total, actors_and_sets_direc,
                         save=save, factores_sp=factores_sp,
                         aux_name='Mod_DMI-N34-asam_STRATO', alpha=0.10,
                         out_dir=out_dir)

print('DMI, N34, U50 - asam --------------------------------------------------')
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'asam':'dmi:n34:asam:u50',
                         'u50':'dmi:n34:u50'}

actors_and_sets_direc = {'dmi':'dmi:n34:asam:u50',
                         'n34':'dmi:n34:asam:u50',
                         'asam':'dmi:n34:asam:u50',
                         'u50':'dmi:n34:asam:u50'}

# factores_sp = {'asam' : {'u50-asam':-0.658,
#                          'dmi-u50-asam':0.29 * -0.658,
#                          'n34-dmi-u50-asam':0.29 * -0.658*0.635},
#                'dmi':{'n34-dmi':0.635}}
factores_sp = None

cen.Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                         actors_and_sets_total, actors_and_sets_direc,
                         save=save, factores_sp=factores_sp,
                         aux_name='Mod_DMI-N34-asam_U50', alpha=0.10,
                         out_dir=out_dir)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
print('Mod Full ASAM -x- SSAM - STRATO ---------------------------------------')
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'strato':'dmi:n34:strato',
                         'asam':'dmi:n34:strato:asam',
                         'ssam':'dmi:n34:ssam:strato'}

actors_and_sets_direc = {'dmi':'dmi:n34:strato:asam:ssam',
                         'n34':'dmi:n34:strato:asam:ssam',
                         'strato':'dmi:n34:strato:asam:ssam',
                         'asam':'dmi:n34:strato:asam',
                         'ssam':'dmi:n34:ssam:strato'}

# factores_sp = {'strato' : {'dmi-strato':0.29,
#                            'n34-strato':-0.14,
#                            'n34-dmi-strato':0.635*-0.14},
#                'asam' : {'dmi-asam':-0.216,
#                          'dmi-strato-asam':0.29*-0.228,
#                          'n34-dmi-asam':0.635*-0.216,
#                          'n34-asam':-0.439,
#                          'strato-asam':-0.228},
#                'ssam' : {'dmi-strato-ssam': 0.29 * -0.565,
#                          'n34-dmi-strato-ssam': 0.635*0.29*-0.658,
#                          'strato-ssam': -0.658},
#                'dmi': {'n34-dmi': 0.635}}

factores_sp = None

cen.Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                         actors_and_sets_total, actors_and_sets_direc,
                         save=save, factores_sp=factores_sp,
                         aux_name='ModFull_ASAMxSSAM_strato', alpha=0.10,
                         out_dir=out_dir)

print('Mod Full ASAM -x- SSAM - u50 ------------------------------------------')
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'u50':'dmi:n34:u50',
                         'asam':'dmi:n34:u50:asam',
                         'ssam':'dmi:n34:ssam:u50'}

actors_and_sets_direc = {'dmi':'dmi:n34:u50:asam:ssam',
                         'n34':'dmi:n34:u50:asam:ssam',
                         'u50':'dmi:n34:u50:asam:ssam',
                         'asam':'dmi:n34:u50:asam',
                         'ssam':'dmi:n34:ssam:u50'}

# factores_sp = {'u50' : {'dmi-u50':0.29,
#                            'n34-u50':-0.14,
#                            'n34-dmi-u50':0.635*-0.14},
#                'asam' : {'dmi-asam':-0.216,
#                          'dmi-u50-asam':0.29*-0.228,
#                          'n34-dmi-asam':0.635*-0.216,
#                          'n34-asam':-0.439,
#                          'u50-asam':-0.228},
#                'ssam' : {'dmi-u50-ssam': 0.29 * -0.565,
#                          'n34-dmi-u50-ssam': 0.635*0.29*-0.658,
#                          'u50-ssam': -0.658},
#                'dmi': {'n34-dmi': 0.635}}

factores_sp = None

cen.Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                         actors_and_sets_total, actors_and_sets_direc,
                         save=save, factores_sp=factores_sp,
                         aux_name='ModFull_ASAMxSSAM_u50', alpha=0.10,
                         out_dir=out_dir)
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #