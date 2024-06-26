"""
Redes cusales ENSO, IOD, SAM, ASAM, STRATO

pasado en limpio sólo para la red mas grande que incluye tdo
"""
import matplotlib.pyplot as plt

################################################################################
# Seteos generales ----------------------------------------------------------- #
save = True
use_strato_index = True
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
from Scales_Cbars import get_cbars
from cen_funciones import AUX_select_actors, Plot, regre
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

################################################################################
"""
HGT y PP no se usan aún
"""
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
hgt200_anom_or = hgt200_anom_or.sel(
    time=hgt200_anom_or.time.dt.month.isin([8,9,10,11]))
hgt200_anom_or = hgt200_anom_or.sel(
    time=hgt200_anom_or.time.dt.month.isin([10]))

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
    hgt200_anom_or = hgt200_anom_or.sel(time =
                            hgt200_anom_or.time.dt.year.isin(
                                strato_indice['time']))
    strato_indice = strato_indice.sel(time = hgt200_anom_or['time.year'])

# ---------------------------------------------------------------------------- #
# SameDate y normalización --------------------------------------------------- #
# ---------------------------------------------------------------------------- #
hgt200_anom = hgt200_anom_or.sel(time=hgt200_anom_or.time.dt.month.isin([10]))

hgt750_anom = SameDateAs(hgt750_anom_or, hgt200_anom)
dmi = SameDateAs(dmi_or, hgt200_anom)
n34 = SameDateAs(n34_or, hgt200_anom)
sam = SameDateAs(sam_or, hgt200_anom)
asam = SameDateAs(asam_or, hgt200_anom)
ssam = SameDateAs(ssam_or, hgt200_anom)
pp = SameDateAs(pp_or, hgt200_anom)
pp_caja = SameDateAs(pp_caja_or, hgt200_anom)
dmi = dmi / dmi.std()
n34 = n34 / n34.std()
sam = sam / sam.std()
asam = asam / asam.std()
ssam = ssam / ssam.std()
hgt200_anom = hgt200_anom / hgt200_anom.std()
hgt750_anom = hgt750_anom / hgt750_anom.std()
pp_caja = pp_caja / pp_caja.std()
pp = pp / pp.std()

amd200 = (hgt200_anom.sel(lon=slice(210, 270), lat=slice(-80, -50)).
       mean(['lon', 'lat']))
amd200 = amd200 / amd200.std()

amd750 = (hgt750_anom.sel(lon=slice(210, 270), lat=slice(-80, -50)).
       mean(['lon', 'lat']))
amd750 = amd750 / amd750.std()

all_3index = {'sam':sam.values, 'asam':asam.values, 'ssam':ssam.values,
              'amd200':amd200['var'].values, 'amd750':amd750['var'].values}
if use_strato_index:
    all_3index['strato'] = strato_indice['var'].values

################################################################################
def pre_regre_ufunc(x, sets, coef, sig=False, alpha=0.05):
    """
    :param x: target, punto de grilla
    :param sets: str de actores separados por : eg. sets = 'dmi:n34'
    :param coef: 'str' coef del que se quiere beta. eg. 'dmi'
    :return: regre coef (beta)
    """
    pre_serie = {'c': x}

    parts = sets.split(':')
    sets_list = [part for part in parts if len(part) > 0]

    actor_list = {'dmi': dmi.values, 'n34': n34.values, 'ssam': ssam.values,
                  'asam': asam.values, 'strato': strato_indice['var'].values,
                  'sam': sam.values}

    series_select = AUX_select_actors(actor_list, sets_list, pre_serie)
    efecto = regre(series_select, False, coef,
                   filter_significance=sig,
                   alpha=alpha)
    return efecto

def compute_regression(x, sets, coef, sig=False, alpha=0.05):
    coef_dataset = xr.apply_ufunc(
        pre_regre_ufunc, x, sets, coef, sig, alpha,
        input_core_dims=[['time'],[], [], [], []],
        vectorize=True)

    return coef_dataset

hgt200_anom2 = hgt200_anom.sel(lat=slice(-80, 20))

def Compute_CEN_and_Plot(variables, name_variables, maps,
                         actors_and_sets_total, actors_and_sets_direc,
                         save=False, factores_sp=None, aux_name='',
                         sig=False, alpha=0.05):
    if save:
        dpi = 100
    else:
        dpi = 70

    for v, v_name, mapa in zip(variables,
                               name_variables,
                               maps):

        v_cmap = get_cbars(v_name)

        for a in actors_and_sets_total:
            sets_total = actors_and_sets_total[a]
            aux = compute_regression(v['var'], sets_total, coef=a,
                                         sig=sig, alpha=alpha)

            titulo = f"{v_name} - {a} efecto total  {aux_name}"
            name_fig = f"{v_name}_{a}_efecto_TOTAL_{aux_name}"

            Plot(aux, v_cmap, mapa, save, dpi, titulo, name_fig, out_dir)

            try:
                sets_direc = actors_and_sets_direc[a]
                aux = compute_regression(v['var'], sets_direc, coef=a,
                                         sig=sig, alpha=alpha)

                titulo = f"{v_name} - {a} efecto directo  {aux_name}"
                name_fig = f"{v_name}_{a}_efecto_DIRECTO_{aux_name}"

                Plot(aux, v_cmap, mapa, save, dpi, titulo, name_fig, out_dir)

                if factores_sp is not None:
                    sp_cmap = get_cbars('snr2')

                    try:
                        factores_sp_a = factores_sp[a]

                        for f_sp in factores_sp_a.keys():
                            aux_f_sp = factores_sp_a[f_sp]

                            titulo = (f"{v_name} - {a} SP Indirecto via {f_sp} "
                                      f"{aux_name}")
                            name_fig = (f"{v_name}_{a}_SP_indirecto_{f_sp}_"
                                        f"{aux_name}")

                            Plot(aux_f_sp * aux, sp_cmap, mapa, save, dpi,
                                 titulo, name_fig, out_dir)
                    except:
                        pass

            except:
                print('Sin efecto directo')

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# DMI, N34, ASAM
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'asam':'dmi:n34:asam'}

actors_and_sets_direc = {'dmi':'dmi:n34:asam',
                         'n34':'dmi:n34:asam',
                         'asam':'dmi:n34:asam'}

factores_sp = {'asam' : {'dmi-asam':-0.28,
                         'n34-dmi-asam':0.635 * -0.28,
                         'n34-asam':-0.40},
               'dmi':{'n34-dmi':0.635}}

Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                     actors_and_sets_total, actors_and_sets_direc, save=save,
                     factores_sp=None, aux_name='ModP1', sig=True, alpha=0.10)

# ---------------------------------------------------------------------------- #
# DMI, N34, ASAM + strato ---------------------------------------------------- #
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'asam':'dmi:n34:asam:strato',
                         'strato':'dmi:n34:strato'}

actors_and_sets_direc = {'dmi':'dmi:n34:asam:strato',
                         'n34':'dmi:n34:asam:strato',
                         'asam':'dmi:n34:asam:strato',
                         'strato':'dmi:n34:asam:strato'}

factores_sp = {'asam' : {'dmi-asam':-0.216,
                         'n34-dmi-asam':0.635*-0.216,
                         'n34-asam':0.439,
                         'strato-asam':-0.228},
               'dmi':{'n34-dmi':0.635},
               'strato':{'dmi-strato':0.29,
                         'n34-strato':-0.14}}

Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                     actors_and_sets_total, actors_and_sets_direc, save=save,
                     factores_sp=None, aux_name='ModP1_strato',
                     sig=True, alpha=0.10)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# DMI, N34, ssam
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'ssam':'dmi:n34:ssam'}

actors_and_sets_direc = {'dmi':'dmi:n34:ssam',
                         'n34':'dmi:n34:ssam',
                         'ssam':'dmi:n34:ssam'}

factores_sp = {'ssam' : {'dmi-ssam':-0.28,
                         'n34-dmi-ssam':0.635*-0.28},
               'dmi':{'n34-dmi':0.635}}

Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                     actors_and_sets_total, actors_and_sets_direc, save=save,
                     factores_sp=None, aux_name='ModP2',
                     sig=True, alpha=0.10)

# ---------------------------------------------------------------------------- #
# DMI, N34, ssam + strato ---------------------------------------------------- #
actors_and_sets_total = {'dmi':'dmi:n34',
                         'n34': 'n34',
                         'ssam':'dmi:n34:ssam:strato',
                         'strato':'dmi:n34:strato'}

actors_and_sets_direc = {'dmi':'dmi:n34:ssam:strato',
                         'n34':'dmi:n34:ssam:strato',
                         'ssam':'dmi:n34:ssam:strato',
                         'strato':'dmi:n34:ssam:strato'}

factores_sp = {'ssam' : {'strato-ssam':-0.658,
                         'dmi-strato-ssam':0.29 * -0.658,
                         'n34-dmi-strato-ssam':0.29 * -0.658*0.635},
               'dmi':{'n34-dmi':0.635}}

Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                     actors_and_sets_total, actors_and_sets_direc, save=save,
                     factores_sp=None, aux_name='ModP2_strato',
                     sig=True, alpha=0.10)

# ---------------------------------------------------------------------------- #
# Mod Full ASAM -x- SSAM
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

factores_sp = {'strato' : {'dmi-strato':0.29,
                           'n34-strato':-0.14,
                           'n34-dmi-strato':0.635*-0.14},
               'asam' : {'dmi-asam':-0.216,
                         'dmi-strato-asam':0.29*-0.228,
                         'n34-dmi-asam':0.635*-0.216,
                         'n34-asam':-0.439,
                         'strato-asam':-0.228},
               'ssam' : {'dmi-strato-ssam': 0.29 * -0.565,
                         'n34-dmi-strato-ssam': 0.635*0.29*-0.658,
                         'strato-ssam': -0.658},
               'dmi': {'n34-dmi': 0.635}}

Compute_CEN_and_Plot([hgt200_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa'],
                     actors_and_sets_total, actors_and_sets_direc, save=save,
                     factores_sp=factores_sp, aux_name='ModFull_ASAMxSSAM')
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #