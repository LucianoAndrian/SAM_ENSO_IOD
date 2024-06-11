"""
Redes cusales ENSO, IOD, SAM, ASAM, STRATO

pasado en limpio sólo para la red mas grande que incluye tdo
"""
################################################################################
# Seteos generales ----------------------------------------------------------- #
save = True
use_strato_index = True
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

# Regre y CN Effects --------------------------------------------------------- #
def regre(series, intercept, coef=0):
    df = pd.DataFrame(series)
    if intercept:
        X = np.column_stack((np.ones_like(df[df.columns[1]]),
                             df[df.columns[1:]]))
    else:
        X = df[df.columns[1:]].values
    y = df[df.columns[0]]

    coefs = np.linalg.lstsq(X, y, rcond=None)[0]

    coefs_results = {}
    for ec, e in enumerate(series.keys()):
        if intercept and ec == 0:
            e = 'constant'
        if e != df.columns[0]:
            if intercept:
                coefs_results[e] = coefs[ec]
            else:
                coefs_results[e] = coefs[ec-1]

    if isinstance(coef, str):
        return coefs_results[coef]
    else:
        return coefs_results

def AUX_select_actors(actor_list, set_series, serie_to_set):
    serie_to_set2 = serie_to_set.copy()
    for key in set_series:
        serie_to_set2[key] = actor_list[key]
    return serie_to_set2


def CN_Effect_2(actor_list, set_series_directo, set_series_totales,
                variables, set_series_directo_particulares = None):

    """
    Versión de CN_Effect general
    :param actor_list: dict con todos las series que se van a utilizar menos
    la/s serie/s target
    :param set_series_directo: list con los nombres de las series que se deben
     incluir en regre para el efecto directo que se usará si
    set_series_directo_particulares = None.(En muchos casos es igual para todos)
    :param set_series_totales: dict, indicando nombre de la serie y predictandos
    de regre incluyendo la misma serie
    :param set_series_directo_particulares: idem anterior para efectos directos
    que no puedan ser cuantificados con set_series_directo
    :param variables: dict con las series target
    :return: dataframe indicando los efectos totales y directos de cada parent
    hacia las seires target
    """

    for k in variables.keys():
        if k in set_series_directo or k in set_series_totales:
            if set_series_directo_particulares is not None:
                if k in set_series_directo_particulares:
                    print('Error: Variables no puede incluir ser un parent')
                    return
            print('Error: Variables no puede incluir ser un parent')
            return

    result_df = pd.DataFrame(columns=['v_efecto', 'b'])

    for x_name in variables.keys():
        x = variables[x_name]

        try:
            pre_serie = {'c': x['var'].values}
        except:
            pre_serie = {'c': x.values}

        series_directo = AUX_select_actors(actor_list, set_series_directo,
                                           pre_serie)

        series_totales = {}
        actors = []
        for k in set_series_totales.keys():
            actors.append(k)

            series_totales[k] = AUX_select_actors(actor_list,
                                                  set_series_totales[k],
                                                  pre_serie)

            if set_series_directo_particulares is not None:
                series_directas_particulares = {}
                series_directas_particulares[k] = \
                    AUX_select_actors(actor_list,
                                      set_series_directo_particulares[k],
                                      pre_serie)

        if all(actor in series_totales for actor in actors):
            for i in actors:
                # Efecto total i --------------------------------------------- #
                i_total = regre(series_totales[i], True, i)
                result_df = result_df.append({'v_efecto': f"{i}_TOTAL_{x_name}",
                                              'b': i_total}, ignore_index=True)

                # Efecto directo i ------------------------------------------- #
                try:
                    i_directo = regre(
                        series_directas_particulares[i], True, i)
                except:
                    i_directo = regre(series_directo, True, i)

                result_df = result_df.append(
                    {'v_efecto': f"{i}_DIRECTO_{x_name}", 'b': i_directo},
                    ignore_index=True)
        else:
            print('Error: faltan actors en las series')


    print(result_df)
    return result_df

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
actor_list = {'dmi': dmi.values, 'n34': n34.values, 'ssam': ssam.values,
              'asam': asam.values, 'strato':strato_indice['var'].values,
              'sam' : sam.values}

# cen DMI, N34, STRATO, ASAM --> SSAM ---------------------------------------- #
CN_Effect_2(actor_list,
            set_series_directo = ['dmi', 'n34', 'asam', 'strato'],
            set_series_totales = {'dmi':['dmi', 'n34'], 'n34':['n34'],
                                  'strato':['dmi', 'n34', 'strato'],
                                  'asam':['dmi', 'n34', 'strato', 'asam']},
            variables = {'ssam':ssam})

# cen DMI, N34, STRATO --> ASAM ---------------------------------------------- #
CN_Effect_2(actor_list,
            set_series_directo = ['dmi', 'n34', 'strato'],
            set_series_totales = {'dmi':['dmi', 'n34'], 'n34':['n34'],
                                  'strato':['dmi', 'n34', 'strato']},
            variables = {'asam':asam})

# cen DMI, N34 --> STRATO ---------------------------------------------------- #
CN_Effect_2(actor_list,
            set_series_directo = ['dmi', 'n34'],
            set_series_totales = {'dmi':['dmi', 'n34'], 'n34':['n34']},
            variables = {'strato':strato_indice['var']})

# cen N34 --> DMI  ----------------------------------------------------------- #
CN_Effect_2(actor_list,
            set_series_directo = ['n34'],
            set_series_totales = {'n34':['n34']},
            variables = {'dmi':dmi})
################################################################################

CN_Effect_2(actor_list,
            set_series_directo = ['n34', 'dmi', 'strato', 'ssam'],
            set_series_totales = {'ssam':['dmi', 'n34', 'strato', 'ssam']},
            variables = {'sam':sam})




dmi_or = dmi_or.sel(time=dmi_or.time.dt.year.isin(strato_indice.time.values))
n34_or = n34_or.sel(time=n34_or.time.dt.year.isin(strato_indice.time.values))
asam_or = asam_or.sel(time=asam_or.time.dt.year.isin(strato_indice.time.values))
ssam_or = ssam_or.sel(time=ssam_or.time.dt.year.isin(strato_indice.time.values))

dmi = dmi_or.sel(time=dmi_or.time.dt.month.isin([9]))
n34 = n34_or.sel(time=n34_or.time.dt.month.isin([9]))
asam = asam_or.sel(time=asam_or.time.dt.month.isin([10]))
ssam = ssam_or.sel(time=ssam_or.time.dt.month.isin([10]))
ssam0 = ssam_or.sel(time=ssam_or.time.dt.month.isin([7]))
# pp = SameDateAs(pp_or, hgt200_anom)
# pp_caja = SameDateAs(pp_caja_or, hgt200_anom)
dmi = dmi / dmi.std()
n34 = n34 / n34.std()
#sam = sam / sam.std()
asam = asam / asam.std()
ssam = ssam / ssam.std()
ssam0 = ssam0 / ssam0.std()

# cen N34 --> DMI  ----------------------------------------------------------- #
################################################################################
actor_list = {'dmi': dmi.values, 'n34': n34.values, 'ssam': ssam.values,
              'asam': asam.values, 'strato':strato_indice['var'].values,
              'sam': sam.values}
CN_Effect_2(actor_list,
            set_series_directo = ['ssam'],
            set_series_totales = {'ssam0':['ssam0']},
            variables = {'dmi':dmi})