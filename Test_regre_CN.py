"""
Testeos conceptuales de CN a partir de modelos de regresión
"""
################################################################################
save=True
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cn_effect/'
use_strato_index = True
################################################################################
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import xarray as xr
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2
from Scales_Cbars import get_cbars
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
# Funciones ####################################################################
################################################################################
# Obs y preProc -------------------------------------------------------------- #
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

def CN_Effect(actor_list, set_series_directo, set_series_dmi_total,
              set_series_n34_total, set_series_sam_total,
              set_series_dmi_directo=None,
              set_series_n34_directo=None,
              set_series_sam_directo=None):

    result_df = pd.DataFrame(columns=['v_efecto', 'b'])

    #for x, x_name in zip([pp_serie, amd], ['pp_serie', 'amd']):
    for x, x_name in zip([amd], ['amd']):
        print('test0')
        pre_serie = {'c': x['var'].values}
        series_directo = AUX_select_actors(actor_list, set_series_directo,
                                           pre_serie)

        series_dmi_total = AUX_select_actors(
            actor_list, set_series_dmi_total, pre_serie)

        series_n34_total = AUX_select_actors(
            actor_list, set_series_n34_total, pre_serie)

        series_sam_total = AUX_select_actors(
            actor_list, set_series_sam_total, pre_serie)
        print('test1')
        series_directo_particular = {}
        if set_series_dmi_directo is not None:
            series_dmi_directo = AUX_select_actors(
                actor_list, set_series_dmi_directo, pre_serie)
            series_directo_particular['dmi_directo'] = series_dmi_directo

        if set_series_n34_directo is not None:
            series_n34_directo = AUX_select_actors(
                actor_list, set_series_n34_directo, pre_serie)
            series_directo_particular['n34_directo'] = series_n34_directo

        if set_series_sam_directo is not None:
            series_sam_directo = AUX_select_actors(
                actor_list, set_series_sam_directo, pre_serie)
            series_directo_particular['sam_directo'] = series_sam_directo
        print('test2')
        if ('asam' in set_series_directo):
            aux_actors = ['dmi', 'n34', 'asam']
            aux_sam = 'asam'
        elif ('ssam' in set_series_directo):
            aux_actors = ['dmi', 'n34', 'ssam']
            aux_sam = 'ssam'
        elif ('strato' in set_series_directo):
            aux_actors = ['dmi', 'n34', 'strato']
            aux_sam = 'strato'
        else:
            aux_actors = ['dmi', 'n34', 'sam']
            aux_sam = 'sam'
        print('test3')
        series = {'dmi_total': series_dmi_total,
                  'n34_total': series_n34_total,
                  aux_sam + '_total': series_sam_total}
        print(aux_actors)
        print(aux_sam)
        for i in aux_actors:
            print(i)
            # Efecto total i --------------------------------------------------#
            i_total = regre(series[f"{i}_total"], True, i)
            result_df = result_df.append({'v_efecto': f"{i}_TOTAL_{x_name}",
                                          'b': i_total},
                                         ignore_index=True)
            print('test4')
            # Efecto directo i ------------------------------------------------#
            try:
                i_directo = regre(
                    series_directo_particular[f"{i}_directo"], True, i)
            except:
                i_directo = regre(series_directo, True, i)


            result_df = result_df.append({'v_efecto': f"{i}_DIRECTO_{x_name}",
                                          'b': i_directo},
                                         ignore_index=True)
        # -------------------------------------------------------------------- #
    print(result_df)

def pre_regre_ufunc(x, modelo, coef, modo):
    pre_serie = {'c': x}

    # modelo A1
    if modelo.upper() == 'A1':
        actor_list = {'dmi': dmi3.values, 'n34': n343.values,
                      'sam': sam3.values}
        set_series_directo = ['dmi', 'n34', 'sam']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_sam_total = ['dmi', 'n34', 'sam']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_sam_directo = None
        aux_sam = 'sam'

    elif modelo.upper() == 'A1ASAM':
        actor_list = {'dmi': dmi3.values, 'n34': n343.values,
                      'asam': asam3.values}
        set_series_directo = ['dmi', 'n34', 'asam']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_sam_total = ['dmi', 'n34', 'asam']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_sam_directo = None
        aux_sam = 'asam'

    elif modelo.upper() == 'A1ASAMWSAM':
        actor_list = {'dmi': dmi3.values, 'n34': n343.values,
                      'asam': asam3.values, 'sam':sam3.values}
        set_series_directo = ['dmi', 'n34', 'asam', 'sam']
        set_series_dmi_total = ['dmi', 'n34', 'sam']
        set_series_n34_total = ['n34', 'sam']
        set_series_sam_total = ['dmi', 'n34', 'asam', 'sam']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_sam_directo = None
        aux_sam = 'asam'

    elif modelo.upper() == 'A1ASAMWSSAM':
        actor_list = {'dmi': dmi3.values, 'n34': n343.values,
                      'asam': asam3.values, 'ssam': ssam3.values}
        set_series_directo = ['dmi', 'n34', 'asam', 'ssam']
        set_series_dmi_total = ['dmi', 'n34', 'ssam']
        set_series_n34_total = ['n34', 'ssam']
        set_series_sam_total = ['dmi', 'n34', 'asam', 'ssam']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_sam_directo = None
        aux_sam = 'asam'
    elif modelo.upper() == 'A1STRATO':
        actor_list = {'dmi': dmi3.values, 'n34': n343.values,
                      'strato': strato_indice['__xarray_dataarray_variable__'].values}
        set_series_directo = ['dmi', 'n34', 'strato']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_sam_total = ['dmi', 'n34', 'strato']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_sam_directo = None
        aux_sam = 'strato'
    else:
        print('ningun modelo seleccionado')
        return None

    series_directo = AUX_select_actors(actor_list, set_series_directo,
                                       pre_serie)

    series_dmi_total = AUX_select_actors(
        actor_list, set_series_dmi_total, pre_serie)

    series_n34_total = AUX_select_actors(
        actor_list, set_series_n34_total, pre_serie)

    series_sam_total = AUX_select_actors(
        actor_list, set_series_sam_total, pre_serie)

    series_directo_particular = {}
    if set_series_dmi_directo is not None:
        series_dmi_directo = AUX_select_actors(
            actor_list, set_series_dmi_directo, pre_serie)
        series_directo_particular['dmi_directo'] = series_dmi_directo

    if set_series_n34_directo is not None:
        series_n34_directo = AUX_select_actors(
            actor_list, set_series_n34_directo, pre_serie)
        series_directo_particular['n34_directo'] = series_n34_directo

    if set_series_sam_directo is not None:
        series_sam_directo = AUX_select_actors(
            actor_list, set_series_sam_directo, pre_serie)
        series_directo_particular['sam_directo'] = series_sam_directo

    series = {'dmi_total': series_dmi_total,
              'n34_total': series_n34_total,
              aux_sam + '_total': series_sam_total}

    if modo.lower() == 'total':
        efecto = regre(series[f"{coef}_total"], True, coef)

    elif modo.lower() == 'directo':
        try:
            efecto = regre(
                series_directo_particular[f"{coef}_directo"], True, coef)
        except:
            efecto = regre(series_directo, True, coef)

    return efecto



def compute_regression(variable, modelo, coef, modo):

    coef_dataset = xr.apply_ufunc(
        pre_regre_ufunc,
        variable,  modelo, coef, modo,
        input_core_dims=[['time'], [], [], []],
        vectorize=True)
    return coef_dataset

# Plot ----------------------------------------------------------------------- #
def Plot(data, cmap, mapa, save, dpi, titulo, name_fig, out_dir):

    if mapa.lower() == 'sa':
        fig_size = (5, 6)
        extent = [270, 330, -60, 20]
        xticks = np.arange(270, 330, 10)
        yticks = np.arange(-60, 40, 20)
        contour = False

    elif mapa.lower() == 'hs':
        fig_size = (9, 3.5)
        extent = [0, 359, -80, 20]
        xticks = np.arange(0, 360, 30)
        yticks = np.arange(-80, 20, 10)
        contour = True


    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()

    ax.set_extent(extent, crs=crs_latlon)

    im = ax.contourf(data.lon, data.lat, data,
                     levels=[-1, -.8, -.6, -.4, -.2, -.1, 0,
                             .1, .2, .4, .6, .8, 1],
                     transform=crs_latlon, cmap=cmap, extend='both')

    if contour:
        values = ax.contour(data.lon, data.lat, data,
                            levels=[-1, -.8, -.6, -.4, -.2, -.1,
                                    .1, .2, .4, .6, .8, 1],
                            transform=crs_latlon, colors='k', linewidths=1)

    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(xticks, crs=crs_latlon)
    ax.set_yticks(yticks, crs=crs_latlon)
    ax.tick_params(labelsize=7)
    plt.title(titulo, fontsize=10)
    plt.tight_layout()
    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()
    plt.show()

################################################################################
# Data
data = OpenObsDataSet(name='pp_pgcc_v2020_1891-2023_1', sa=True, dir=dir_pp)
data = data.rename({'precip':'var'})
data_40_20 = data.sel(time=slice('1940-01-16', '2020-12-16'))
del data

data_40_20 = Weights(data_40_20)
data_40_20 = data_40_20.sel(lat=slice(20, -80)) # HS
data_40_20 = data_40_20.rolling(time=3, center=True).mean()
aux = data_40_20.sel(time=data_40_20.time.dt.month.isin([8,9,10,11]))
pp = Detrend(aux, 'time')

pp_serie = pp.sel(lat=slice(-30,-40), lon=slice(295,310)).mean(['lon', 'lat'])
pp_serie['var'][-1]=0 # aca nse que pasa...

# ---------------------------------------------------------------------------- #
hgt = xr.open_dataset(hgt_dir + 'ERA5_HGT200_40-20.nc')
hgt = hgt.rename({'longitude': 'lon', 'latitude': 'lat', 'z': 'var'})
hgt = hgt.interp(lon=np.arange(0,360,2), lat=np.arange(-90, 90, 2))

hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
hgt_anom = hgt.groupby('time.month') - \
           hgt_clim.groupby('time.month').mean('time')

use_sam=True
if use_sam:
    weights = np.sqrt(np.abs(np.cos(np.radians(hgt_anom.lat))))
    hgt_anom = hgt_anom * weights

hgt_anom = hgt_anom.rolling(time=3, center=True).mean()
hgt_anom = hgt_anom.sel(time=slice('1940-02-01', '2020-11-01'))
hgt_anom = hgt_anom.sel(time=hgt_anom.time.dt.month.isin([8,9,10,11]))
hgt_anom = hgt_anom.sel(time=hgt_anom.time.dt.month.isin([10]))
# ---------------------------------------------------------------------------- #
# indices
# ---------------------------------------------------------------------------- #
sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam = sam.rolling(time=3, center=True).mean()

asam = xr.open_dataset(sam_dir + 'asam_700.nc')['mean_estimate']
asam = asam.rolling(time=3, center=True).mean()

ssam = xr.open_dataset(sam_dir + 'ssam_700.nc')['mean_estimate']
ssam = ssam.rolling(time=3, center=True).mean()

dmi = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/"
                      "sst.mnmean.nc")
aux = aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34 = Nino34CPC(aux, start=1920, end=2020)[0]

if use_strato_index:
    strato_indice = xr.open_dataset('strato_index.nc').rename({'year':'time'})
    hgt_anom = hgt_anom.sel(time =
                            hgt_anom.time.dt.year.isin(strato_indice['time']))
    strato_indice = strato_indice.sel(time = hgt_anom['time.year'])
# ---------------------------------------------------------------------------- #
# SameDate y normalización --------------------------------------------------- #
# ---------------------------------------------------------------------------- #
dmi2 = SameDateAs(dmi, hgt_anom)
n342 = SameDateAs(n34, hgt_anom)
sam2 = SameDateAs(sam, hgt_anom)
asam2 = SameDateAs(asam, hgt_anom)
ssam2 = SameDateAs(ssam, hgt_anom)
pp = SameDateAs(pp, hgt_anom)
pp_serie = SameDateAs(pp_serie, hgt_anom)

#sam3 = sam2
#c = c/c.std()
dmi3 = dmi2/dmi2.std()
n343 = n342/n342.std()
sam3 = sam2/sam2.std()
asam3 = asam2/asam2.std()
ssam3 = ssam2/ssam2.std()

amd = hgt_anom.sel(lon=slice(210,270), lat=slice(-80,-50)).mean(['lon', 'lat'])
amd = amd/amd.std()
hgt_anom = hgt_anom/hgt_anom.std()

pp_serie = pp_serie/pp_serie.std()
pp = pp/pp.std()
################################################################################
print('#######################################################################')
actor_list = {'dmi':dmi3.values, 'n34':n343.values, 'sam':sam3.values}
print('Modelo A: N34->IOD, N34->SAM (todos a C)')
print('A1 IOD->SAM -----------------------------------------------------------')
CN_Effect(actor_list,  set_series_directo = ['dmi', 'n34', 'sam'],
          set_series_dmi_total=['dmi', 'n34'],
          set_series_n34_total=['n34'],
          set_series_sam_total=['dmi', 'n34', 'sam'],
          set_series_n34_directo=None,
          set_series_dmi_directo=None,
          set_series_sam_directo=None)

print('Modelo A con A-SAM: N34->IOD, N34->A-SAM, IOD->A-SAM (todos a C)')
print('A1 IOD->SAM -----------------------------------------------------------')
actor_list = {'dmi':dmi3.values, 'n34':n343.values, 'asam':asam3.values}
CN_Effect(actor_list,  set_series_directo = ['dmi', 'n34', 'asam'],
          set_series_dmi_total=['dmi', 'n34'],
          set_series_n34_total=['n34'],
          set_series_sam_total=['dmi', 'n34', 'asam'],
          set_series_n34_directo=None,
          set_series_dmi_directo=None,
          set_series_sam_directo=None)

actor_list = {'dmi':dmi3.values, 'n34':n343.values, 'ssam':ssam3.values}
CN_Effect(actor_list,  set_series_directo = ['dmi', 'n34', 'ssam'],
          set_series_dmi_total=['dmi', 'n34'],
          set_series_n34_total=['n34'],
          set_series_sam_total=['dmi', 'n34', 'ssam'],
          set_series_n34_directo=None,
          set_series_dmi_directo=None,
          set_series_sam_directo=None)

actor_list = {'dmi':dmi3.values, 'n34':n343.values, 'asam':asam3.values,'ssam':ssam3.values}
CN_Effect(actor_list,  set_series_directo = ['dmi', 'n34', 'asam', 'ssam'],
          set_series_dmi_total=['dmi', 'n34', 'ssam'],
          set_series_n34_total=['n34', 'ssam'],
          set_series_sam_total=['dmi', 'n34', 'asam', 'ssam'],
          set_series_n34_directo=None,
          set_series_dmi_directo=None,
          set_series_sam_directo=None)

if use_strato_index:
    actor_list = {'dmi': dmi3.values, 'n34': n343.values,
                  'strato': strato_indice['__xarray_dataarray_variable__'].values}
    CN_Effect(actor_list, set_series_directo=['dmi', 'n34', 'strato'],
              set_series_dmi_total=['dmi', 'n34'],
              set_series_n34_total=['n34'],
              set_series_sam_total=['dmi', 'n34', 'strato'],
              set_series_n34_directo=None,
              set_series_dmi_directo=None,
              set_series_sam_directo=None)


# ---------------------------------------------------------------------------- #
print('#######################################################################')
print('Mapas...')
print('#######################################################################')
# Lo mismo en mapas
################################################################################

hgt_anom2 = hgt_anom.sel(lat=slice(-80,20))
hgt_cmap = get_cbars('hgt200')
pp_cmap = get_cbars('pp')

actors_target = {'A1':['dmi', 'n34', 'sam'], 'A1ASAM':['dmi', 'n34', 'asam'],
                 'A1ASAMwSAM':['dmi', 'n34', 'asam'],
                 'A1ASAMwSSAM':['dmi', 'n34', 'asam'],
                 'A1STRATO':['dmi', 'n34', 'strato']}

#for v, v_name, mapa in zip([hgt_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa']):
for v, v_name, mapa in zip([hgt_anom2], ['hgt200'], ['hs']):
    v_cmap = get_cbars(v_name)
    for modelo in ['A1', 'A1ASAM']: #'A1ASAMwSAM', 'A1ASAMwSSAM']:
    #for modelo in ['A1STRATO']:
        for actor in actors_target[modelo]:

            for modo in ['total', 'directo']:

                name_fig = f"{v_name}_Mod{modelo}_Efecto_{modo}_{actor}"
                titulo = f"{v_name}_Mod{modelo} Efecto {modo} {actor}"

                efecto = compute_regression(v['var'], modelo, actor, modo)

                Plot(efecto, v_cmap, mapa, save, dpi, titulo, name_fig, out_dir)
print('#######################################################################')
print('Done')
print('#######################################################################')
# ################################################################################
# print('A2 IOD<-SAM -----------------------------------------------------------')
# CN_Effect(actor_list,  set_series_directo = ['dmi', 'n34', 'sam'],
#           set_series_dmi_total=['dmi', 'n34', 'sam'],
#           set_series_n34_total=['n34'],
#           set_series_sam_total=['n34', 'sam'],
#           set_series_n34_directo=None,
#           set_series_dmi_directo=None,
#           set_series_sam_directo=None)
# # ---------------------------------------------------------------------------- #
# print('A3 IOD<->SAM ----------------------------------------------------------')
# CN_Effect(actor_list,  set_series_directo = ['dmi', 'n34', 'sam'],
#           set_series_dmi_total=['dmi', 'n34', 'sam'],
#           set_series_n34_total=['dmi', 'n34', 'sam'],
#           set_series_sam_total=['n34', 'sam'],
#           set_series_n34_directo=None,
#           set_series_dmi_directo=None,
#           set_series_sam_directo=None)
#
# print('#######################################################################')
# print('Modelo B: N34<-IOD, N34->SAM (todos a C)')
# print('B1 IOD->SAM -----------------------------------------------------------')
# CN_Effect(actor_list,  set_series_directo = ['dmi', 'n34', 'sam'],
#           set_series_dmi_total=['dmi'],
#           set_series_n34_total=['dmi', 'n34'],
#           set_series_sam_total=['dmi', 'n34', 'sam'],
#           set_series_n34_directo=None,
#           set_series_dmi_directo=None,
#           set_series_sam_directo=None)
#
# # ---------------------------------------------------------------------------- #
# print('B2 IOD<-SAM -----------------------------------------------------------')
# print("Can't determine causal effects for cyclic models")
# print("Se crea un ciclo IOD->N34->SAM->IOD que "
#       "no permite detectar efectos causales")
#
# # ---------------------------------------------------------------------------- #
# print('B3 IOD<->SAM ----------------------------------------------------------')
# # No se pueden evaluar los efectos totales de DMI y N34
# CN_Effect(actor_list,  set_series_directo = ['dmi', 'n34', 'sam'],
#           set_series_dmi_total=['dmi', 'n34', 'sam'],
#           set_series_n34_total=['dmi', 'n34', 'sam'],
#           set_series_sam_total=['dmi', 'n34', 'sam'],
#           set_series_n34_directo=None,
#           set_series_dmi_directo=None,
#           set_series_sam_directo=None)
#
# print('#######################################################################')
# print('Modelo C: SAM -x- IOD (todos a C))')
# print('C1 IOD->N34 -----------------------------------------------------------')
# CN_Effect(actor_list,  set_series_directo = ['dmi', 'n34', 'sam'],
#           set_series_dmi_total=['dmi'],
#           set_series_n34_total=['dmi', 'n34'],
#           set_series_sam_total=['n34', 'sam'],
#           set_series_n34_directo=['dmi', 'n34'],
#           set_series_dmi_directo=None,
#           set_series_sam_directo=['n34', 'sam'])
#
# print('C2 IOD<-N34 -----------------------------------------------------------')
# CN_Effect(actor_list,  set_series_directo = ['dmi', 'n34', 'sam'],
#           set_series_dmi_total=['dmi', 'n34'],
#           set_series_n34_total=['n34'],
#           set_series_sam_total=['dmi', 'n34', 'sam'],
#           set_series_n34_directo=['dmi', 'n34'],
#           set_series_dmi_directo=None,
#           set_series_sam_directo=['n34', 'sam'])
#
# print('C2 IOD<->N34 ----------------------------------------------------------')
# CN_Effect(actor_list,  set_series_directo = ['dmi', 'n34', 'sam'],
#           set_series_dmi_total=['dmi', 'n34'],
#           set_series_n34_total=['dmi', 'n34'],
#           set_series_sam_total=['n34', 'sam'],
#           set_series_n34_directo=['dmi', 'n34'],
#           set_series_dmi_directo=['dmi', 'n34'],
#           set_series_sam_directo=None)



#### ESTO ERA DE LA FUNCION

#
# # modelo A2
# elif modelo.upper() == 'A2':
# set_series_directo = ['dmi', 'n34', 'sam']
# set_series_dmi_total = ['dmi', 'n34', 'sam']
# set_series_n34_total = ['n34']
# set_series_sam_total = ['n34', 'sam']
# set_series_n34_directo = None
# set_series_dmi_directo = None
# set_series_sam_directo = None
#
# # modelo A3
# elif modelo.upper() == 'A3':
# set_series_directo = ['dmi', 'n34', 'sam']
# set_series_dmi_total = ['dmi', 'n34', 'sam']
# set_series_n34_total = ['dmi', 'n34', 'sam']
# set_series_sam_total = ['n34', 'sam']
# set_series_n34_directo = None
# set_series_dmi_directo = None
# set_series_sam_directo = None
#
# # modelo B1
# elif modelo.upper() == 'B1':
# set_series_directo = ['dmi', 'n34', 'sam']
# set_series_dmi_total = ['dmi']
# set_series_n34_total = ['dmi', 'n34']
# set_series_sam_total = ['dmi', 'n34', 'sam']
# set_series_n34_directo = None
# set_series_dmi_directo = None
# set_series_sam_directo = None
#
# # Modelo B2
# elif modelo.upper() == 'B2':
# print('Modelo B2 no permite establecer efectos causales')
# return
#
# # Modelo B3
# elif modelo.upper() == 'B3':
# set_series_directo = ['dmi', 'n34', 'sam']
# set_series_dmi_total = ['dmi', 'n34', 'sam']
# set_series_n34_total = ['dmi', 'n34', 'sam']
# set_series_sam_total = ['dmi', 'n34', 'sam']
# set_series_n34_directo = None
# set_series_dmi_directo = None
# set_series_sam_directo = None
#
# # Modelo C1
# if modelo.upper() == 'C1':
#     set_series_directo = ['dmi', 'n34', 'sam']
#     set_series_dmi_total = ['dmi']
#     set_series_n34_total = ['dmi', 'n34']
#     set_series_sam_total = ['n34', 'sam']
#     set_series_n34_directo = ['dmi', 'n34']
#     set_series_dmi_directo = None
#     set_series_sam_directo = ['n34', 'sam']
#
# # Modelo C2
# elif modelo.upper() == 'C2':
#     set_series_directo = ['dmi', 'n34', 'sam']
#     set_series_dmi_total = ['dmi', 'n34']
#     set_series_n34_total = ['n34']
#     set_series_sam_total = ['dmi', 'n34', 'sam']
#     set_series_n34_directo = ['dmi', 'n34']
#     set_series_dmi_directo = None
#     set_series_sam_directo = ['n34', 'sam']
#
# # Modelo C3
# elif modelo.upper() == 'C3':
#     set_series_directo = ['dmi', 'n34', 'sam']
#     set_series_dmi_total = ['dmi', 'n34']
#     set_series_n34_total = ['dmi', 'n34']
#     set_series_sam_total = ['n34', 'sam']
#     set_series_n34_directo = ['dmi', 'n34']
#     set_series_dmi_directo = ['dmi', 'n34']
#     set_series_sam_directo = None
