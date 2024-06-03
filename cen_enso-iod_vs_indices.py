"""
Redes cusales ENSO-IOD-SAM/ASAM/STRATO_indice

Hay modelos de causalidad ya definidos dentro de la funcion pre_regre_ufunc
que cumple la funcion de preparar setear tdo para usar la funcio regre
junto con xarray_ufunc para aplicar los modelos en tdos los puntos de grilla

En caso de querer analizar otros deben agregarse a mano en esta funcion.
"""
################################################################################
# Seteos generales ----------------------------------------------------------- #
save = True
use_strato_index = True
plot_mapas = False
plot_corr_scatter = False
create_df = False
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
from scipy.stats import pearsonr
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


def CN_Effect(actor_list, set_series_directo, set_series_dmi_total,
              set_series_n34_total, set_series_3index_total,
              set_series_dmi_directo=None,
              set_series_n34_directo=None,
              set_series_3index_directo=None,
              name = 'cn_result', variables={}):

    result_df = pd.DataFrame(columns=['v_efecto', 'b'])

    for x_name in variables.keys():
        x = variables[x_name]

        try:
            pre_serie = {'c': x['var'].values}
        except:
            pre_serie = {'c': x.values}

        series_directo = AUX_select_actors(actor_list, set_series_directo,
                                           pre_serie)

        if ('asam' in set_series_directo):
            aux_actors = ['dmi', 'n34', 'asam']
            aux_3index = 'asam'
        elif ('ssam' in set_series_directo):
            aux_actors = ['dmi', 'n34', 'ssam']
            aux_3index = 'ssam'
        elif ('strato' in set_series_directo):
            aux_actors = ['dmi', 'n34', 'strato']
            aux_3index = 'strato'
        elif ('sam' in set_series_directo):
            aux_actors = ['dmi', 'n34', 'sam']
            aux_3index = 'sam'
        else:
            aux_actors = ['dmi', 'n34']
            aux_3index = None

        series = {}
        if set_series_dmi_total is not None:
            series_dmi_total = AUX_select_actors(
                actor_list, set_series_dmi_total, pre_serie)
            series['dmi_total'] = series_dmi_total

        if set_series_n34_total is not None:
            series_n34_total = AUX_select_actors(
                actor_list, set_series_n34_total, pre_serie)
            series['n34_total'] = series_n34_total

        if set_series_3index_total is not None:
            series_3index_total = AUX_select_actors(
            actor_list, set_series_3index_total, pre_serie)
            series[aux_3index + '_total'] = series_3index_total

        series_directo_particular = {}
        if set_series_dmi_directo is not None:
            series_dmi_directo = AUX_select_actors(
                actor_list, set_series_dmi_directo, pre_serie)
            series_directo_particular['dmi_directo'] = series_dmi_directo

        if set_series_n34_directo is not None:
            series_n34_directo = AUX_select_actors(
                actor_list, set_series_n34_directo, pre_serie)
            series_directo_particular['n34_directo'] = series_n34_directo

        if set_series_3index_directo is not None:
            series_3index_directo = AUX_select_actors(
                actor_list, set_series_3index_directo, pre_serie)
            series_directo_particular['3index_directo'] = series_3index_directo

        for i in aux_actors:
            # Efecto total i --------------------------------------------------#
            i_total = regre(series[f"{i}_total"], True, i)
            result_df = result_df.append({'v_efecto': f"{i}_TOTAL_{x_name}",
                                          'b': i_total},
                                         ignore_index=True)

            # Efecto directo i ------------------------------------------------#
            try:
                i_directo = regre(
                    series_directo_particular[f"{i}_directo"], True, i)
            except:
                i_directo = regre(series_directo, True, i)


            result_df = result_df.append({'v_efecto': f"{i}_DIRECTO_{x_name}",
                                          'b': i_directo}, ignore_index=True)
        # -------------------------------------------------------------------- #

    result_df.to_csv(f"{out_dir}{name}.txt", sep='\t', index=False)
    print(result_df)

def pre_regre_ufunc(x, modelo, coef, modo):
    pre_serie = {'c': x}

    # modelo A1
    if modelo.upper() == 'A1':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'sam': sam.values}
        set_series_directo = ['dmi', 'n34', 'sam']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_3index_total = ['dmi', 'n34', 'sam']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'sam'

    elif modelo.upper() == 'A1ASAM':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'asam': asam.values}
        set_series_directo = ['dmi', 'n34', 'asam']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_3index_total = ['dmi', 'n34', 'asam']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'asam'

    elif modelo.upper() == 'AMD200':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'amd200': amd200['var'].values}
        set_series_directo = ['dmi', 'n34', 'amd200']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_3index_total = ['dmi', 'n34', 'amd200']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'amd200'

    elif modelo.upper() == 'AMD750':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'amd750': amd750['var'].values}
        set_series_directo = ['dmi', 'n34', 'amd750']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_3index_total = ['dmi', 'n34', 'amd750']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'amd750'

    elif modelo.upper() == 'A1SSAM':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'ssam': ssam.values}
        set_series_directo = ['dmi', 'n34', 'ssam']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_3index_total = ['dmi', 'n34', 'ssam']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'ssam'

    elif modelo.upper() == 'A1ASAMWSSAM':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'asam': asam.values, 'ssam': ssam.values}
        set_series_directo = ['dmi', 'n34', 'asam', 'ssam']
        set_series_dmi_total = ['dmi', 'n34', 'ssam']
        set_series_n34_total = ['n34', 'ssam']
        set_series_3index_total = ['dmi', 'n34', 'asam', 'ssam']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'asam'

    elif modelo.upper() == 'A1STRATO':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'strato': strato_indice['var'].values}
        set_series_directo = ['dmi', 'n34', 'strato']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_3index_total = ['dmi', 'n34', 'strato']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'strato'

    elif modelo.upper() == 'A_SIMPLE':
        actor_list = {'dmi': dmi.values, 'n34': n34.values}
        set_series_directo=['dmi', 'n34']
        set_series_dmi_total=['dmi', 'n34']
        set_series_n34_total=['n34']
        set_series_3index_total=None
        set_series_n34_directo=None
        set_series_dmi_directo=None
        set_series_3index_directo=None
        aux_3index = None

    else:
        print('ningun modelo seleccionado')
        return None

    series_directo = AUX_select_actors(actor_list, set_series_directo,
                                       pre_serie)

    series = {}
    if set_series_dmi_total is not None:
        series_dmi_total = AUX_select_actors(
            actor_list, set_series_dmi_total, pre_serie)
        series['dmi_total'] = series_dmi_total

    if set_series_n34_total is not None:
        series_n34_total = AUX_select_actors(
            actor_list, set_series_n34_total, pre_serie)
        series['n34_total'] = series_n34_total

    if set_series_3index_total is not None:
        series_3index_total = AUX_select_actors(
            actor_list, set_series_3index_total, pre_serie)
        series[aux_3index + '_total'] = series_3index_total

    series_directo_particular = {}
    if set_series_dmi_directo is not None:
        series_dmi_directo = AUX_select_actors(
            actor_list, set_series_dmi_directo, pre_serie)
        series_directo_particular['dmi_directo'] = series_dmi_directo

    if set_series_n34_directo is not None:
        series_n34_directo = AUX_select_actors(
            actor_list, set_series_n34_directo, pre_serie)
        series_directo_particular['n34_directo'] = series_n34_directo

    if set_series_3index_directo is not None:
        series_3index_directo = AUX_select_actors(
            actor_list, set_series_3index_directo, pre_serie)
        series_directo_particular['3index_directo'] = series_3index_directo

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

def ScatterPlot(xserie, yserie, xlabel, ylabel, title, name_fig, dpi, save):
    fig, ax = plt.subplots(dpi=dpi)

    plt.scatter(x=xserie, y=yserie, marker='o', s=20,
                edgecolor='k', color='dimgray', alpha=1)

    plt.ylim((-5, 5))
    plt.xlim((-5, 5))
    plt.axhspan(-.5, .5, alpha=0.2, color='black', zorder=0)
    plt.axvspan(-.5, .5, alpha=0.2, color='black', zorder=0)
    ax.grid()
    fig.set_size_inches(6, 6)
    plt.xlabel(xlabel, size=15)
    plt.ylabel(ylabel, size=15)
    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
    else:
        plt.show()

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
if plot_corr_scatter:
    # Correlación ------------------------------------------------------------ #
    aux_r = np.round(pearsonr(dmi, n34), 3)
    ScatterPlot(dmi, n34, 'DMI', 'N34',
                f"DMI vs N34 - {per} - r = {aux_r[0]} pvalue = {aux_r[1]}",
                f"DMI_N34_{per}", dpi, save)

    # ------------------------------------------------------------------------ #
    aux_r = np.round(pearsonr(dmi, sam), 3)
    ScatterPlot(dmi, asam, 'DMI', 'SAM',
                f"DMI vs SAM - {per} - r = {aux_r[0]} pvalue = {aux_r[1]}",
                f"DMI_SAM_{per}", dpi, save)

    aux_r = np.round(pearsonr(dmi, asam), 3)
    ScatterPlot(dmi, asam, 'DMI', 'ASAM',
                f"DMI vs ASAM - {per} - r = {aux_r[0]} pvalue = {aux_r[1]}",
                f"DMI_SAM_{per}", dpi, save)

    aux_r = np.round(pearsonr(dmi, ssam), 3)
    ScatterPlot(dmi, asam, 'DMI', 'SSAM',
                f"DMI vs SSAM - {per} - r = {aux_r[0]} pvalue = {aux_r[1]}",
                f"DMI_SSAM_{per}", dpi, save)

    aux_r = np.round(pearsonr(dmi, amd200['var'].values), 3)
    ScatterPlot(dmi, amd200['var'].values, 'DMI', 'AMD200',
                f"DMI vs AMD200 - {per} - r = {aux_r[0]} pvalue = {aux_r[1]}",
                f"DMI_200_{per}", dpi, save)

    aux_r = np.round(pearsonr(dmi, amd750['var'].values), 3)
    ScatterPlot(dmi, amd750['var'].values, 'DMI', 'AMD750',
                f"DMI vs AMD750 - {per} - r = {aux_r[0]} pvalue = {aux_r[1]}",
                f"DMI_750_{per}", dpi, save)

    # ------------------------------------------------------------------------ #
    aux_r = np.round(pearsonr(n34, sam), 3)
    ScatterPlot(n34, asam, 'n34', 'SAM',
                f"N34 vs SAM - {per} - r = {aux_r[0]} pvalue = {aux_r[1]}",
                f"N34_SAM_{per}", dpi, save)

    aux_r = np.round(pearsonr(n34, asam), 3)
    ScatterPlot(n34, asam, 'n34', 'ASAM',
                f"N34 vs ASAM - {per} - r = {aux_r[0]} pvalue = {aux_r[1]}",
                f"N34_SAM_{per}", dpi, save)

    aux_r = np.round(pearsonr(n34, ssam), 3)
    ScatterPlot(n34, asam, 'n34', 'SSAM',
                f"N34 vs SSAM - {per} - r = {aux_r[0]} pvalue = {aux_r[1]}",
                f"N34_sSAM_{per}", dpi, save)

    aux_r = np.round(pearsonr(n34, amd200['var'].values), 3)
    ScatterPlot(n34, amd200['var'].values, 'N34', 'AMD200',
                f"N34 vs AMD200 - {per} - r = {aux_r[0]} pvalue = {aux_r[1]}",
                f"N34_AMD200_{per}", dpi, save)

    aux_r = np.round(pearsonr(n34, amd750['var'].values), 3)
    ScatterPlot(n34, amd750['var'].values, 'N34', 'AMD750',
                f"N34 vs AMD750 - {per} - r = {aux_r[0]} pvalue = {aux_r[1]}",
                f"N34_AMD750_{per}", dpi, save)
    plt.close('all')
    # ------------------------------------------------------------------------ #

    if use_strato_index:
        aux_r = np.round(pearsonr(dmi, strato_indice['var'].values), 3)
        ScatterPlot(dmi, strato_indice['var'].values, 'DMI', 'STRATO',
                    f"DMI vs STRATO - {per} - r = {aux_r[0]} "
                    f"pvalue = {aux_r[1]}", f"DMI_STRATO_{per}", dpi, save)

        aux_r = np.round(pearsonr(n34, strato_indice['var'].values), 3)
        ScatterPlot(n34, strato_indice['var'].values, 'N34', 'STRATO',
                    f"N34 vs STRATO - {per} - r = {aux_r[0]} "
                    f"pvalue = {aux_r[1]}", f"N34_STRATO_{per}", dpi, save)

        aux_r = np.round(pearsonr(sam, strato_indice['var'].values), 3)
        ScatterPlot(sam, strato_indice['var'].values, 'SAM', 'STRATO',
                    f"SAM vs STRATO - {per} - r = {aux_r[0]} "
                    f"pvalue = {aux_r[1]}", f"SAM_STRATO_{per}", dpi, save)

        aux_r = np.round(pearsonr(ssam, strato_indice['var'].values), 3)
        ScatterPlot(ssam, strato_indice['var'].values, 'SSAM', 'STRATO',
                    f"SSAM vs STRATO - {per} - r = {aux_r[0]} "
                    f"pvalue = {aux_r[1]}", f"SSAM_STRATO_{per}", dpi, save)

        aux_r = np.round(pearsonr(asam, strato_indice['var'].values), 3)
        ScatterPlot(asam, strato_indice['var'].values, 'ASAM', 'STRATO',
                    f"ASAM vs STRATO - {per} - r = {aux_r[0]}"
                    f" pvalue = {aux_r[1]}", f"ASAM_STRATO_{per}", dpi, save)

        aux_r = np.round(pearsonr(strato_indice['var'].values,
                                  amd200['var'].values), 3)
        ScatterPlot(strato_indice['var'].values, amd200['var'].values,
                    'STRATO', 'AMD200',
                    f"STRATO vs AMD200 - {per} - r = {aux_r[0]}"
                    f" pvalue = {aux_r[1]}", f"STRATO_AMD200_{per}", dpi, save)

        aux_r = np.round(pearsonr(strato_indice['var'].values,
                                  amd750['var'].values), 3)
        ScatterPlot(strato_indice['var'].values, amd750['var'].values,
                    'STRATO', 'AMD750',
                    f"STRATO vs AMD750 - {per} - r = {aux_r[0]} "
                    f"pvalue = {aux_r[1]}", f"STRATO_AMD750_{per}", dpi, save)

        plt.close('all')

# ---------------------------------------------------------------------------- #
################################################################################
# CEN ------------------------------------------------------------------------ #
if create_df:
    variables = {'pp_serie':pp_caja, 'amd200':amd200, 'adm750':amd750}
    print('###################################################################')
    print('Modelo A_SIMPLE: N34->IOD (todos a C)')
    actor_list = {'dmi': dmi.values, 'n34': n34.values}
    CN_Effect(actor_list, set_series_directo=['dmi', 'n34'],
              set_series_dmi_total=['dmi', 'n34'],
              set_series_n34_total=['n34'],
              set_series_3index_total=None,
              set_series_n34_directo=None,
              set_series_dmi_directo=None,
              set_series_3index_directo=None,
              name=f"A_SIMPLE_{per}",
              variables=variables)

    variables = {'ssam': ssam, 'asam': asam}
    actor_list = {'dmi': dmi.values, 'n34': n34.values}
    CN_Effect(actor_list, set_series_directo=['dmi', 'n34'],
              set_series_dmi_total=['dmi', 'n34'],
              set_series_n34_total=['n34'],
              set_series_3index_total=None,
              set_series_n34_directo=None,
              set_series_dmi_directo=None,
              set_series_3index_directo=None,
              name=f"A_SIMPLE_vs_sams{per}",
              variables=variables)

    variables = {'strato': strato_indice}
    CN_Effect(actor_list, set_series_directo=['dmi', 'n34'],
              set_series_dmi_total=['dmi', 'n34'],
              set_series_n34_total=['n34'],
              set_series_3index_total=None,
              set_series_n34_directo=None,
              set_series_dmi_directo=None,
              set_series_3index_directo=None,
              name=f"A_SIMPLE_vs_STRATO_{per}",
              variables=variables)

    # print('Modelo A: N34->IOD, N34->SAM (todos a C)')
    # print('A1 IOD->SAM -------------------------------------------------------')
    # actor_list = {'dmi': dmi.values, 'n34': n34.values, 'sam': sam.values}
    # CN_Effect(actor_list, set_series_directo=['dmi', 'n34', 'sam'],
    #           set_series_dmi_total=['dmi', 'n34'],
    #           set_series_n34_total=['n34'],
    #           set_series_3index_total=['dmi', 'n34', 'sam'],
    #           set_series_n34_directo=None,
    #           set_series_dmi_directo=None,
    #           set_series_3index_directo=None,
    #           name=f"A1_{per}",
    #           variables=variables)

    # print('Modelo A con A-SAM: N34->IOD, N34->A-SAM, IOD->A-SAM (todos a C)')
    # actor_list = {'dmi': dmi.values, 'n34': n34.values, 'asam': asam.values}
    # CN_Effect(actor_list, set_series_directo=['dmi', 'n34', 'asam'],
    #           set_series_dmi_total=['dmi', 'n34'],
    #           set_series_n34_total=['n34'],
    #           set_series_3index_total=['dmi', 'n34', 'asam'],
    #           set_series_n34_directo=None,
    #           set_series_dmi_directo=None,
    #           set_series_3index_directo=None,
    #           name=f"A1wA-SAM_{per}",
    #           variables=variables)

    # print('Modelo A con S-SAM: N34->IOD, N34->S-SAM, IOD->S-SAM (todos a C)')
    # actor_list = {'dmi': dmi.values, 'n34': n34.values, 'ssam': ssam.values}
    # CN_Effect(actor_list, set_series_directo=['dmi', 'n34', 'ssam'],
    #           set_series_dmi_total=['dmi', 'n34'],
    #           set_series_n34_total=['n34'],
    #           set_series_3index_total=['dmi', 'n34', 'ssam'],
    #           set_series_n34_directo=None,
    #           set_series_dmi_directo=None,
    #           set_series_3index_directo=None,
    #           name=f"A1wS-SAM_{per}",
    #           variables=variables)

    print('Modelo A con A-SAM y S-SAM: N34->IOD, N34->A-SAM, IOD->A-SAM y'
          ' S-SAM independdiente (todos a C)')
    actor_list = {'dmi': dmi.values, 'n34': n34.values, 'asam': asam.values,
                  'ssam': ssam.values}
    CN_Effect(actor_list, set_series_directo=['dmi', 'n34', 'asam', 'ssam'],
              set_series_dmi_total=['dmi', 'n34', 'ssam'],
              set_series_n34_total=['n34', 'ssam'],
              set_series_3index_total=['dmi', 'n34', 'asam', 'ssam'],
              set_series_n34_directo=None,
              set_series_dmi_directo=None,
              set_series_3index_directo=None,
              name=f"A1wASAM_SSAM_{per}",
              variables=variables)

    if use_strato_index:
        print('Modelo A con Strato index: N34->IOD, N34->strato_index, '
              'IOD->strato_index independdiente (todos a C)')
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'strato': strato_indice['var'].values}
        CN_Effect(actor_list, set_series_directo=['dmi', 'n34', 'strato'],
                  set_series_dmi_total=['dmi', 'n34'],
                  set_series_n34_total=['n34'],
                  set_series_3index_total=['dmi', 'n34', 'strato'],
                  set_series_n34_directo=None,
                  set_series_dmi_directo=None,
                  set_series_3index_directo=None,
                  name=f"A1STRATO_{per}",
                  variables=variables)

        variables = {'ssam': ssam, 'asam': asam}
        CN_Effect(actor_list, set_series_directo=['dmi', 'n34', 'strato'],
                  set_series_dmi_total=['dmi', 'n34'],
                  set_series_n34_total=['n34'],
                  set_series_3index_total=['dmi', 'n34', 'strato'],
                  set_series_n34_directo=None,
                  set_series_dmi_directo=None,
                  set_series_3index_directo=None,
                  name=f"A1STRATO_vs_sam_{per}",
                  variables=variables)

# ---------------------------------------------------------------------------- #
if plot_mapas:
    print('###################################################################')
    print('Mapas...')
    print('###################################################################')

    hgt200_anom2 = hgt200_anom.sel(lat=slice(-80, 20))
    hgt750_anom2 = hgt750_anom.sel(lat=slice(-80, 20))

    hgt_cmap = get_cbars('hgt200')
    pp_cmap = get_cbars('pp')
    sp_cmap = get_cbars('snr2')

    actors_target = {'A_SIMPLE':['dmi', 'n34'],
                     'A1': ['dmi', 'n34', 'sam'],
                     'A1ASAM': ['dmi', 'n34', 'asam'],
                     'A1ASAMwSAM': ['dmi', 'n34', 'asam'],
                     'A1ASAMwSSAM': ['dmi', 'n34', 'asam'],
                     'A1STRATO': ['dmi', 'n34', 'strato'],
                     'AMD200': ['dmi', 'n34', 'amd200'],
                     'AMD750': ['dmi', 'n34', 'amd750'],
                     'A1SSAM': ['dmi', 'n34', 'ssam']}

    modelos = ['A_SIMPLE','A1', 'A1ASAM', 'A1SSAM', 'A1ASAMwSSAM', 'AMD200',
               'AMD750']

    if use_strato_index:
        modelos = ['A_SIMPLE', 'A1', 'A1ASAM', 'A1SSAM', 'A1ASAMwSSAM',
                   'AMD200', 'AMD750', 'A1STRATO']
        per = '1979_2020'

    for v, v_name, mapa in zip([hgt200_anom2, hgt750_anom2, pp],
                               ['hgt200', 'hgt750', 'pp'],
                               ['hs', 'hs', 'sa']):

        v_cmap = get_cbars(v_name)
        for modelo in modelos:
            # para no tener ploteos iguales con distinto titulo
            # los graficos de efecto total son siempre iguales
            # y los total y directo del 3er indice son iguales
            # por como esta construida la red
            if modelo == 'A_SIMPLE':
                modos = ['total', 'directo']
            else:
                modos = ['directo']

            for actor_number, actor in enumerate(actors_target[modelo]):
                for modo in modos:
                    name_fig = (f"{v_name}_Mod{modelo}_Efecto_{modo}_{actor}"
                                f"_{per}")
                    titulo = (f"{v_name}_Mod{modelo} Efecto {modo} {actor} -"
                              f" {per}")

                    efecto = compute_regression(v['var'], modelo, actor, modo)

                    Plot(efecto, v_cmap, mapa, save, dpi, titulo, name_fig,
                         out_dir)

                    if actor_number == 2:
                        aux_actor_list = {actor:all_3index[actor].copy()}
                        aux_actor_list['dmi'] = dmi.values
                        aux_actor_list['n34'] = n34.values

                        for i in ['dmi', 'n34']:
                            strenght_pathway = efecto * regre(aux_actor_list,
                                                              True, i)

                            name_fig = (
                                f"SP_{i.upper()}_{v_name}_Mod{modelo}_{per}")

                            titulo = (
                                f"Strenght of Indirect pathway of "
                                f"{i.upper()} - {v_name}_Mod{modelo} - {per}")

                            Plot(strenght_pathway, sp_cmap, mapa, save, dpi,
                                 titulo, name_fig, out_dir)

                    if modelo == 'A_SIMPLE' and modo== 'directo' \
                            and actor == 'dmi':

                        aux_actor_list = {'dmi':dmi.values}
                        aux_actor_list['n34'] = n34.values
                        strenght_pathway = efecto * regre(aux_actor_list,
                                                           True, 'n34')
                        name_fig = (
                            f"SP_n34-dmi_{v_name}_Mod{modelo}_{per}")

                        titulo = (
                            f"Strenght of Indirect pathway of "
                            f"N34-DMI - {v_name}_Mod{modelo} - {per}")

                        Plot(strenght_pathway, sp_cmap, mapa, save, dpi,
                             titulo, name_fig, out_dir)

print('-----------------------------------------------------------------------')
print('done')
print('out_dir: /pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cn_effect/')
print('-----------------------------------------------------------------------')
################################################################################
# Test VIF
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
print('Variance Inflation Factor (VIF)')
df = pd.DataFrame({
    #'strato': strato,
    'n34': n34,
    'dmi': dmi,
})
X = sm.add_constant(df)
vif = pd.DataFrame()
vif['Variable'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
if (vif['VIF'].values>5).mean() == 0:
    print('No existe multicolinealidad')

# Testeos rapidos
variables_full = {'ssam': ssam, 'dmi':dmi, 'n34':n34, 'asam':asam,
                  'strato':strato_indice['var'].values}

variables = {'ssam': ssam, 'dmi':dmi, 'n34':n34}
print(regre(variables, True, 'dmi'))
print(regre(variables_full, True, 'dmi'))

variables = {'ssam': ssam, 'n34':n34}
print(regre(variables, True, 'n34'))
variables_test = {'ssam': ssam, 'n34':n34, 'asam':asam}
print(regre(variables_test, True, 'n34'))
print(regre(variables_full, True, 'n34'))

variables = {'ssam': ssam, 'dmi':dmi, 'n34':n34,
             'strato':strato_indice['var'].values}
print(regre(variables, True, 'strato'))
print(regre(variables_full, True, 'strato'))

print(regre(variables_full, True, 'asam'))


variables = {'asam': asam, 'dmi':dmi, 'n34':n34,
             'strato':strato_indice['var'].values}
print(regre(variables, True, 'n34'))

variables = {'asam': asam, 'n34':n34}
print(regre(variables, True, 'n34'))
################################################################################