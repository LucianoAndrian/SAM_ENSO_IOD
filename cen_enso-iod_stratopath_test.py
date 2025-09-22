"""
Similar a enso-iod vs indices pero testeando especificamente contra stratoindex
Lo mismo puede hacerse en cen_enso-iod_vs_indices.py escribiendo los modelos
adecuados
"""
################################################################################
# Seteos generales ----------------------------------------------------------- #
save = False
plot_mapas = True
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

warnings.filterwarnings("ignore", module="matplotlib\..*")
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2
from Scales_Cbars import get_cbars

################################################################################
if save:
    dpi = 200
else:
    dpi = 70
use_strato_index = True
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
                coefs_results[e] = coefs[ec - 1]

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
              name='cn_result'):
    result_df = pd.DataFrame(columns=['v_efecto', 'b'])

    for x, x_name in zip([pp_caja, amd], ['pp_serie', 'amd']):

        pre_serie = {'c': x['var'].values}
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

        # series = {'dmi_total': series_dmi_total,
        #           'n34_total': series_n34_total,
        #           aux_3index + '_total': series_3index_total}

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
                                          'b': i_directo},
                                         ignore_index=True)
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

    elif modelo.upper() == 'AMD':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'amd': amd['var'].values}
        set_series_directo = ['dmi', 'n34', 'amd']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_3index_total = ['dmi', 'n34', 'amd']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'amd'

    elif modelo.upper() == 'A1SSAM':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'ssam': asam.values}
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
        variable, modelo, coef, modo,
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
hgt = hgt.interp(lon=np.arange(0, 360, 2), lat=np.arange(-90, 90, 2))

hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
hgt_anom_or = hgt.groupby('time.month') - \
              hgt_clim.groupby('time.month').mean('time')

weights = np.sqrt(np.abs(np.cos(np.radians(hgt_anom_or.lat))))
hgt_anom_or = hgt_anom_or * weights

hgt_anom_or = hgt_anom_or.rolling(time=3, center=True).mean()
hgt_anom_or = hgt_anom_or.sel(time=slice('1940-02-01', '2020-11-01'))
hgt_anom_or = hgt_anom_or.sel(
    time=hgt_anom_or.time.dt.month.isin([8, 9, 10, 11]))
hgt_anom_or = hgt_anom_or.sel(time=hgt_anom_or.time.dt.month.isin([10]))

# PP ------------------------------------------------------------------------- #
pp_or = OpenObsDataSet(name='pp_pgcc_v2020_1891-2023_1', sa=True, dir=dir_pp)
pp_or = pp_or.rename({'precip': 'var'})
pp_or = pp_or.sel(time=slice('1940-01-16', '2020-12-16'))

pp_or = Weights(pp_or)
pp_or = pp_or.sel(lat=slice(20, -60), lon=slice(270, 330))  # SA
pp_or = pp_or.rolling(time=3, center=True).mean()
pp_or = pp_or.sel(time=pp_or.time.dt.month.isin([8, 9, 10, 11]))
pp_or = Detrend(pp_or, 'time')

# Caja PP
pp_caja_or = pp_or.sel(lat=slice(pp_lats[0], pp_lats[1]),
                       lon=slice(pp_lons[0], pp_lons[1])).mean(['lon', 'lat'])
pp_caja_or['var'][-1] = 0  # aca nse que pasa.

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
    strato_indice = xr.open_dataset('strato_index.nc').rename({'year': 'time'})
    strato_indice = strato_indice.rename(
        {'__xarray_dataarray_variable__': 'var'})
    hgt_anom_or = hgt_anom_or.sel(time=
    hgt_anom_or.time.dt.year.isin(
        strato_indice['time']))
    strato_indice = strato_indice.sel(time=hgt_anom_or['time.year'])

# ---------------------------------------------------------------------------- #
# SameDate y normalización --------------------------------------------------- #
# ---------------------------------------------------------------------------- #
hgt_anom = hgt_anom_or.sel(time=hgt_anom_or.time.dt.month.isin([10]))

dmi = SameDateAs(dmi_or, hgt_anom)
n34 = SameDateAs(n34_or, hgt_anom)
sam = SameDateAs(sam_or, hgt_anom)
asam = SameDateAs(asam_or, hgt_anom)
ssam = SameDateAs(ssam_or, hgt_anom)
pp = SameDateAs(pp_or, hgt_anom)
pp_caja = SameDateAs(pp_caja_or, hgt_anom)
dmi = dmi / dmi.std()
n34 = n34 / n34.std()
sam = sam / sam.std()
asam = asam / asam.std()
ssam = ssam / ssam.std()
amd = (hgt_anom.sel(lon=slice(210, 270), lat=slice(-80, -50)).
       mean(['lon', 'lat']))
amd = amd / amd.std()
hgt_anom = hgt_anom / hgt_anom.std()
pp_caja = pp_caja / pp_caja.std()
pp = pp / pp.std()

# CEN ------------------------------------------------------------------------ #
print('#######################################################################')
print('Modelo A: N34->IOD (todos a C) sin 3er indice')
actor_list = {'dmi':dmi.values, 'n34':n34.values}
CN_Effect(actor_list,  set_series_directo = ['dmi', 'n34'],
          set_series_dmi_total=['dmi', 'n34'],
          set_series_n34_total=['n34'],
          set_series_3index_total=None,
          set_series_n34_directo=None,
          set_series_dmi_directo=None,
          set_series_3index_directo=None,
          name=f"A_simple_DMI_N34_{per}")

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
          name=f"A1STRATO_{per}")

# ---------------------------------------------------------------------------- #
if plot_mapas:
    print('###################################################################')
    print('Mapas...')
    print('###################################################################')

    hgt_anom2 = hgt_anom.sel(lat=slice(-80, 20))
    hgt_cmap = get_cbars('hgt200')
    pp_cmap = get_cbars('pp')

    actors_target = {'A_SIMPLE': ['dmi', 'n34'],
                     'A1STRATO': ['dmi', 'n34', 'strato']}

    modelos = ['A_SIMPLE', 'A1STRATO']


    for v, v_name, mapa in zip([hgt_anom2, pp], ['hgt200', 'pp'], ['hs', 'sa']):
        v_cmap = get_cbars(v_name)
        for modelo in modelos:
            for actor in actors_target[modelo]:
                for modo in ['total', 'directo']:
                    name_fig = (f"{v_name}_Mod{modelo}_Efecto_{modo}_{actor}"
                                f"_{per}")
                    titulo = (f"{v_name}_Mod{modelo} Efecto {modo} {actor} -"
                              f" {per}")

                    efecto = compute_regression(v['var'], modelo, actor, modo)

                    Plot(efecto, v_cmap, mapa, save, dpi, titulo, name_fig,
                         out_dir)