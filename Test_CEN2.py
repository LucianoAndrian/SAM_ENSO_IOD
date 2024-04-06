"""
Intento simple de usar pcmci en dmi-enso por punto de grilla
"""

################################################################################
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import xarray as xr
import pandas as pd
pd.options.mode.chained_assignment = None

from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2
from PCMCI import PCMCI

import statsmodels.api as sm
import concurrent.futures
from datetime import datetime
import time

import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
################################################################################
save=False
use_sam=False
################################################################################
# ---------------------------------------------------------------------------- #
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
hgt_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
#out_dir = ''
# ---------------------------------------------------------------------------- #
hgt = xr.open_dataset(hgt_dir + 'ERA5_HGT200_40-20.nc')
hgt = hgt.rename({'longitude': 'lon', 'latitude': 'lat', 'z': 'var'})
hgt = hgt.interp(lon=np.arange(0,360,2), lat=np.arange(-90, 90, 2))

hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
hgt_anom = hgt.groupby('time.month') - \
           hgt_clim.groupby('time.month').mean('time')

if use_sam:
    weights = np.sqrt(np.abs(np.cos(np.radians(hgt_anom.lat))))
    hgt_anom = hgt_anom * weights

#hgt_anom2 = hgt_anom.sel(lat=slice(-80, 0), lon=slice(60, 70))
hgt_anom = hgt_anom.rolling(time=3, center=True).mean()
hgt_anom = hgt_anom.sel(time=slice('1940-02-01', '2020-11-01'))
hgt_anom = hgt_anom.sel(time=hgt_anom.time.dt.month.isin([8,9,10,11]))
################################################################################
# indices
# ---------------------------------------------------------------------------- #
sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam = sam.rolling(time=3, center=True).mean()

dmi = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
aux = aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34 = Nino34CPC(aux, start=1920, end=2020)[0]
# ---------------------------------------------------------------------------- #
dmi2 = SameDateAs(dmi, hgt_anom)
n342 = SameDateAs(n34, hgt_anom)
sam2 = SameDateAs(sam, hgt_anom)

#sam3 = sam2
#c = c/c.std()
dmi3 = dmi2/dmi2.std()
n343 = n342/n342.std()
sam3 = sam2/sam2.std()
#------------------------------------------------------------------------------#
################################################################################
# Funciones ####################################################################
def recursive_cut(s1, s2):
    i=1
    while(len(s1)!=len(s2)):
        i += 1
        if len(s1)>len(s2):
            s1 = s1[1:]
        else:
            s2 = s2[1:]

        if i > 100:
            print('Error recursive_cut +100 iter.')
            return

    return s1, s2


def min_len(st):
    min_len = float('inf')
    min_st = None

    for s in st:
        cu_len = len(s)
        if cu_len < min_len:
            min_len = cu_len
            min_st = s

    return min_st


def SetLags(x, y, ty, series, parents):

    z_series = []
    for i, p  in enumerate(parents):

        zn = series[p.split('_lag_')[0]]
        tz = np.int(p.split('_lag_')[1])
        len_series = len(zn)

        if i==0:
            ty_aux = max(0, tz - ty)
            tz_aux = max(0, ty - tz)

            x = x[tz+tz_aux:]
            y = y[ty_aux:len_series-ty]
            z_series.append(zn[tz_aux:len_series-tz])
            tz_prev = tz

        if i>0:
            if i>1:
                t_aux = 0
            else:
                t_aux = max(ty, tz_prev)

            z_aux = zn[t_aux:len_series - tz]
            z_series.append(z_aux)

            aux = min_len([x, y, z_aux])

            x,_ = recursive_cut(x, aux)
            y,_ = recursive_cut(y, aux)

            for iz, zs in enumerate(z_series):
                z_series[iz] ,_ = recursive_cut(zs, aux)

    z_columns = [f'parent{i}' for i in range(1, len(z_series) + 1)]
    z_data = {column: z_series[i] for i, column in enumerate(z_columns)}

    return pd.DataFrame({'x':x, 'y':y, **z_data})


# def regre(df):
#
#     x_model = sm.OLS(df['y'], sm.add_constant(df[df.columns[2:]])).fit()
#     x_res = x_model.resid
#
#     return x_model.predict()


def regre(df):
    X = np.column_stack((np.ones_like(df[df.columns[1:2]]), df[df.columns[2:]]))
    y = df['y']

    # Calcular los coeficientes de la regresión lineal
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Calcular los residuos
    x_res = y - np.dot(X, beta)

    #return x_res
    return  np.dot(X, beta)


def regre_res(df):
    X = np.column_stack((np.ones_like(df[df.columns[1:2]]), df[df.columns[2:]]))
    y = df['x']

    # Calcular los coeficientes de la regresión lineal
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Calcular los residuos
    x_res = y - np.dot(X, beta)
    return x_res


def resize_serie(s1, len_min):
    i=1
    while(len(s1)!=len_min):
        i += 1
        s1 = s1[1:]
        if i > 100:
            print('Error recursive_cut +100 iter.')
            return
    return s1


def Compute_PCMCI_MLR(x):
    target='sam'
    dc2020=False
    lag=0
    # PENDIENTE
    # SETEAR LAAAG!
    x_values = x
    series = {'c': x_values, 'dmi': dmi3.values, 'n34': n343.values, 'sam':sam3.values}

    result_df = PCMCI(series=series, tau_max=2, pc_alpha=0.2, mci_alpha=0.05)

    actors_parents=[]
    for actor in ['c', 'dmi', 'n34']:
        aux_df = result_df.loc[result_df['Target'] == actor]
        links = list(aux_df['Actor'])

        # Solo autocorrelacion
        own_links = []
        for element in links:
            if actor in element:
                own_links.append(element)

        # own_links=links
        df = SetLags(series[actor], series[actor], ty=lag,
                     series=series, parents=own_links)

        if actor == 'c':
            actors_parents.append(regre(df)) #0
        elif actor == target:
            actors_parents.append(regre(df)) #1
            actors_parents.append(regre_res(df)) #2
        else:
            actors_parents.append(df['x'].values) #3
            actors_parents.append(regre(df)) # 4
            actors_parents.append(regre_res(df)) #5


    # longitud de las series =
    len_min = min(len(serie) for serie in actors_parents)
    for i, a_res in enumerate(actors_parents):
        actors_parents[i] = resize_serie(a_res, len_min)

    # CEN
    if dc2020:
        aux_df = pd.DataFrame({'c_or': resize_serie(series['c'], len_min),
                               target: resize_serie(series[target], len_min),
                               target + '_t': actors_parents[1],
                               'c_t': actors_parents[0],
                               '2actor': actors_parents[3]
                               })
    else:
        # set
        aux_df = pd.DataFrame({'c_or': resize_serie(series['c'], len_min),
                               # target:actors_parents[2],
                               target: resize_serie(series[target], len_min),
                               'n34': resize_serie(series['n34'], len_min),
                               'c_t': actors_parents[0],
                               target + '_t': actors_parents[1],
                               # '2actor': actors_parents[3],
                               '2actor_t': actors_parents[4],
                               })
        # 'c_l': actors_parents[0].values,
        # target +'_l': actors_parents[1],
        # 'actor_WO': actors_parents[3]
        # 'n34_res':actors_parents[2].values})

    model = sm.OLS(aux_df['c_or'], aux_df[aux_df.columns[1:]]).fit()

    # aca se selecciona el beta asociado al target!
    if model.pvalues[0]<0.05:
        return model.params[0]
    else:
        return np.nan

def ComputeByXrUf(c):
    hgt_anom2 = hgt_anom.sel(lat=slice(c[0], c[1]), lon=slice(c[2], c[3]))
    reg_array = xr.apply_ufunc(
        Compute_PCMCI_MLR,
        hgt_anom2['var'],
        input_core_dims=[['time']],
        # dask='parallelized',
        vectorize=True,
        output_dtypes=[float])
    return reg_array

################################################################################
# Subregiones a computar en paralelo
lonlat = [[-80, -60, 30, 60], [-80, -60, 61, 90], [-80, -60, 91, 120],
          [-80, -60, 121, 150], [-80, -60, 151, 180], [-80, -60, 181, 210],
          [-80, -60, 211, 240], [-80, -60, 241, 270], [-80, -60, 271, 300],
          [-80, -60, 301, 330],
          [-59, -30, 30, 60], [-59, -30, 61, 90], [-59, -30, 91, 120],
          [-59, -30, 121, 150], [-59, -30, 151, 180], [-59, -30, 181, 210],
          [-59, -30, 211, 240], [-59, -30, 241, 270], [-59, -30, 271, 300],
          [-59, -30, 301, 330],
          [-31, 0, 30, 60], [-31, 0, 61, 90], [-31, 0, 91, 120],
          [-31, 0, 121, 150], [-31, 0, 151, 180], [-31, 0, 181, 210],
          [-31, 0, 211, 240], [-31, 0, 241, 270], [-31, 0, 271, 300],
          [-31, 0, 301, 330]]

# Fake Dask ------------------------------------------------------------------ #
print(f'Hora de inicio: {datetime.now()}')
time0 = time.time()

with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
    aux_result = list(executor.map(ComputeByXrUf, lonlat))

delta_t = time.time() - time0
print(f"Tiempo: {delta_t/60} minutos")

# uniendo las regiones calculadas en paralelo
aux_xr = xr.merge(aux_result)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# plot
# ---------------------------------------------------------------------------- #
cbar_hgt = colors.ListedColormap(['#9B1C00', '#B9391B', '#CD4838',
                                      '#E25E55', '#F28C89', '#FFCECC',
                                      'white',
                                      '#B3DBFF', '#83B9EB', '#5E9AD7',
                                      '#3C7DC3', '#2064AF', '#014A9B'][::-1])
cbar_hgt.set_over('#641B00')
cbar_hgt.set_under('#012A52')
cbar_hgt.set_bad(color='white')
# ---------------------------------------------------------------------------- #

fig = plt.figure(figsize=(7, 3), dpi=100)
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
crs_latlon = ccrs.PlateCarree()

ax.set_extent([30, 330, -80, 0], crs=crs_latlon)

im = ax.contourf(aux_xr.lon, aux_xr.lat, aux_xr['var'],
                 levels=[-150, -100, -75, -50, -25, -15,
                         0, 15, 25, 50, 75, 100, 150],
                 transform=crs_latlon, cmap=cbar_hgt, extend='both')

values = ax.contour(aux_xr.lon, aux_xr.lat, aux_xr['var'],
                    levels=[-150, -100, -75, -50, -25, -15,
                         0, 15, 25, 50, 75, 100, 150],
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
ax.tick_params(labelsize=7)
plt.title('test', fontsize=10)
plt.tight_layout()
if save:
    plt.savefig('namefig.jpg')
    plt.close()
else:
    plt.show()
plt.show()



