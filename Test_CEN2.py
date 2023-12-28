"""
TEST Algoritmo de descubrimiento causal
en UN punto de grilla
"""

################################################################################
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import xarray as xr
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import statsmodels.api as sm
from PCMCI import PCMCI
import concurrent.futures
import time
import matplotlib.pyplot as plt
################################################################################
################################################################################
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/mlr/'
era5_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
t_pp_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_obs_d_w_c/'
# ---------------------------------------------------------------------------- #
path = '/pikachu/datos/luciano.andrian/cases_fields/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/eof/'
################################################################################
################################################################################
ruta = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
hgt = xr.open_dataset(ruta + 'ERA5_HGT200_40-20.nc')
hgt = hgt.rename({'longitude': 'lon', 'latitude': 'lat', 'z': 'var'})
hgt = hgt.interp(lon=np.arange(0,360,2), lat=np.arange(-90, 90, 2))

hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
hgt_anom = hgt.groupby('time.month') - \
           hgt_clim.groupby('time.month').mean('time')

# weights = np.sqrt(np.abs(np.cos(np.radians(hgt_anom.lat))))
# hgt_anom = hgt_anom * weights

#hgt_anom2 = hgt_anom.sel(lat=slice(-80, 0), lon=slice(60, 70))
hgt_anom = hgt_anom.rolling(time=3, center=True).mean()
hgt_anom = hgt_anom.sel(time=slice('1940-02-01', '2020-11-01'))
#hgt_anom = hgt_anom.sel(time=hgt_anom.time.dt.month.isin([8,9,10,11]))
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam = sam.rolling(time=3, center=True).mean()

dmi = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

aux = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")
n34 = Nino34CPC(aux, start=1920, end=2020)[0]
################################################################################
#c = hgt_anom2.sel(lon=65, lat=-75)#.sel(time=hgt_anom.time.dt.year.isin(np.arange(1980,2020)))
#c = c.sel(time=c.time.dt.year.isin([np.arange(2010,2018)]))
#c = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
dmi2 = SameDateAs(dmi, hgt_anom)
n342 = SameDateAs(n34, hgt_anom)
#sam2 = SameDateAs(sam, hgt_anom)

#sam3 = sam2
#c = c/c.std()
dmi3 = dmi2/dmi2.std()
n343 = n342/n342.std()
# sam3 = sam2/sam2.std()
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

def regre(df):
    import statsmodels.api as sm

    x_model = sm.OLS(df['x'], sm.add_constant(df[df.columns[2:]])).fit()
    x_res = x_model.resid

    return x_res

def regre(df):
    X = np.column_stack((np.ones_like(df[df.columns[2:]]), df[df.columns[2:]]))
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
################################################################################
def aux_func(x):
    x_values = x
    series = {'c': x_values, 'dmi': dmi3.values, 'n34': n343.values}
    result_df = PCMCI(series=series, tau_max=2, pc_alpha=0.05, mci_alpha=0.05)
    actors_res = []
    for actor in ['c', 'dmi', 'n34']:
        aux_df = result_df.loc[result_df['Target'] == actor]
        links = list(aux_df['Actor'])
        df = SetLags(series[actor], series['c'], ty=0, series=series, parents=links)
        actors_res.append(regre(df))

    len_min = min(len(serie) for serie in actors_res)

    for i, a_res in enumerate(actors_res):
        actors_res[i] = resize_serie(a_res.values, len_min)
    aux_df = pd.DataFrame({'c': actors_res[0], 'dmi': actors_res[1], 'n34': actors_res[2]})
    model = sm.OLS(aux_df['c'], sm.add_constant(aux_df[aux_df.columns[1:]])).fit()

    return model.params[1]

def fakedaks(c):
    try:
        hgt_anom2 = hgt_anom.sel(lat=slice(c[0], c[1]), lon=slice(c[2], c[3]))
        reg_array = xr.apply_ufunc(
            aux_func,
            hgt_anom2['var'],
            input_core_dims=[['time']],
            #dask='parallelized',
            vectorize=True,
            output_dtypes=[float])
        return reg_array
    except Exception as e:
        print(f"Error in fakedaks for chunk {c}: {e}")


# hacer funcion para esto
lonlat = [[-90, -60, 60, 90], [-90, -60, 91, 120], [-90, -60, 121, 150],
          [-90, -60, 151, 180], [-90, -60, 181, 210], [-90, -60, 211, 240],
          [-90, -60, 241, 270], [-90, -60, 271, 300], [-90, -60, 301, 330],
          [-61, -30, 60, 90], [-61, -30, 91, 120], [-61, -30, 121, 150],
          [-61, -30, 151, 180], [-61, -30, 181, 210], [-61, -30, 211, 240],
          [-61, -30, 241, 270], [-61, -30, 271, 300], [-61, -30, 301, 330],
          [-31, 0, 60, 90], [-31, 0, 91, 120], [-31, 0, 121, 150],
          [-31, 0, 151, 180], [-31, 0, 181, 210], [-31, 0, 211, 240],
          [-31, 0, 241, 270], [-31, 0, 271, 300], [-31, 0, 301, 330]]


time0 = time.time()

with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
    aux_result = list(executor.map(fakedaks, lonlat))

delta_t = time.time() - time0
print(f"Tiempo: {delta_t} segundos")

aux_xr = xr.merge(aux_result)

plt.contourf(aux_xr.where(aux_xr)['var'], cmap='RdBu_r',
             levels=np.arange(-150, 150, 25),
             extend='both')
plt.colorbar()
plt.show()


