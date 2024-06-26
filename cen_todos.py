"""
Redes cusales ENSO, IOD, SAM, ASAM, STRATO

pasado en limpio sólo para la red mas grande que incluye tdo
"""
################################################################################
# Seteos generales ----------------------------------------------------------- #
save = False
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
from cen_funciones import CN_Effect_2
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

# CEN DMI, N34 Strato -------------------------------------------------------- #
for i in [0.05, 0.1, 0.15]:
    print(f"significancia {i}")
    CN_Effect_2(actor_list,
                set_series_directo=['dmi', 'n34'],
                set_series_totales={'dmi': ['dmi', 'n34'],
                                    'n34': ['n34']},
                variables={'strato': strato_indice['var']}, alpha=i,
                sig=True)

# CEN DMI, N34 SSAM ---------------------------------------------------------- #
for i in [0.05, 0.1, 0.15]:
    print(f"significancia {i}")
    CN_Effect_2(actor_list,
                set_series_directo=['dmi', 'n34'],
                set_series_totales={'dmi': ['dmi', 'n34'],
                                    'n34': ['n34']},
                variables={'ssam': ssam}, alpha=i, sig=True)

# CEN DMI, N34 SSAM STRATO  -------------------------------------------------- #
for i in [0.05, 0.1, 0.15]:
    print(f"significancia {i}")
    CN_Effect_2(actor_list,
                set_series_directo=['dmi', 'n34', 'strato'],
                set_series_totales={'dmi': ['dmi', 'n34'], 'n34': ['n34'],
                                    'strato': ['dmi', 'n34', 'strato']},
                variables={'ssam': ssam}, sig=True, alpha=i)

# CEN DMI, N34 asam ---------------------------------------------------------- #
for i in [0.05, 0.1, 0.15]:
    print(f"significancia {i}")
    CN_Effect_2(actor_list,
                set_series_directo=['dmi', 'n34'],
                set_series_totales={'dmi': ['dmi', 'n34'], 'n34': ['n34']},
                variables={'asam': asam}, sig=True, alpha=i)

# CEN DMI, N34 asam STRATO  -------------------------------------------------- #
for i in [0.05, 0.1, 0.15]:
    print(f"significancia {i}")
    CN_Effect_2(actor_list,
                set_series_directo=['dmi', 'n34', 'strato'],
                set_series_totales={'dmi': ['dmi', 'n34'], 'n34': ['n34'],
                                    'strato': ['dmi', 'n34', 'strato']},
                variables={'asam': asam}, sig=True, alpha=i)

################################################################################
# lags
hgt200_anom = hgt200_anom_or.sel(time=hgt200_anom_or.time.dt.month.isin([10]))
hgt750_anom = SameDateAs(hgt750_anom_or, hgt200_anom)
# DMI y N34 en ASO
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

#for l in [10,9,8,7,6,5,4,3]:
dmi = dmi_or.sel(time=dmi_or.time.dt.month.isin([10]))
dmi = dmi.sel(time=dmi.time.dt.year.isin(hgt200_anom.time.dt.year))

    # years_to_remove = [2019, 1997, 1994, 2015, 2006, 2009, 2010, 2008, 1996,
    #                    2016, 1998, 2005, 2001,1982]
    # aux_dmi = dmi.sel(time=~dmi['time.year'].isin(years_to_remove))
    # aux_asam = asam.sel(time=~asam['time.year'].isin(years_to_remove))
    #
    # fig, ax = plt.subplots(dpi=dpi)
    # # todos
    # ax.scatter(x=aux_asam, y=aux_dmi, marker='.',
    #            s=20, edgecolor='k', color='dimgray', alpha=1)
    # for i, txt in enumerate(aux_dmi.time.dt.year.values):
    #     ax.annotate(txt, (aux_asam.values[i], aux_dmi.values[i]))
    # plt.title(f"lag {l}: {round(pearsonr(aux_dmi, aux_asam)[0],3)} "
    #           f"{round(pearsonr(aux_dmi, aux_asam)[1],3)}")
    # plt.show()


n34 = SameDateAs(n34_or, dmi)
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

# CEN DMI, N34 --------------------------------------------------------------- #
for i in [0.05, 0.1, 0.15]:
    print(f"significancia {i}")
    CN_Effect_2(actor_list,
                set_series_directo=['dmi'],
                set_series_totales={'dmi': ['dmi']},
                variables={'n34': n34}, alpha=i,
                sig=True)

# CEN DMI, N34 Strato -------------------------------------------------------- #
for i in [0.05, 0.1, 0.15]:
    print(f"significancia {i}")
    CN_Effect_2(actor_list,
                set_series_directo=['dmi', 'n34'],
                set_series_totales={'dmi': ['dmi', 'n34'],
                                    'n34': ['n34']},
                variables={'strato': strato_indice['var']}, alpha=i,
                sig=True)

# CEN DMI, N34 SSAM ---------------------------------------------------------- #
for i in [0.05, 0.1, 0.15]:
    print(f"significancia {i}")
    CN_Effect_2(actor_list,
                set_series_directo=['dmi', 'n34'],
                set_series_totales={'dmi': ['dmi', 'n34'],
                                    'n34': ['n34']},
                variables={'ssam': ssam}, alpha=i, sig=True)

# CEN DMI, N34 SSAM STRATO  -------------------------------------------------- #
for i in [0.05, 0.1, 0.15]:
    print(f"significancia {i}")
    CN_Effect_2(actor_list,
                set_series_directo=['dmi', 'n34', 'strato'],
                set_series_totales={'dmi': ['dmi', 'n34'], 'n34': ['n34'],
                                    'strato': ['dmi', 'n34', 'strato']},
                variables={'ssam': ssam}, sig=True, alpha=i)

# CEN DMI, N34 asam ---------------------------------------------------------- #
for i in [0.05, 0.1, 0.15]:
    print(f"significancia {i}")
    CN_Effect_2(actor_list,
                set_series_directo=['dmi', 'n34'],
                set_series_totales={'dmi': ['dmi', 'n34'], 'n34': ['n34']},
                variables={'asam': asam}, sig=True, alpha=i)

# CEN DMI, N34 asam STRATO  -------------------------------------------------- #
for i in [0.05, 0.1, 0.15]:
    print(f"significancia {i}")
    CN_Effect_2(actor_list,
                set_series_directo=['dmi', 'n34', 'strato'],
                set_series_totales={'dmi': ['dmi', 'n34'], 'n34': ['n34'],
                                    'strato': ['dmi', 'n34', 'strato']},
                variables={'asam': asam}, sig=True, alpha=i)
