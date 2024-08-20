"""
Scatter plot enso-iod-u50-X'
"""
################################################################################
save = True
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/scatter_in_com/'

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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
from ENSO_IOD_Funciones import Nino34CPC, DMI2
from cen_funciones import (OpenObsDataSet, Detrend, Weights,
                           auxSetLags_ActorList)
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
def ScatterPlot(a, b, c, d, save, x_label='a', y_label='b', c_label='c',
                d_label='d', fix_marker_size=50, tick_label_size=5,
                label_legend_size=0, title='',
                cmap='RdBu', namefig='fig', dpi=300):

    c_pos = dmi[np.where(c > 0)].time.dt.year.values
    c_neg = dmi[np.where(c < 0)].time.dt.year.values

    a_pos = a.sel(time=a.time.dt.year.isin(c_pos))
    a_neg = a.sel(time=a.time.dt.year.isin(c_neg))

    b_pos = b.sel(time=b.time.dt.year.isin(c_pos))
    b_neg = b.sel(time=b.time.dt.year.isin(c_neg))

    d_pos = d.sel(time=d.time.dt.year.isin(c_pos))
    d_neg = d.sel(time=d.time.dt.year.isin(c_neg))

    # by chatpgt ------------------------------------------------------------- #
    # Normaliza los valores de d para usar en el mapa de colores
    norm = mcolors.Normalize(vmin=min(d.min().item(), -d.max().item()),
                             vmax=max(d.max().item(), -d.min().item()))
    cmap = cm.ScalarMappable(norm=norm, cmap=cmap)

    # ------------------------------------------------------------------------ #
    fig, ax = plt.subplots(dpi=dpi, figsize=(7.08661, 7.08661))

    # ax.scatter(x=a, y=b, marker='.', s=20, edgecolor='k',
    #            color='k', alpha=0.75)

    scatter_pos = ax.scatter(x=a_pos, y=b_pos, marker='^',
                             #s=np.abs(np.array(d_pos)) * fix_marker_size,
                             s= fix_marker_size,
                             edgecolor='k', c=cmap.to_rgba(np.array(d_pos)),
                             alpha=0.75, label=f"{c_label} > 0")

    scatter_neg = ax.scatter(x=a_neg, y=b_neg, marker='v',
                             #s=np.abs(np.array(d_neg)) * fix_marker_size,
                             s=fix_marker_size,
                             edgecolor='k', c=cmap.to_rgba(np.array(d_neg)),
                             alpha=0.75, label=f"{c_label} < 0")

    ax.scatter(x=[], y=[], marker='',
               s=0, alpha=0,
               label=f"Color & Size = {d_label}")


    cbar = plt.colorbar(cmap, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label(f'{d_label}', fontsize=tick_label_size)
    cbar.ax.tick_params(labelsize=tick_label_size)

    lgnd = ax.legend(loc=(.01, .80), fontsize=label_legend_size)
    lgnd.legendHandles[0]._sizes = [100]
    lgnd.legendHandles[1]._sizes = [100]
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size, pad=1)
    ax.set_ylim((-3, 3))
    ax.set_xlim((-3, 3))
    ax.grid()
    ax.hlines(y=0, xmin=-3, xmax=3, colors='k')
    ax.vlines(x=0, ymin=-3, ymax=3, colors='k')
    ax.set_title(title, size=label_legend_size)

    ax.set_xlabel(x_label, size=tick_label_size)
    ax.set_ylabel(y_label, size=tick_label_size)

    plt.tight_layout()

    if save:
        plt.savefig(f"{out_dir}{namefig}.jpg", dpi=dpi, bbox_inches='tight')
        plt.close('all')
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

# 750hpa
# hgt = xr.open_dataset(hgt_dir + 'ERA5_HGT750_40-20.nc')
# hgt = hgt.rename({'longitude': 'lon', 'latitude': 'lat', 'z': 'var'})
# hgt = hgt.interp(lon=np.arange(0,360,2), lat=np.arange(-90, 90, 2))
#
# hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
# hgt750_anom_or = hgt.groupby('time.month') - \
#            hgt_clim.groupby('time.month').mean('time')
#
# weights = np.sqrt(np.abs(np.cos(np.radians(hgt750_anom_or.lat))))
# hgt750_anom_or = hgt750_anom_or * weights
#
# hgt750_anom_or = hgt750_anom_or.rolling(time=3, center=True).mean()
# hgt750_anom_or = hgt750_anom_or.sel(time=slice('1940-02-01', '2020-11-01'))

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
# sam_or = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
# sam_or = sam_or.rolling(time=3, center=True).mean()
#
# asam_or = xr.open_dataset(sam_dir + 'asam_700.nc')['mean_estimate']
# asam_or = asam_or.rolling(time=3, center=True).mean()
#
# ssam_or = xr.open_dataset(sam_dir + 'ssam_700.nc')['mean_estimate']
# ssam_or = ssam_or.rolling(time=3, center=True).mean()

dmi_or = DMI2(filter_bwa=False, start_per='1959', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]
dmi_or = dmi_or / dmi_or.std('time')

sst_aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/"
                      "sst.mnmean.nc")
sst_aux = sst_aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34_or = Nino34CPC(sst_aux, start=1920, end=2020)[0]
n34_or = n34_or / n34_or.std('time')

u50_or = xr.open_dataset('/pikachu/datos/luciano.andrian/observado/'
                         'ncfiles/ERA5/downloaded/ERA5_U50hpa_40-20.mon.nc')
u50_or = u50_or.rename({'u': 'var'})
u50_or = u50_or.rename({'longitude': 'lon'})
u50_or = u50_or.rename({'latitude': 'lat'})
u50_or = Weights(u50_or)
u50_or = u50_or.sel(lat=-60)
u50_or = u50_or - u50_or.mean('time')
u50_or = u50_or.rolling(time=3, center=True).mean()
u50_or = Detrend(u50_or, 'time')
u50_or = u50_or.sel(expver=1).drop('expver')
u50_or = u50_or.mean('lon')
u50_or = xr.DataArray(u50_or['var'].drop('lat'))
u50_or = u50_or / u50_or.std('time')
################################################################################
hgt200_anom_or =\
    hgt200_anom_or.sel(time=hgt200_anom_or.time.dt.year.isin(range(1959,2021)))

lags = {'SON':[10,10,10],
        'ASO--SON':[10, 9, 9],
        'JAS_ASO--SON':[10, 8, 9],
        'JAS--SON':[10, 8, 8]}

for l_count, lag_key in enumerate(lags.keys()):
    seasons_lags = lags[lag_key]
    print(f"{lag_key} ########################################################")

    hgt200_anom, pp, asam, ssam, u50, strato_indice2, dmi, n34, actor_list, \
    dmi_aux, n34_aux, u50_aux, sam_aux, aux_ssam, aux_asam  = \
        auxSetLags_ActorList(lag_target=seasons_lags[0],
                             lag_dmin34=seasons_lags[1],
                             lag_strato=seasons_lags[2],
                             hgt200_anom_or=hgt200_anom_or, pp_or=pp_or,
                             dmi_or=dmi_or, n34_or=n34_or, u50_or=u50_or,
                             strato_indice=None,
                             years_to_remove=[2002,2019])

    pp_aux = pp.sel(lat=slice(pp_lats[0], pp_lats[1]),
                    lon=slice(pp_lons[0],pp_lons[1])).mean(['lon', 'lat'])
    pp_aux = pp_aux/pp_aux.std('time')

    u50_aux = u50-u50.mean()
    u50_aux = u50_aux/u50_aux.std('time')

    ScatterPlot(pp_aux['var'], n34, dmi, u50_aux, save, x_label='PP anom.',
                y_label='N34', c_label='DMI', d_label="u50",
                fix_marker_size=300, tick_label_size=15, label_legend_size=12,
                title=f"Lag: {lag_key}",
                cmap='RdBu_r',
                namefig=f"pp_vs_n34_cDMI_{lag_key}")

    ScatterPlot(pp_aux['var'], n34, u50_aux, dmi, save, x_label='PP anom.',
                y_label='N34', c_label='U50', d_label="DMI",
                fix_marker_size=300, tick_label_size=15, label_legend_size=12,
                title=f"Lag: {lag_key}", cmap='PRGn',
                namefig=f"pp_vs_n34_cU50_{lag_key}")

    ScatterPlot(dmi, n34, u50_aux, pp_aux['var'], save, x_label='dmi',
                y_label='n34', c_label='u50', d_label="pp",
                fix_marker_size=300, tick_label_size=15, label_legend_size=12,
                title=f"Lag: {lag_key}",cmap='PuOr_r',
                namefig=f"dmi_vs_n34_cU50_{lag_key}")
################################################################################