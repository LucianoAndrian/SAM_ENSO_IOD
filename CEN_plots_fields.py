"""
Plotes de las salidas de CEN_trinity_*_fields.py
"""
# ---------------------------------------------------------------------------- #
save = False
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas_cen/salidas_fields/'
efecto_total = True

# ---------------------------------------------------------------------------- #
import os
import xarray as xr

from funciones.scales_and_cbars import get_scales, get_cbars
from funciones.plots import Plot_Contourf_simple

data_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas_cen/'

# aux funciones -------------------------------------------------------------- #
def aux_lag_name(file, endfile='seasonal.nc'):
    lag_0 = file.split('_')[4]

    if file.split('_')[5] != endfile:
        lag_1 = file.split('_')[4]
        lag = f'{lag_0}_{lag_1}'
    else:
        lag = lag_0

    return lag

# Seasonal ------------------------------------------------------------------- #
files = os.listdir(data_dir)
files = [f for f in files if f.endswith('seasonal.nc')]
totales = [f for f in files if 'totales' in f]
directos = [f for f in files if 'directo' in f]

cbar = get_cbars('cbar_rdbu')
scale = [-1, -0.8, -0.6, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.6, 0.8, 1]
actors = ['dmi', 'n34', 'u50']

for ft, fd in zip(totales, directos):
    data_t = xr.open_dataset(f'{data_dir}{ft}')
    lag_t = aux_lag_name(ft)

    data_d = xr.open_dataset(f'{data_dir}{fd}')
    lag_d = aux_lag_name(fd)

    if lag_t == lag_d:
        for a in actors:
            data_d_ctf = data_d.sel(actor=f'{a}_sig')
            data_d_ctn = data_d.sel(actor=f'{a}')

            titulo = f'Effecto directo {a} - HGT200 -  Lag: {lag_t.upper()}'
            namefig = f'hgt200_efecto_directo_{a}_{lag_t}'

            Plot_Contourf_simple(data=data_d_ctf, levels=scale,
                                 cmap=cbar, map='hs',
                                 title=titulo, namefig=namefig,
                                 save=save, out_dir=out_dir,
                                 data_ctn=data_d_ctn,
                                 levels_ctn=None, color_ctn='k',
                                 high=1.5, width=4,
                                 cbar_pos='V', plot_step=1,
                                 pdf=False, ocean_mask=False,
                                 data_ctn_no_ocean_mask=False)

            if efecto_total:
                data_t_ctf = data_t.sel(actor=f'{a}_sig')
                data_t_ctn = data_t.sel(actor=f'{a}')

                titulo = f'Effecto total {a} - HGT200 -  Lag: {lag_t.upper()}'
                namefig = f'hgt200_efecto_total_{a}_{lag_t}'

                Plot_Contourf_simple(data=data_t_ctf, levels=scale,
                                     cmap=cbar, map='hs',
                                     title=titulo, namefig=namefig,
                                     save=save, out_dir=out_dir,
                                     data_ctn=data_t_ctn,
                                     levels_ctn=None, color_ctn='k',
                                     high=1.5, width=4,
                                     cbar_pos='V', plot_step=1,
                                     pdf=False, ocean_mask=False,
                                     data_ctn_no_ocean_mask=False)
    else:
        print('Error en lags')
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #