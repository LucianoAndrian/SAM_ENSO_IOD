"""
Plotes de las salidas de CEN_trinity_*_fields.py
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas_cen/salidas_fields/'
efecto_total = True

# ---------------------------------------------------------------------------- #
import os
import xarray as xr

from funciones.scales_and_cbars import get_cbars
from funciones.plots import Plot_Contourf_simple

data_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas_cen/'

# aux funciones -------------------------------------------------------------- #
def aux_lag_name(file, endfile='seasonal.nc'):
    lag_0 = file.split('_')[4]

    if file.split('_')[5] != endfile:
        lag_1 = file.split('_')[5]
        lag = f'{lag_0}_{lag_1}'
    else:
        lag = lag_0

    return lag

def Plot(data_dir, name_variable, cbar, scale, map, lags=None,
         cbar_position='V',
         width=4, high=1.5,
         pdf=False, ocean_mask=False,
         data_ctn_no_ocean_mask=False,
         plot_efecto_totales=True, save=save):

    actors = ['dmi', 'n34', 'u50']
    files_or = os.listdir(data_dir)
    for lag in lags:

        files = [f for f in files_or if f.endswith('seasonal.nc')]
        files = [f for f in files if name_variable.lower() in f]
        files = [f for f in files if f'_{lag.lower()}_' in f]
        totales = [f for f in files if 'totales' in f]
        directos = [f for f in files if 'directo' in f]

        for ft, fd in zip(totales, directos):
            data_t = xr.open_dataset(f'{data_dir}{ft}')
            lag_t = aux_lag_name(ft)

            data_d = xr.open_dataset(f'{data_dir}{fd}')
            lag_d = aux_lag_name(fd)

            if lag_t == lag_d:
                for a in actors:
                    data_d_ctf = data_d.sel(actor=f'{a}_sig')
                    data_d_ctn = data_d.sel(actor=f'{a}')

                    titulo = (f'Effecto directo {a} - '
                              f'{name_variable.upper()} - Lag: {lag_t.upper()}')
                    namefig = f'{name_variable.lower()}_efecto_directo_{a}_{lag_t}'

                    Plot_Contourf_simple(data=data_d_ctf, levels=scale,
                                         cmap=cbar, map=map,
                                         title=titulo, namefig=namefig,
                                         save=save, out_dir=out_dir,
                                         data_ctn=data_d_ctn,
                                         levels_ctn=None, color_ctn='k',
                                         high=high, width=width,
                                         cbar_pos=cbar_position, plot_step=1,
                                         pdf=pdf, ocean_mask=ocean_mask,
                                         data_ctn_no_ocean_mask=
                                         data_ctn_no_ocean_mask)

                    if plot_efecto_totales:
                        data_t_ctf = data_t.sel(actor=f'{a}_sig')
                        data_t_ctn = data_t.sel(actor=f'{a}')

                        titulo = (f'Effecto total {a} - '
                                  f'{name_variable.upper()} - Lag: {lag_t.upper()}')
                        namefig = f'{name_variable.lower()}_efecto_total_{a}_{lag_t}'

                        Plot_Contourf_simple(data=data_t_ctf, levels=scale,
                                             cmap=cbar, map=map,
                                             title=titulo, namefig=namefig,
                                             save=save, out_dir=out_dir,
                                             data_ctn=data_t_ctn,
                                             levels_ctn=None, color_ctn='k',
                                             high=high, width=width,
                                             cbar_pos=cbar_position, plot_step=1,
                                             pdf=pdf, ocean_mask=ocean_mask,
                                             data_ctn_no_ocean_mask=
                                             data_ctn_no_ocean_mask)
            else:
                print('Error en lags')

# Seasonal ------------------------------------------------------------------- #
lags = ['SON', 'ASO--SON', 'ASO', 'JAS_ASO--SON', 'JAS--SON']

# hgt200
Plot(data_dir=data_dir, name_variable='hgt200',
     cbar=get_cbars('cbar_rdbu'),
     scale= [-1, -0.8, -0.6, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.6, 0.8, 1],
     map='hs', cbar_position='V',
     width=4, high=1.5,
     pdf=False, ocean_mask=False,
     data_ctn_no_ocean_mask=False,
     plot_efecto_totales=efecto_total,
     save=save,
     lags=lags)

#prec
Plot(data_dir=data_dir, name_variable='prec',
     cbar=get_cbars('pp'),
     scale= [-1, -0.8, -0.6, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.6, 0.8, 1],
     map='sa', cbar_position='V',
     width=2.5, high=3.5,
     pdf=False, ocean_mask=True,
     data_ctn_no_ocean_mask=True,
     plot_efecto_totales=efecto_total,
     save=save,
     lags=lags)

#tref
Plot(data_dir=data_dir, name_variable='tref',
     cbar=get_cbars('cbar_rdbu'),
     scale= [-1, -0.8, -0.6, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.6, 0.8, 1],
     map='sa', cbar_position='V',
     width=2.5, high=3.5,
     pdf=False, ocean_mask=True,
     data_ctn_no_ocean_mask=True,
     plot_efecto_totales=efecto_total,
     save=save,
     lags=lags)
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #