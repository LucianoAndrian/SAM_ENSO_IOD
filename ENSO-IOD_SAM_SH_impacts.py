"""
Impactos en HGT en SH de eventos SAM-ENSO-IOD, individuales y por signo del
SAM y categoria del IOD
"""
################################################################################
dates_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/hgt_anoms/'
era5_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
nc_date_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
              'nc_composites_dates_no_ind_sst_anom/'
################################################################################
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib import colors
import pandas as pd
import numpy as np
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
from ENSO_IOD_Funciones import DMI, Nino34CPC, SameDateAs, DMI2
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
################################################################################
save = True
# usa de climatologia los años neutros enso-iod, recomendado: True
climatology_neutro = True
# sirve para comparar con los resultados de ENSO_IOD
# sino va tomar la media de todos los años
seasons = ['SON']
mmonth = [10]

if save:
    dpi = 300
else:
    dpi = 50
################################################################################
def OpenObsDataSet(name, sa=True,
                   dir='/pikachu/datos/luciano.andrian/observado/ncfiles'
                        '/data_obs_d_w_c/'):

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


def Plot(comp, levels, cmap, title, name_fig, dpi, save, out_dir):

    fig_size =(9, 3.5)
    extent= [0, 359, -80, 20]
    xticks = np.arange(0, 360, 30)
    yticks = np.arange(-80, 20, 10)

    comp_var = comp['var']

    levels_contour = levels.copy()
    levels_contour.remove(0)

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent(extent, crs=crs_latlon)

    ax.contour(comp.lon, comp.lat, comp_var,
               linewidths=.8, levels=levels_contour, transform=crs_latlon,
               colors='black')

    im = ax.contourf(comp.lon, comp.lat, comp_var, levels=levels,
                     transform=crs_latlon, cmap=cmap, extend='both')

    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.7)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='lightgrey',
                   edgecolor='k')
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
    ax.coastlines(color='k', linestyle='-', alpha=1)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(xticks, crs=crs_latlon)
    ax.set_yticks(yticks, crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=8)
    plt.title(title, fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()


def Subplots(data_array, level, cmap, title, save, dpi, name_fig, out_dir):

    extent= [0, 359, -80, 20]
    crs_latlon = ccrs.PlateCarree()

    time_values = data_array.time.values
    time_steps = len(time_values)
    num_cols = 4
    # cantidad de filas necesarias
    num_rows = np.ceil(time_steps/num_cols).astype(int)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(20, 3*num_rows),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

    for i, (ax, time_val) in enumerate(zip(axes.flatten(), time_values)):

        aux = data_array.sel(time=time_val)
        aux_var = aux['var']
        im = ax.contourf(aux.lon, aux.lat, aux_var, levels=level,
                         transform=crs_latlon, cmap=cmap, extend='both')

        ax.add_feature(cartopy.feature.LAND, facecolor='lightgrey',
                       edgecolor='k')
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
        ax.coastlines(color='k', linestyle='-', alpha=1)
        ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
        ax.set_extent(extent, crs=crs_latlon)
        ax.set_title(f'Time: {pd.Timestamp(time_val).year}')

    # Eliminar los lugares en blanco que existan
    for i in range(time_steps, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    pos = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='horizontal')
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=16, y=0.98)

    if save:
        plt.savefig(out_dir + name_fig + '.jpg', dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def CreateDirectory(out_dir, s, v, sam_component):
    if not os.path.exists(out_dir + s):
        os.mkdir(out_dir + s)

    sub_dir1 = out_dir + s + '/' + v
    if not os.path.exists(sub_dir1):
        os.mkdir(sub_dir1)

    sub_dir2 = sub_dir1 + '/' + sam_component
    if not os.path.exists(sub_dir2):
        os.mkdir(sub_dir2)
################################################################################
cbar = colors.ListedColormap(['#9B1C00', '#B9391B', '#CD4838', '#E25E55',
                              '#F28C89', '#FFCECC',
                              'white',
                              '#B3DBFF', '#83B9EB', '#5E9AD7', '#3C7DC3',
                              '#2064AF', '#014A9B'][::-1])
cbar.set_over('#641B00')
cbar.set_under('#012A52')
cbar.set_bad(color='white')
################################################################################

cases_dmi = ['dmi_un_pos', 'dmi_sim_pos', 'dmi_un_neg', 'dmi_sim_neg']
cases_n34 = ['n34_un_pos', 'n34_sim_pos', 'n34_un_neg', 'n34_sim_neg']

variables = ['HGT200']#, 'HGT750']
scale_comp = [-300, -250, -200, -150, -100, -50, -25,
          0, 25, 50, 100, 150, 200, 250, 300]

scale = [-400, -350, -300, -250, -200, -150, -100, -50,
          0, 50, 100, 150, 200, 250, 300, 350, 400]
cmap = cbar

# # for test
# s = 'SON'
# c = 'dmi_un_neg'
# v = 'ppgpcc_w_c_d_1'
# v_count = 0
# sam_component = 'sam'

for cases in [cases_dmi, cases_n34]:
    for s in seasons:
        print(s)

        # by cases ------------------------------------------------------------#
        for c_count, c in enumerate(cases):
            print(c)

            # for sam_component in ['sam', 'ssam', 'asam']:
            for sam_component in ['sam', 'asam']:

                # dataframe con las fechas, dmi y sam para cada iod case
                data = pd.read_csv(
                    dates_dir + c + '_vs_' + sam_component + '_' +
                    s + '.txt', sep='\t', parse_dates=['time'])

                # by variables ------------------------------------------------#
                for v_count, v in enumerate(variables):
                    print(v)

                    variable = xr.open_dataset(era5_dir + v + '_' + s +
                                               '_mer_d_w.nc')

                    # carpetas para guardar cada cosa por separado
                    CreateDirectory(out_dir, s, v, sam_component)

                    if climatology_neutro:
                        aux = xr.open_dataset(
                            nc_date_dir + '1920_2020' + '_' + s + '.nc')
                        variable_mean = variable.sel(
                            time=variable.time.dt.year.isin(aux.Neutral)).mean(
                            'time')
                    else:
                        variable_mean = variable.mean('time')

                    # Selección de IOD según el signo del SAM
                    sam_pos = data.loc[data['sam'] > 0]
                    sam_neg = data.loc[data['sam'] < 0]

                    # by sam sign ---------------------------------------------#
                    for sam, sam_title in zip([sam_pos, sam_neg],
                                              [sam_component + '>0',
                                               sam_component + '<0']):
                        print(sam_title)
                        if len(sam) == 0:
                            continue

                        years = sam.time.dt.year
                        variable_selected = variable.sel(
                            time=variable.time.dt.year.isin(years))

                        # Comp e individual
                        comp = variable_selected.mean('time') - variable_mean
                        variable_selected = variable_selected - variable_mean

                        print('Plots ----------------------------------------')
                        print('Plot comp')
                        title = 'Composite - ' + c + ' with ' + \
                                sam_title.upper() + \
                                '\n' + v + ' - ' + s
                        name_fig = v + '_comp_' + c + '_w_' + sam_title + '_' \
                                   + s
                        Plot(comp, scale_comp, cmap, title, name_fig,
                             dpi,
                             save, out_dir + s + '/' +
                             v + '/' + sam_component + '/')

                        print('Multiple individual plots...')
                        title = 'Events: ' + c + ' with ' \
                                + sam_title.upper() + ' - ' + v + ' - ' + s
                        name_fig = v + '_Events_' + c + '_w_' + sam_title \
                                   + '_' + s
                        Subplots(variable_selected, scale, cmap,
                                 title, save, dpi, name_fig, out_dir + s + '/' +
                                 v + '/' + sam_component + '/')

# -----------------------------------------------------------------------------#
print('#######################################################################')
print('done')
print('out_dir = ' + out_dir)
print('#######################################################################')
################################################################################