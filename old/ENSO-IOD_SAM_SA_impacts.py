"""
Impactos en pp y t en SA de eventos SAM-ENSO-IOD, individuales y por signo del
SAM y categoria del IOD
"""
################################################################################
dates_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/pp_t_anoms/'

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

    fig_size = (5, 6)
    extent= [270, 330, -60, 20]
    xticks = np.arange(270, 330, 10)
    yticks = np.arange(-60, 40, 20)

    comp_var = comp['var']

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent(extent, crs=crs_latlon)

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

    extent= [270, 330, -60, 20]
    crs_latlon = ccrs.PlateCarree()

    time_values = data_array.time.values
    time_steps = len(time_values)
    num_cols = 4
    # cantidad de filas necesarias
    num_rows = np.ceil(time_steps/num_cols).astype(int)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 6*num_rows),
                             subplot_kw={'projection': crs_latlon })

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

    pos = fig.add_axes([0.2, 0.05, 0.6, 0.01])
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

    sub_dir1 = out_dir + s + '/' + v.split('_')[0]
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

cbar_pp = colors.ListedColormap(['#003C30', '#004C42', '#0C7169', '#79C8BC',
                                 '#B4E2DB',
                                 'white',
                                 '#F1DFB3', '#DCBC75', '#995D13','#6A3D07',
                                 '#543005'][::-1])
cbar_pp.set_under('#3F2404')
cbar_pp.set_over('#00221A')
cbar_pp.set_bad(color='white')
################################################################################

cases_dmi = ['dmi_un_pos', 'dmi_sim_pos', 'dmi_un_neg', 'dmi_sim_neg']
cases_n34 = ['n34_un_pos', 'n34_sim_pos', 'n34_un_neg', 'n34_sim_neg']

variables_tpp = ['ppgpcc_w_c_d_1', 'tcru_w_c_d_0.25']
scales = [np.linspace(-45, 45, 13),# pp
          np.linspace(-1, 1 ,17)]  # t
cmap= [cbar_pp, cbar]
# for test
s = 'SON'
c = 'dmi_un_neg'
v = 'ppgpcc_w_c_d_1'
v_count = 0
sam_component = 'sam'

for cases in [cases_dmi, cases_n34]:
    for s in seasons:
        print(s)

        # by cases ----------------------------------------------------------------#
        for c_count, c in enumerate(cases):
            print(c)

            # for sam_component in ['sam', 'ssam', 'asam']:
            for sam_component in ['sam', 'asam']:

                # dataframe con las fechas, dmi y sam para cada iod case
                data = pd.read_csv(
                    dates_dir + c + '_vs_' + sam_component + '_' +
                    s + '.txt', sep='\t', parse_dates=['time'])

                # by variables ----------------------------------------------------#
                for v_count, v in enumerate(variables_tpp):
                    print(v)

                    variable = OpenObsDataSet(v + '_' + s)

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

                    # by sam sign -------------------------------------------------#
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

                        print(
                            'Plots ---------------------------------------------')
                        print('Plot comp')
                        title = 'Composite - ' + c + ' with ' + \
                                sam_title.upper() + '\n' + v + ' - ' + s
                        name_fig = v + '_comp_' + c + '_w_' + sam_title \
                                   + '_' + s
                        Plot(comp, scales[v_count], cmap[v_count], title,
                             name_fig,
                             dpi,
                             save, out_dir + s + '/' +
                             v.split('_')[0] + '/' + sam_component + '/')

                        print('Multiple individual plots...')
                        title = 'Events: ' + c + ' with ' + \
                                sam_title.upper() + ' - ' + v + ' - ' + s
                        name_fig = v + '_Events_' + c + '_w_' + sam_title +\
                                   '_' + s
                        Subplots(variable_selected, scales[v_count],
                                 cmap[v_count],
                                 title, save, dpi, name_fig, out_dir + s + '/' +
                                 v.split('_')[0] + '/' + sam_component + '/')


#------------------------------------------------------------------------------#
print('#######################################################################')
print('done')
print('out_dir = ' + out_dir)
print('#######################################################################')
################################################################################