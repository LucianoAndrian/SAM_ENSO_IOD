"""
Composiciones de HGT200 a partir de los outputs de
ENSO_IOD_CFSv2_preSELECT_HGT.py
"""
# ---------------------------------------------------------------------------- #
save = True
save_nc = False
# ---------------------------------------------------------------------------- #
cases_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cases_fields/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cfsv2/composites/'
#out_dir2 = '/pikachu/datos/luciano.andrian/esquemas/'
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from matplotlib import colors
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import os
import warnings
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
warnings.filterwarnings('ignore')
# ---------------------------------------------------------------------------- #
if save:
    dpi = 300
else:
    dpi = 100
# Funciones ####################################################################
def Plot(comp, comp_var, levels = np.linspace(-1,1,11),
         cmap='RdBu', dpi=100, save=True, step=1, name_fig='fig',
         title='title', color_map='grey'):

    import matplotlib.pyplot as plt
    levels_contour = levels.copy()
    levels_contour.remove(0)
    #comp_var = comp['var']
    fig = plt.figure(figsize=(9, 3.5), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([0,359, -80,20], crs_latlon)

    ax.contour(comp.lon[::step], comp.lat[::step], comp_var[::step, ::step],
               linewidths=.8, levels=levels_contour, transform=crs_latlon,
               colors='black')

    im = ax.contourf(comp.lon[::step], comp.lat[::step],
                     comp_var[::step, ::step],
                     levels=levels, transform=crs_latlon, cmap=cmap,
                     extend='both')
    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='white', edgecolor=color_map)
    ax.coastlines(color=color_map, linestyle='-', alpha=1)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.arange(0, 360, 30), crs=crs_latlon)
    ax.set_yticks(np.arange(-80, 20, 10), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)

    plt.title(title, fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

def Weights(data):
    weights = np.transpose(np.tile(np.cos(data.lat * np.pi / 180),
                                   (len(data.lon), 1)))
    data_w = data * weights
    return data_w

def PlotSST(comp, levels = np.linspace(-1,1,11), cmap='RdBu',
         dpi=100, save=True, step=1,
         name_fig='fig', title='title', color_map='#4B4B4B'):

    import matplotlib.pyplot as plt

    levels_contour = levels.copy()
    comp_var = comp['var']
    if isinstance(levels, np.ndarray):
        levels_contour = levels[levels != 0]
    else:
        levels_contour.remove(0)
    fig = plt.figure(figsize=(7, 2), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([50,270, -20,20], crs_latlon)

    im = ax.contourf(comp.lon[::step], comp.lat[::step],
                     comp_var[::step, ::step],
                     levels=levels, transform=crs_latlon, cmap=cmap,
                     extend='both')
    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='lightgrey',
                   edgecolor=color_map)
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
    ax.coastlines(color=color_map, linestyle='-', alpha=1)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.arange(50, 270, 60), crs=crs_latlon)
    ax.set_yticks(np.arange(-20, 40, 20), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)

    plt.title(title, fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()


def PlotPP_T(comp, levels, cmap, title, name_fig, dpi, save, out_dir):
    import matplotlib.pyplot as plt
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
################################################################################
seasons = ['JJA','SON']
seasons = ['SON']

cases = ['sim_pos', 'sim_neg', #triple sim
         'dmi_puros_pos', 'dmi_puros_neg', # puros, sin los otros dos
         'n34_puros_pos', 'n34_puros_neg', # puros, sin los otros dos
         'sam_puros_pos', 'sam_puros_neg', # puros, sin los otros dos
         'dmi_pos_n34_pos_sam_neg', # simultaneo signos opuestos
         'dmi_pos_n34_neg_sam_pos', # simultaneo signos opuestos
         'dmi_pos_n34_neg_sam_neg', # simultaneo signos opuestos
         'dmi_neg_n34_pos_sam_neg', # simultaneo signos opuestos
         'dmi_neg_n34_pos_sam_pos', # simultaneo signos opuestos
         'dmi_neg_n34_neg_sam_pos', # simultaneo signos opuestos
         'dmi_pos', 'dmi_neg', # todos los de uno, sin importar el resto
         'n34_pos', 'n34_neg', # todos los de uno, sin importar el resto
         'sam_pos', 'sam_neg', # todos los de uno, sin importar el resto
         #'neutros', # todos neutros
         'dmi_sim_n34_pos_wo_sam', # dos eventos simultaneos sin el otro
         'dmi_sim_sam_pos_wo_n34', # dos eventos simultaneos sin el otro
         'n34_sim_sam_pos_wo_dmi', # dos eventos simultaneos sin el otro
         #'n34_sim_dmi_pos_wo_sam', # dos eventos simultaneos sin el otro
         #'sam_sim_n34_pos_wo_dmi', # dos eventos simultaneos sin el otro
         #'sam_sim_dmi_pos_wo_n34', # dos eventos simultaneos sin el otro
         'dmi_sim_n34_neg_wo_sam', # dos eventos simultaneos sin el otro
         'n34_sim_sam_neg_wo_dmi', # dos eventos simultaneos sin el otro
         #'n34_sim_dmi_neg_wo_sam', # dos eventos simultaneos sin el otro
         #'sam_sim_n34_neg_wo_dmi', # dos eventos simultaneos sin el otro
         'sam_sim_dmi_neg_wo_n34', # dos eventos simultaneos sin el otro]
         #'dmi_sim_sam_neg_wo_n34', # dos eventos simultaneos sin el otro
         'dmi_pos_n34_neg_wo_sam',
         'dmi_pos_sam_neg_wo_n34',
         'n34_pos_sam_neg_wo_dmi',
         'n34_pos_dmi_neg_wo_sam',
         'sam_pos_dmi_neg_wo_n34',
         'sam_pos_n34_neg_wo_dmi']
         #'dmi_neg_n34_pos_wo_sam',
         #'dmi_neg_sam_pos_wo_n34',
         #'n34_neg_sam_pos_wo_dmi',
         #'n34_neg_dmi_pos_wo_sam']#,
         #'sam_neg_dmi_pos_wo_n34']#,
         #'sam_neg_n34_pos_wo_dmi']#,
         #'sam', 'dmi', 'n34']

title_case = cases

cbar = colors.ListedColormap(['#9B1C00','#B9391B', '#CD4838', '#E25E55',
                              '#F28C89', '#FFCECC',
                              'white',
                              '#B3DBFF', '#83B9EB', '#5E9AD7', '#3C7DC3',
                              '#2064AF', '#014A9B'][::-1])
cbar.set_over('#641B00')
cbar.set_under('#012A52')
cbar.set_bad(color='white')

cbar_snr = colors.ListedColormap(['#070B4F','#2E07AC', '#387AE4' ,'#6FFE9B',
                                  '#FFFFFF',
                                  '#FFFFFF', '#FFFFFF',
                                  '#FEB77E','#CA3E72','#782281','#251255'])
cbar_snr.set_over('#251255')
cbar_snr.set_under('#070B4F')
cbar_snr.set_bad(color='white')

cbar_sst = colors.ListedColormap(['#B9391B', '#CD4838', '#E25E55', '#F28C89',
                                  '#FFCECC', 'white', '#B3DBFF', '#83B9EB',
                                  '#5E9AD7', '#3C7DC3', '#2064AF'][::-1])
cbar_sst.set_over('#9B1C00')
cbar_sst.set_under('#014A9B')
cbar_sst.set_bad(color='white')

cbar_pp = colors.ListedColormap(['#003C30', '#004C42', '#0C7169', '#79C8BC',
                                 '#B4E2DB',
                                 'white',
                                 '#F1DFB3', '#DCBC75', '#995D13','#6A3D07',
                                 '#543005'][::-1])
cbar_pp.set_under('#3F2404')
cbar_pp.set_over('#00221A')
cbar_pp.set_bad(color='white')

cbar_snr_pp = colors.ListedColormap(['#1E6D5A' ,'#52C39D',
                                     '#6FFE9B','#FFFFFF',
                                  '#FFFFFF','#FFFFFF',
                                  '#DCBC75', '#995D13','#6A3D07'][::-1])
cbar_snr_pp.set_under('#6A3D07')
cbar_snr_pp.set_over('#1E6D5A')
cbar_snr_pp.set_bad(color='white')
# ---------------------------------------------------------------------------- #
# HGT ------------------------------------------------------------------------ #
print('z200')
scale = [-300,-250,-200,-150,-100,-50,-25,0,25,50,100,150,200,250,300]
scale_cont= [-300,-250,-200,-150,-100,-50,-25,25,50,100,150,200,250,300]
# ---------------------------------------------------------------------------- #
for s in seasons:
    neutro = xr.open_dataset(
        cases_dir + 'hgt_neutros_' + s + '.nc').rename({'hgt':'var'})
    neutro = Weights(neutro.__mul__(9.80665))

    for c_count, c in enumerate(cases):
        case = xr.open_dataset(
            cases_dir + 'hgt_' + c + '_' + s + '.nc').rename({'hgt':'var'})
        case = Weights(case.__mul__(9.80665))

        try:
            num_case = len(case.time)
            comp = case.mean('time') - neutro.mean('time')
            comp_var = comp['var']
            Plot(comp, comp_var, levels=scale,
                 cmap=cbar, dpi=dpi, step=1, name_fig='hgt_' + c + '_' + s,
                 title='Mean Composite - CFSv2 - ' + s + '\n' +
                       title_case[c_count] + '\n' + ' ' + 'HGT 200hPa'
                       + ' - ' + 'Cases: ' + str(num_case),
                 save=save)

            spread = case - comp
            spread = spread.std('time')
            snr = comp / spread

            if save_nc & (s=='SON'):
                print('save_nc no yet set')
                #snr.to_netcdf(out_dir2 + 'SNR_hgt_CFSv2' + c + '_' + s + '.nc')

            Plot(snr, snr['var'],
                 levels = [-1,-.8,-.6,-.5,-.1,0,0.1,0.5,0.6,0.8,1],
                 cmap=cbar_snr, dpi=dpi, step=1,
                 name_fig='SNR_hgt_' + c + '_' + s,
                 title='Signal-to-noise ratio - CFSv2 - ' + s + '\n' +
                       title_case[c_count] + '\n' + ' ' + 'HGT 200hPa'
                       + ' - ' + 'Cases: ' + str(num_case),
                 save=save)

        except:
            print('Error in ' + c + ' - ' + s)

# ---------------------------------------------------------------------------- #
# SST ------------------------------------------------------------------------ #
print('sst')
scale = [-1.5, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 1.5]
for s in seasons:
    neutro = xr.open_dataset(
        cases_dir + 'sst_neutros_' + s + '.nc').rename({'sst':'var'})

    for c_count, c in enumerate(cases):
        case = xr.open_dataset(
            cases_dir + 'sst_' + c + '_' + s + '.nc').rename({'sst':'var'})

        try:
            num_case = len(case.time)
            comp = case.mean('time') - neutro.mean('time')

            PlotSST(comp, levels=scale, cmap=cbar_sst, dpi=dpi, step=1,
                 name_fig='sst_' + c + '_' + s,
                 title='Mean Composite - CFSv2 - ' + s + '\n' +
                       title_case[c_count] + '\n' + ' ' + 'SST'
                       + ' - ' + 'Cases: ' + str(num_case),
                 save=save)
        except:
            print('Error in ' + c + ' - ' + s)
# ---------------------------------------------------------------------------- #
# PP T ----------------------------------------------------------------------- #
print('pp')
from ENSO_IOD_Funciones import MakeMask
scale = np.linspace(-45, 45, 13)
scale_snr = [-1,-.5,-.25,0,0.25,0.5,1]
for s in seasons:
    neutro = xr.open_dataset(
        cases_dir + 'pp_neutros_' + s + '.nc').rename({'prec':'var'})
    neutro *= 30
    for c_count, c in enumerate(cases):
        case = xr.open_dataset(
            cases_dir + 'pp_' + c + '_' + s + '.nc').rename({'prec':'var'})
        case *= 30
        try:
            num_case = len(case.time)
            comp = case.mean('time') - neutro.mean('time')
            comp *= MakeMask(comp, 'var')
            PlotPP_T(comp, levels=scale, cmap=cbar_pp, dpi=dpi,
                 name_fig='pp_' + c + '_' + s,
                 title='Mean Composite - CFSv2 - ' + s + '\n' +
                       title_case[c_count] + '\n' + ' ' + 'PP'
                       + ' - ' + 'Cases: ' + str(num_case),
                 save=save, out_dir=out_dir)

            spread = case - comp
            spread = spread.std('time')
            snr = comp / spread

            PlotPP_T(snr, levels=scale_snr, cmap=cbar_snr_pp, dpi=dpi,
                 name_fig='SNR_pp_' + c + '_' + s,
                 title='Mean Composite - CFSv2 - ' + s + '\n' +
                       title_case[c_count] + '\n' + ' ' + 'PP'
                       + ' - ' + 'Cases: ' + str(num_case),
                 save=save, out_dir=out_dir)

        except:
            print('Error in ' + c + ' - ' + s)