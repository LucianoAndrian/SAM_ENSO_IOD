"""
Composiciones de HGT200 a partir de los outputs de
CFSv2_HGT_PreSelect.py
"""
# ---------------------------------------------------------------------------- #
save = False
save_nc = False
# ---------------------------------------------------------------------------- #
cases_dir = ('/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
             'IOD_hLat_cic/cases_fields_selected/')
out_dir = ('/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
             'IOD_hLat_cic/cfsv2_comps/')
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import xarray as xr
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
from Scales_Cbars import get_cbars

# ---------------------------------------------------------------------------- #
if save:
    dpi = 200
else:
    dpi = 100
# ---------------------------------------------------------------------------- #
def set_cases(indice1, indice2, todos=False):
    if todos:
        cases = [# puros
            f'{indice1}_puros_pos', f'{indice1}_puros_neg',
            f'{indice2}_puros_pos', f'{indice2}_puros_neg',
            # sim misma fase
            f'sim_pos_{indice1}-{indice2}',
            f'sim_neg_{indice1}-{indice2}',
            # fases opuestas
            f'{indice1}_pos_{indice2}_neg',
            f'{indice1}_neg_{indice2}_pos',
            f'neutros_{indice1}-{indice2}',  # neutros
            # restantes
            f'{indice1}_pos', f'{indice1}_neg',
            f'{indice2}_pos', f'{indice2}_neg']

    else:
        cases = [  # puros
            f'{indice1}_puros_pos', f'{indice1}_puros_neg',
            f'{indice2}_puros_pos', f'{indice2}_puros_neg',
            # sim misma fase
            f'sim_pos_{indice1}-{indice2}',
            f'sim_neg_{indice1}-{indice2}',
            # fases opuestas
            f'{indice1}_pos_{indice2}_neg',
            f'{indice1}_neg_{indice2}_pos',
            f'neutros_{indice1}-{indice2}']

    return cases

def Plot(comp, comp_var, levels = np.linspace(-1,1,11),
         cmap='RdBu', dpi=100, save=True, step=1, name_fig='fig',
         title='title', color_map='grey', waf=False, px=None, py=None,
         waf_scale=None, waf_qlimite=99, data_ref_waf=None, step_waf = 3):

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

    if waf:
        from numpy import ma

        Q60 = np.percentile(np.sqrt(np.add(np.power(px, 2), np.power(py, 2))),
                            0)
        M = np.sqrt(np.add(np.power(px, 2), np.power(py, 2))) < Q60
        # mask array
        px_mask = ma.array(px, mask=M)
        py_mask = ma.array(py, mask=M)

        QL = np.nanpercentile(np.sqrt(np.add(np.power(px, 2),
                                              np.power(py, 2))), waf_qlimite)
        M = np.sqrt(np.add(np.power(px, 2), np.power(py, 2))) > QL
        # mask array
        px_mask = ma.array(px_mask, mask=M)
        py_mask = ma.array(py_mask, mask=M)
        # plot vectors
        lons, lats = np.meshgrid(data_ref_waf.lon.values,
                                 data_ref_waf.lat.values)
        ax.quiver(lons[::step_waf, ::step_waf], lats[::step_waf, ::step_waf],
                  px_mask[0, ::step_waf, ::step_waf],
                  py_mask[0, ::step_waf, ::step_waf],
                  transform=crs_latlon, pivot='tail', width=1.5e-3,
                  headwidth=3, alpha=1,
                  headlength=2.5, color='k', scale=waf_scale)

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
# ---------------------------------------------------------------------------- #
cases = set_cases('DMI', 'U50')
seasons = ['MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND']
seasons = ['SON']

cmap = get_cbars('hgt')
cmap_snr = get_cbars('snr')

for s in seasons:
    neutro = (xr.open_dataset(f'{cases_dir}hgt_{cases[-1]}_{s}_05.nc')
              .__mul__(9.80665))

    for c in cases:
        case = (xr.open_dataset(f'{cases_dir}hgt_{c}_{s}_05.nc')
                .__mul__(9.80665))

        comp = case.mean('time') - neutro.mean('time')

        spread = case - comp
        spread = spread.std('time')
        snr = comp / spread

        # Ver tema normalizacion. como se hizo antes
        # Plot(comp, comp.hgt, levels=[-150, -50,0, 50, 150, 300],
        #      cmap=cmap, dpi=dpi, save=save, step=1,
        #      name_fig='fig',
        #      title='title',
        #      color_map='grey')





# ---------------------------------------------------------------------------- #