"""
Funciones de ploteo y auxiliares
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import cartopy.feature
import string
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")

from funciones.utils import MakeMask

# ---------------------------------------------------------------------------- #
def aux_PlotScatter_ParIndex(case, idx1_name, idx2_name,
                             idx1_sd, idx2_sd, by_r=False,
                             open_idx1=False, open_idx2=False,
                             cases_dir='', index_dir='',
                             remove_cfsv2_str=True):

    if remove_cfsv2_str:
        case = case.replace("CFSv2_", "")

    aux_var_name = 'index'
    try:
        aux = xr.open_dataset(f'{cases_dir}{idx1_name}_{case}').__mul__(1/idx1_sd)
        aux = aux.rename({list(aux.data_vars)[0]:aux_var_name})
        aux2 = xr.open_dataset(f'{cases_dir}{idx2_name}_{case}').__mul__(1/idx2_sd)
        aux2 = aux2.rename({list(aux2.data_vars)[0]:aux_var_name})
        check = True
    except:
        print('No se pueden abrir:')
        print(f'{cases_dir}{idx1_name}_{case}')
        print(f'{cases_dir}{idx2_name}_{case}')
        check = False

    if check is True:
        if by_r:
            if open_idx1:
                idx1 = (xr.open_dataset(
                    f'{index_dir}{idx1_name}_SON_Leads_r_CFSv2.nc')
                        .__mul__(1 / idx1_sd))

                aux3 = idx1.sel(r=aux.r, time=aux.time)
                aux3 = aux3.rename({list(aux3.data_vars)[0]: aux_var_name})

                if len(np.where(aux3.L.values == aux.L.values)[0]):
                    return aux[aux_var_name].values.round(2), \
                        aux3[aux_var_name].values.round(2)
                else:
                    print('Error: CASES')
                    return [], []

            if open_idx2:
                idx2 = (xr.open_dataset(
                    f'{index_dir}{idx2_name}_SON_Leads_r_CFSv2.nc')
                        .__mul__(1 / idx2_sd))

                aux3 = idx2.sel(r=aux.r, time=aux.time)
                aux3 = aux3.rename({list(aux3.data_vars)[0]: aux_var_name})

                if len(np.where(aux3.L.values == aux.L.values)[0]):
                    return aux[aux_var_name].values.round(2), \
                        aux3[aux_var_name].values.round(2)
                else:
                    print('Error: CASES')
                    return [], []
        else:
            aux2 = aux2.sel(time=aux2.time.isin([aux.time.values]))

            if len(aux2.time) == len(aux.time):
                return aux[aux_var_name].values.round(2), \
                    aux2[aux_var_name].values.round(2)
            else:
                print('Error: CASES')
                return [], []

    else:
        return [], []

def PlotScatter(idx1_name, idx2_name, idx1_sd, idx2_sd, save=False,
                out_dir=None,  name_fig='fig',
                cases_dir='', index_dir='', add_all=False,
                remove_cfsv2_str=False):
    
    if save:
        dpi = 300
    else:
        dpi = 100

    idx1_name = idx1_name#.upper()
    idx1 = idx1_name.lower()
    idx2_name = idx2_name#.upper()
    idx2 = idx2_name.lower()

    case = f'CFSv2_neutros_SON.nc'
    idx1_neutros, idx2_neutros = aux_PlotScatter_ParIndex(
        case=case, idx1_name=idx1_name, idx2_name=idx2_name,
        idx1_sd=idx1_sd, idx2_sd=idx2_sd, by_r=False,
        open_idx1=False, open_idx2=False,
        cases_dir=cases_dir, index_dir=index_dir,
        remove_cfsv2_str=remove_cfsv2_str)

    case = f'CFSv2_simultaneos_dobles_{idx1}_{idx2}_pos_SON.nc'
    idx1_sim_pos, idx2_sim_pos = aux_PlotScatter_ParIndex(
        case=case, idx1_name=idx1_name, idx2_name=idx2_name,
        idx1_sd=idx1_sd, idx2_sd=idx2_sd, by_r=False,
        open_idx1=False, open_idx2=False,
        cases_dir=cases_dir, index_dir=index_dir,
        remove_cfsv2_str=remove_cfsv2_str)

    case = f'CFSv2_simultaneos_dobles_{idx1}_{idx2}_neg_SON.nc'
    idx1_sim_neg, idx2_sim_neg = aux_PlotScatter_ParIndex(
        case=case, idx1_name=idx1_name, idx2_name=idx2_name,
        idx1_sd=idx1_sd, idx2_sd=idx2_sd, by_r=False,
        open_idx1=False, open_idx2=False,
        cases_dir=cases_dir, index_dir=index_dir,
        remove_cfsv2_str=remove_cfsv2_str)

    case = f'CFSv2_simultaneos_dobles_op_{idx1}_pos_{idx2}_neg_SON.nc'
    idx1_pos_idx2_neg, idx2_in_idx1_pos_idx2_neg = aux_PlotScatter_ParIndex(
        case=case, idx1_name=idx1_name, idx2_name=idx2_name,
        idx1_sd=idx1_sd, idx2_sd=idx2_sd, by_r=False,
        open_idx1=False, open_idx2=False,
        cases_dir=cases_dir, index_dir=index_dir,
        remove_cfsv2_str=remove_cfsv2_str)

    case = f'CFSv2_simultaneos_dobles_op_{idx1}_neg_{idx2}_pos_SON.nc'
    idx1_neg_idx2_pos, idx2_in_idx1_neg_idx2_pos = aux_PlotScatter_ParIndex(
        case=case, idx1_name=idx1_name, idx2_name=idx2_name,
        idx1_sd=idx1_sd, idx2_sd=idx2_sd, by_r=False,
        open_idx1=False, open_idx2=False,
        cases_dir=cases_dir, index_dir=index_dir,
        remove_cfsv2_str=remove_cfsv2_str)

    case = f'CFSv2_puros_{idx1}_neg_SON.nc'
    idx1_puros_neg, idx2_in_idx1_puros_neg = aux_PlotScatter_ParIndex(
        case=case, idx1_name=idx1_name, idx2_name=idx2_name,
        idx1_sd=idx1_sd, idx2_sd=idx2_sd, by_r=False,
        open_idx1=False, open_idx2=False,
        cases_dir=cases_dir, index_dir=index_dir,
        remove_cfsv2_str=remove_cfsv2_str)

    case = f'CFSv2_puros_{idx1}_pos_SON.nc'
    idx1_puros_pos, idx2_in_idx1_puros_pos = aux_PlotScatter_ParIndex(
        case=case, idx1_name=idx1_name, idx2_name=idx2_name,
        idx1_sd=idx1_sd, idx2_sd=idx2_sd, by_r=False,
        open_idx1=False, open_idx2=False,
        cases_dir=cases_dir, index_dir=index_dir,
        remove_cfsv2_str=remove_cfsv2_str)


    case = f'CFSv2_puros_{idx2}_pos_SON.nc'
    idx2_puros_pos, idx1_in_idx2_puros_pos = aux_PlotScatter_ParIndex(
        case=case, idx1_name=idx2_name, idx2_name=idx1_name,
        idx1_sd=idx2_sd, idx2_sd=idx1_sd, by_r=False,
        open_idx1=False, open_idx2=False,
        cases_dir=cases_dir, index_dir=index_dir,
        remove_cfsv2_str=remove_cfsv2_str)

    case = f'CFSv2_puros_{idx2}_neg_SON.nc'
    idx2_puros_neg, idx1_in_idx2_puros_neg = aux_PlotScatter_ParIndex(
        case=case, idx1_name=idx2_name, idx2_name=idx1_name,
        idx1_sd=idx2_sd, idx2_sd=idx1_sd, by_r=False,
        open_idx1=False, open_idx2=False,
        cases_dir=cases_dir, index_dir=index_dir,
        remove_cfsv2_str=remove_cfsv2_str)

    if add_all is True:
        # todos
        case = f'CFSv2_todo_{idx1}_neg_SON.nc'
        idx1_todo_neg, idx2_in_idx1_todo_neg = aux_PlotScatter_ParIndex(
            case=case, idx1_name=idx1_name, idx2_name=idx2_name,
            idx1_sd=idx1_sd, idx2_sd=idx2_sd, by_r=False,
            open_idx1=False, open_idx2=False,
            cases_dir=cases_dir, index_dir=index_dir,
            remove_cfsv2_str=remove_cfsv2_str)

        case = f'CFSv2_todo_{idx1}_pos_SON.nc'
        idx1_todo_pos, idx2_in_idx1_todo_pos = aux_PlotScatter_ParIndex(
            case=case, idx1_name=idx1_name, idx2_name=idx2_name,
            idx1_sd=idx1_sd, idx2_sd=idx2_sd, by_r=False,
            open_idx1=False, open_idx2=False,
            cases_dir=cases_dir, index_dir=index_dir,
            remove_cfsv2_str=remove_cfsv2_str)

        case = f'CFSv2_todo_{idx2}_pos_SON.nc'
        idx2_todo_pos, idx1_in_idx2_todo_pos = aux_PlotScatter_ParIndex(
            case=case, idx1_name=idx2_name, idx2_name=idx1_name,
            idx1_sd=idx2_sd, idx2_sd=idx1_sd, by_r=False,
            open_idx1=False, open_idx2=False,
            cases_dir=cases_dir, index_dir=index_dir,
            remove_cfsv2_str=remove_cfsv2_str)

        case = f'CFSv2_todo_{idx2}_neg_SON.nc'
        idx2_todo_neg, idx1_in_idx2_todo_neg = aux_PlotScatter_ParIndex(
            case=case, idx1_name=idx2_name, idx2_name=idx1_name,
            idx1_sd=idx2_sd, idx2_sd=idx1_sd, by_r=False,
            open_idx1=False, open_idx2=False,
            cases_dir=cases_dir, index_dir=index_dir,
            remove_cfsv2_str=remove_cfsv2_str)

    in_label_size = 13
    label_legend_size = 12
    tick_label_size = 11
    scatter_size_fix = 3
    fig, ax = plt.subplots(dpi=dpi, figsize=(7.08661, 7.08661))

    # neutros
    ax.scatter(x=idx1_neutros, y=idx2_neutros, marker='o',
               label=f'{idx1_name} vs. {idx2_name}',
               s=30 * scatter_size_fix, edgecolor='k', color='dimgray',
               alpha=1)
    # idx1 puros
    ax.scatter(x=idx1_puros_pos, y=idx2_in_idx1_puros_pos, marker='o',
               s=30 * scatter_size_fix, edgecolor='k', color='#8B1E1E',
               alpha=1, label=f'{idx1_name}+')
    ax.scatter(x=idx1_puros_neg, y=idx2_in_idx1_puros_neg, marker='o',
               s=30 * scatter_size_fix, edgecolor='k', color='#7CCD73',
               alpha=1, label=f'{idx1_name}-')

    # idx2 puros
    ax.scatter(x=idx1_in_idx2_puros_pos, y=idx2_puros_pos, marker='o',
               s=30 * scatter_size_fix, edgecolor='k', color='navy',
               alpha=1,
               label=f'{idx2_name}+')
    ax.scatter(x=idx1_in_idx2_puros_neg, y=idx2_puros_neg, marker='o',
               s=30 * scatter_size_fix, edgecolor='k', color='#DE00FF',
               alpha=1, label=f'{idx2_name}-')

    # sim
    ax.scatter(x=idx1_sim_pos, y=idx2_sim_pos, marker='o',
               s=30 * scatter_size_fix,
               edgecolor='k', color='#FF5B12', alpha=1,
               label=f'{idx1_name}+ & {idx2_name}+')
    ax.scatter(x=idx1_sim_neg, y=idx2_sim_neg, marker='o',
               s=30 * scatter_size_fix,
               edgecolor='k', color='#63A6FF', alpha=1,
               label=f'{idx1_name}- & {idx2_name}-')

    # sim opp. sing
    ax.scatter(x=idx1_pos_idx2_neg, y=idx2_in_idx1_pos_idx2_neg, marker='o',
               s=30 * scatter_size_fix, edgecolor='k', color='#FF9232',
               alpha=1,
               label=f'{idx1_name}+ & {idx2_name}-')
    ax.scatter(x=idx1_neg_idx2_pos, y=idx2_in_idx1_neg_idx2_pos, marker='o',
               s=30 * scatter_size_fix, edgecolor='k', color='gold',
               alpha=1,
               label=f'{idx1_name}- & {idx2_name}+')

    if add_all is True:
        # todos
        ax.scatter(x=idx1_todo_pos, y=idx2_in_idx1_todo_pos, marker='x',
                   s=3 * scatter_size_fix, edgecolor='k', color='k', alpha=0.5)

        ax.scatter(x=idx1_todo_neg, y=idx2_in_idx1_todo_neg, marker='x',
                   s=3* scatter_size_fix, edgecolor='k', color='k', alpha=0.5)

        ax.scatter(x=idx1_in_idx2_todo_pos, y=idx2_todo_pos, marker='x',
                   s=3 * scatter_size_fix, edgecolor='k', color='k', alpha=0.5)

        ax.scatter(x=idx1_in_idx2_todo_neg, y=idx2_todo_neg, marker='x',
                   s=3 * scatter_size_fix, edgecolor='k', color='k', alpha=0.5)

        ax.scatter(x=idx1_neutros, y=idx2_neutros, marker='x',
                   s=3 * scatter_size_fix, edgecolor='k', color='k', alpha=0.5)

    ax.legend(loc=(.01, .57), fontsize=label_legend_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size, pad=1)
    ax.set_ylim((-5, 5))
    ax.set_xlim((-5, 5))
    ax.axhspan(-.5 , .5,
               alpha=0.2, color='black', zorder=0)
    ax.axvspan(-.5 , .5,
               alpha=0.2, color='black', zorder=0)
    ax.set_xlabel(f'{idx1_name}', size=in_label_size)
    ax.set_ylabel(f'{idx2_name}', size=in_label_size)
    ax.text(-4.9, 4.6, f'{idx2_name}+/{idx1_name}-', dict(size=in_label_size))
    ax.text(-.4, 4.6,  f'{idx2_name}+', dict(size=in_label_size))
    ax.text(+3.4, 4.6, f'{idx2_name}+/{idx1_name}+', dict(size=in_label_size))
    ax.text(+4.2, -.1,  f'{idx1_name}+', dict(size=in_label_size))
    ax.text(+3.4, -4.9, f'{idx2_name}-/{idx1_name}+', dict(size=in_label_size))
    ax.text(-.4, -4.9, f'{idx2_name}-', dict(size=in_label_size))
    ax.text(-4.9, -4.9, f'{idx2_name}-/{idx1_name}-', dict(size=in_label_size))
    ax.text(-4.9, -.1, f'{idx1_name}-', dict(size=in_label_size))
    plt.tight_layout()

    if save:
        plt.savefig(f'{out_dir}{name_fig}.png', dpi=dpi, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()
        
# ---------------------------------------------------------------------------- #

def set_mapa_param(map):
    # mapa
    if map.upper() == 'HS':
        extent = [0, 359, -80, 20]
        xticks = np.arange(0, 360, 60)
        yticks = np.arange(-80, 20, 20)
        lon_localator = 60
    elif map.upper() == 'TR':
        extent = [45, 270, -20, 20]
        xticks = np.arange(0, 360, 60)
        np.arange(-80, 20, 20)
        lon_localator = 60
    elif map.upper() == 'HS_EX':
        xticks = np.arange(0, 360, 60)
        extent = [0, 359, -65, -20]
        lon_localator = 60
    elif map.upper() == 'SA':
        extent = [275, 330, -60, 20]
        yticks = np.arange(-60, 15+1, 20)
        xticks = np.arange(275, 330+1, 20)
        lon_localator = 20
    else:
        print(f"Mapa {map} no seteado")
        return

    return xticks, yticks, lon_localator, extent

def remove_0_level(levels):
    try:
        if isinstance(levels, np.ndarray):
            levels = levels[levels != 0]
        else:
            levels.remove(0)
    except:
        pass
    return levels

def get_xr_values(data):
    try:
        var = list(data.data_vars)[0]
        data = data[var].values
    except:
        data = data.values

    return data

def check_data(data):
    check_no_zero = data.mean().values != 0

    try:
        check_no_nan = ~np.isnan(data.mean().values)
    except:
        var_ctn = list(data.data_vars)[0]
        check_no_nan = ~np.isnan(data[var_ctn].mean().values)

    if check_no_zero and check_no_nan:
        check = True
    else:
        check = False

    return check

def Plot_ENSO_IOD_SAM_comp(data, levels, cmap, titles, namefig, map,
                           save, out_dir, data_ctn=None,
                           levels_ctn=None, color_ctn=None,
                           plots_titles=None,
                           high=2, width = 7.08661,
                           cbar_pos='H', plot_step=3,
                           pdf=True, ocean_mask=False,
                           data_ctn_no_ocean_mask=False,
                           sig_data=None, hatches=None,
                           num_cols=3, num_rows = 4):

    if save:
        dpi=300
    else:
        dpi=100
    # num_cols = 3
    # num_rows = 4
    high = high
    plots = data.plots.values
    crs_latlon = ccrs.PlateCarree()

    xticks, yticks, lon_localator, extent = set_mapa_param(map)

    # plot
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, high * num_rows),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'wspace': 0.05, 'hspace': 0.01})

    i2 = 0
    for i, (ax, plot) in enumerate(zip(axes.flatten(), plots)):
        remove_axes = False

        if i in [2, 5, 8, 11]:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_yticks(yticks, crs=crs_latlon)
            ax.tick_params(width=0.3, pad=1)
            lat_formatter = LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)

        if i in [9, 10, 11]:
            ax.set_xticks(xticks, crs=crs_latlon)
            ax.tick_params(width=0.3, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.tick_params(labelsize=3)

        ax.tick_params(width=0.5, pad=1, labelsize=4)

        if plot > 0:
            ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i2]}) "
                                   f"$N={titles[plot]}$ - "
                                   f"{plots_titles[plot]}",
                    transform=ax.transAxes, size=6)

        # Plot --------------------------------------------------------------- #
        # Contour ------------------------------------------------------------ #
        aux_ctn = data_ctn.sel(plots=plot)
        if check_data(aux_ctn):
            if data_ctn is not None:
                if levels_ctn is None:
                    levels_ctn = levels.copy()
                    levels_ctn = remove_0_level(levels_ctn)

            if ocean_mask is True and data_ctn_no_ocean_mask is False:
                mask_ocean = MakeMask(aux_ctn)
                aux_ctn = aux_ctn * mask_ocean.mask

            aux_ctn_var = get_xr_values(aux_ctn)

            ax.contour(data_ctn.lon.values[::plot_step],
                       data_ctn.lat.values[::plot_step],
                       aux_ctn_var[::plot_step, ::plot_step],
                       linewidths=0.4,
                       levels=levels_ctn, transform=crs_latlon,
                       colors=color_ctn)

        # significance ------------------------------------------------------- #
        if sig_data is not None:
            aux_sig_points = sig_data.sel(plots=plot)
            if aux_sig_points.mean().values != 0:

                if ocean_mask is True:
                    mask_ocean = MakeMask(aux_sig_points)
                    aux_sig_points = aux_sig_points * mask_ocean.mask

                # hatches = '....'
                colors_l = ['k', 'k']

                comp_sig_var = get_xr_values(comp_sig_var)

                cs = ax.contourf(aux_sig_points.lon,
                                 aux_sig_points.lat,
                                 comp_sig_var,
                                 transform=crs_latlon, colors='none',
                                 hatches=[hatches, hatches], extend='lower',
                                 zorder=5)

                for i3, collection in enumerate(cs.collections):
                    collection.set_edgecolor(colors_l[i3 % len(colors_l)])

                for collection in cs.collections:
                    collection.set_linewidth(0.)

        # Contourf ----------------------------------------------------------- #
        aux = data.sel(plots=plot)
        aux_var = get_xr_values(aux)
        if check_data(aux):

            i2 += 1
            if ocean_mask is True:
                mask_ocean = MakeMask(aux)
                aux_var = aux_var * mask_ocean.mask

            im = ax.contourf(aux.lon.values[::plot_step],
                             aux.lat.values[::plot_step],
                             aux_var[::plot_step, ::plot_step],
                             levels=levels,
                             transform=crs_latlon, cmap=cmap, extend='both', zorder=1)

            ax.add_feature(cartopy.feature.LAND, facecolor='white',
                           linewidth=0.5)
            ax.coastlines(color='k', linestyle='-', alpha=1, linewidth=0.2,
                          resolution='110m')
            gl = ax.gridlines(draw_labels=False, linewidth=0.1,
                              linestyle='-', zorder=20)
            gl.ylocator = plt.MultipleLocator(20)
            gl.xlocator = plt.MultipleLocator(lon_localator)

        else:
            remove_axes = True

        if remove_axes:
            ax.axis('off')

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    # ax_ref = axes[2, 0]
    # y_line = ax_ref.get_position().y0
    # fig.add_artist(plt.Line2D([0, 1], [y_line-0.04, y_line-0.04],
    #                           color='k', lw=1.5, transform=fig.transFigure))

    # cbar_pos = 'H'
    if cbar_pos.upper() == 'H':
        pos = fig.add_axes([0.261, 0, 0.5, 0.02])
        cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='horizontal')
        cb.ax.tick_params(labelsize=4, pad=1)
        fig.subplots_adjust(bottom=0.05, wspace=0, hspace=0.25, left=0, right=1,
                            top=1)

    elif cbar_pos.upper() == 'V':
        pos = fig.add_axes([0.95, 0.2, 0.02, 0.5])
        cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='vertical')
        cb.ax.tick_params(labelsize=4, pad=1)
        fig.subplots_adjust(bottom=0, wspace=0.5, hspace=0.25, left=0.02,
                            right=0.9, top=1)
    else:
        print(f"cbar_pos {cbar_pos} no valido")

    if save:
        if pdf:
            plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
        else:
            plt.savefig(f"{out_dir}{namefig}.jpg", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ---------------------------------------------------------------------------- #

def Plot_Contourf_simple(data, levels, cmap, map, title, namefig,
                         save, out_dir, data_ctn=None,
                         levels_ctn=None, color_ctn=None,
                         high=2, width = 7,
                         cbar_pos='H', plot_step=3,
                         pdf=True, ocean_mask=False,
                         data_ctn_no_ocean_mask=False):

    dpi = 300 if save else 100
    crs_latlon = ccrs.PlateCarree()

    xticks, yticks, lon_localator, extent = set_mapa_param(map)

    # figura + ejes
    fig, ax = plt.subplots(
        figsize=(width, high),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
    )

    # Contour ---------------------------------------------------------------- #
    if data_ctn is not None:
        aux_ctn = data_ctn
        if levels_ctn is None:
            levels_ctn = levels.copy()
            levels_ctn = remove_0_level(levels_ctn)

        if ocean_mask is True and data_ctn_no_ocean_mask is False:
            mask_ocean = MakeMask(aux_ctn)
            aux_ctn = aux_ctn * mask_ocean.mask

        aux_ctn_var = get_xr_values(aux_ctn)

        ax.contour(data_ctn.lon.values[::plot_step],
                   data_ctn.lat.values[::plot_step],
                   aux_ctn_var[::plot_step, ::plot_step],
                   linewidths=0.4,
                   levels=levels_ctn, transform=crs_latlon,
                   colors=color_ctn)
    # Contourf ----------------------------------------------------------- #
    aux_var = get_xr_values(data)
    if ocean_mask:
        mask_ocean = MakeMask(data)
        aux_var = aux_var * mask_ocean.mask

    im = ax.contourf(data.lon.values[::plot_step],
                     data.lat.values[::plot_step],
                     aux_var[::plot_step, ::plot_step],
                     levels=levels,
                     transform=crs_latlon, cmap=cmap,
                     extend='both', zorder=1)

    ax.add_feature(cartopy.feature.LAND, facecolor='white', linewidth=0.5)
    ax.coastlines(color='k', linestyle='-', alpha=1, linewidth=0.2,
                  resolution='110m')
    gl = ax.gridlines(draw_labels=False, linewidth=0.1, linestyle='-',
                      zorder=20)
    gl.ylocator = plt.MultipleLocator(20)
    gl.xlocator = plt.MultipleLocator(lon_localator)

    # -----------------
    ax.add_feature(cartopy.feature.LAND, facecolor='white',
                   linewidth=0.5)
    ax.coastlines(color='k', linestyle='-', alpha=1,
                  linewidth=0.2,
                  resolution='110m')
    if map.upper() == 'SA':
        ax.add_feature(cartopy.feature.BORDERS, alpha=1,
                       linestyle='-', linewidth=0.2, color='k')
    gl = ax.gridlines(draw_labels=False, linewidth=0.1, linestyle='-',
                      zorder=20)
    gl.ylocator = plt.MultipleLocator(20)
    ax.set_xticks(xticks, crs=crs_latlon)
    ax.set_yticks(yticks, crs=crs_latlon)
    ax.tick_params(width=0.5, pad=1)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=4)
    ax.set_extent(extent, crs=crs_latlon)

    ax.set_aspect('equal')

    # ----------------

    ax.set_title(title, fontsize=8)

    # cbar_pos = 'H'
    if cbar_pos.upper() == 'H':
        pos = fig.add_axes([0.261, 0, 0.5, 0.02])
        cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='horizontal')
        cb.ax.tick_params(labelsize=4, pad=1)
        fig.subplots_adjust(bottom=0.05, wspace=0, hspace=0.25, left=0, right=1,
                            top=1)

    elif cbar_pos.upper() == 'V':
        pos = fig.add_axes([0.92, 0.2, 0.02, 0.5])
        cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='vertical')
        cb.ax.tick_params(labelsize=4, pad=1)
        fig.subplots_adjust(bottom=0, wspace=0.5, hspace=0.25, left=0.02,
                            right=0.9, top=1)
    else:
        print(f"cbar_pos {cbar_pos} no valido")

    if save:
        if pdf:
            plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
        else:
            plt.savefig(f"{out_dir}{namefig}.jpg", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
# ---------------------------------------------------------------------------- #