import pandas as pd
import numpy as np
import statsmodels.api as sm
import xarray as xr

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

def identify_lags(lags):
    lag_target = lags[0]
    indices_lags = lags[1:]
    return lag_target, indices_lags

def Setlag(indice, lag, variable_target, years_to_remove):
    indice = indice.sel(time=indice.time.dt.month.isin(lag))
    indice = indice.sel(
        time=indice.time.dt.year.isin(variable_target.time.dt.year))
    indice = indice.sel(time=~indice.time.dt.year.isin(years_to_remove))
    indice = indice / indice.std()

    return indice

def SetLag_to_ActorList(variable_target, month_target, indices, lags,
                     years_to_remove, verbose=1):

    # Seteo de variable target
    variable_target = variable_target.sel(
        time=variable_target.time.dt.month.isin([month_target]))
    if verbose > 1: print('month_target seteado')
    variable_target = variable_target / variable_target.std()
    if verbose > 1: print('normalizado')
    variable_target = variable_target.sel(
        time=~variable_target.time.dt.month.isin(years_to_remove))
    if verbose > 1: print('years_to_remove ok')

    actor_list = {}
    for (indice_name, indice), lag in zip(indices.items(), lags):
        if verbose > 0: print(f'indice: {indice_name} - lag: {lag}')
        actor_list[indice_name] = Setlag(indice, lag, variable_target,
                                         years_to_remove)
    return variable_target, actor_list

def regre(series, intercept, coef=0, filter_significance=True, alpha=1):
    df = pd.DataFrame(series)
    if intercept:
        X = sm.add_constant(df[df.columns[1:]])
    else:
        X = df[df.columns[1:]]
    y = df[df.columns[0]]

    model = sm.OLS(y, X).fit()
    coefs_results = model.params
    p_values = model.pvalues
    # t_values = model.tvalues

    results = {}
    # p_val = {}
    # t_val = {}
    for col in df.columns[1:]:
        if filter_significance:
            if p_values[col] <= alpha:
                results[col] = coefs_results[col]
            else:
                results[col] = None
        else:
            results[col] = coefs_results[col]

        # p_val[col] = p_values[col]
        # t_val[col] = t_values[col]

    if isinstance(coef, str):
        return results.get(coef, 0)
    else:
        return results

def AUX_select_actors(actor_list, set_series, serie_to_set):
    serie_to_set2 = serie_to_set.copy()
    for key in set_series:
        serie_to_set2[key] = actor_list[key]
    return serie_to_set2

def CN_Effect(actor_list, set_series_directo, set_series_totales,
              variables, set_series_directo_particulares = None,
              sig=False, alpha=0.05):

    """
    Versión de CN_Effect general
    :param actor_list: dict con todos las series que se van a utilizar menos
    la/s serie/s target
    :param set_series_directo: list con los nombres de las series que se deben
     incluir en regre para el efecto directo que se usará si
    set_series_directo_particulares = None.(En muchos casos es igual para todos)
    :param set_series_totales: dict, indicando nombre de la serie y predictandos
    de regre incluyendo la misma serie
    :param set_series_directo_particulares: idem anterior para efectos directos
    que no puedan ser cuantificados con set_series_directo
    :param variables: dict con las series target
    :return: dataframe indicando los efectos totales y directos de cada parent
    hacia las seires target
    """

    for k in variables.keys():
        if k in set_series_directo or k in set_series_totales:
            if set_series_directo_particulares is not None:
                if k in set_series_directo_particulares:
                    print('Error: Variables no puede incluir ser un parent')
                    return
            print('Error: Variables no puede incluir ser un parent')
            return

    result_df = pd.DataFrame(columns=['v_efecto', 'b'])

    for x_name in variables.keys():
        x = variables[x_name]

        try:
            pre_serie = {'c': x['var'].values}
        except:
            pre_serie = {'c': x.values}

        series_directo = AUX_select_actors(actor_list, set_series_directo,
                                           pre_serie)

        series_totales = {}
        actors = []
        for k in set_series_totales.keys():
            actors.append(k)

            series_totales[k] = AUX_select_actors(actor_list,
                                                  set_series_totales[k],
                                                  pre_serie)

            if set_series_directo_particulares is not None:
                series_directas_particulares = {}
                series_directas_particulares[k] = \
                    AUX_select_actors(actor_list,
                                      set_series_directo_particulares[k],
                                      pre_serie)

        if all(actor in series_totales for actor in actors):
            for i in actors:
                # Efecto total i --------------------------------------------- #
                i_total = regre(series_totales[i], True, i, sig, alpha)
                result_df = result_df.append({'v_efecto': f"{i}_TOTAL_{x_name}",
                                              'b': i_total}, ignore_index=True)

                # Efecto directo i ------------------------------------------- #
                try:
                    i_directo = regre(
                        series_directas_particulares[i], True, i, sig, alpha)
                except:
                    i_directo = regre(series_directo, True, i, sig, alpha)

                result_df = result_df.append(
                    {'v_efecto': f"{i}_DIRECTO_{x_name}", 'b': i_directo},
                    ignore_index=True)
        else:
            print('Error: faltan actors en las series')

    return result_df

def set_actor_effect_dict(target, totales, directos,
                          directos_particulares=None):

    # dict ------------------------------------------------------------------- #
    efectos_totales = {}
    efectos_directos = []
    efectos_directos_particulares = None
    effect_dict = {'variable_target': target,
                    'efectos_totales': efectos_totales,
                    'efectos_directos': efectos_directos,
                    'efectos_directos_particulares':
                        efectos_directos_particulares}
    # ------------------------------------------------------------------------ #
    # totales
    for e in totales:
        indice = e.split(':')[0]

        aux = []
        for p in e.split(':')[1].split('+'):
            aux.append(p)

        effect_dict['efectos_totales'][indice] = aux

    # directos:
    effect_dict['efectos_directos'] = directos

    # particulares
    if directos_particulares is not None:
        print('Error: directos_particulares, no seteado')

    return effect_dict

def apply_CEN_effect(actor_list, effects_dict, alpha_sig=[None], sig=True):


    alpha_sig.insert(0, 1)
    alpha_sig = [a for a in alpha_sig if a is not None]

    aux_target = effects_dict['variable_target']
    target = {aux_target : actor_list[aux_target]}

    efectos_directos = effects_dict['efectos_directos']
    efectos_totales = effects_dict['efectos_totales']
    efectos_directos_particulares = effects_dict['efectos_directos_particulares']

    for i in alpha_sig:
        df = CN_Effect(actor_list=actor_list,
                       set_series_directo=efectos_directos,
                       set_series_totales=efectos_totales,
                       variables=target,
                       alpha=i, sig=sig,
                       set_series_directo_particulares=
                       efectos_directos_particulares)


        if i == alpha_sig[0]:
            df['b'] = np.round(df['b'], 3)
            df_final = df
        else:
            df_final[f'alpha_{i}'] = \
                ['*' if i is not None else '' for i in df['b'].values]

    return df_final


# import matplotlib.pyplot as plt
# import cartopy.feature
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import cartopy.crs as ccrs
# import matplotlib
# import matplotlib.ticker as ticker
# def regre_forplot(series, intercept, coef=0, alpha=1):
#     #, filter_significance=True, alpha=1):
#     df = pd.DataFrame(series)
#     if intercept:
#         X = sm.add_constant(df[df.columns[1:]])
#     else:
#         X = df[df.columns[1:]]
#     y = df[df.columns[0]]
#
#     model = sm.OLS(y, X).fit()
#     coefs_results = model.params
#     p_values = model.pvalues
#     # t_values = model.tvalues
#
#     results_sig = {}
#     results_all = {}
#     # p_val = {}
#     # t_val = {}
#     for col in df.columns[1:]:
#         if p_values[col] <= alpha:
#             results_sig[col] = coefs_results[col]
#         else:
#             results_sig[col] = None
#
#         results_all[col] = coefs_results[col]
#
#     if isinstance(coef, str):
#         return results_sig.get(coef, 0), results_all.get(coef, 0)
#     else:
#         return results_sig, results_all
#
# def AUX_select_actors(actor_list, set_series, serie_to_set):
#     serie_to_set2 = serie_to_set.copy()
#     for key in set_series:
#         serie_to_set2[key] = actor_list[key]
#     return serie_to_set2

# def Plot(data, cmap, mapa, save, dpi, titulo, name_fig, out_dir,
#          step=1, data_ctn=None):
#
#     if mapa.lower() == 'sa':
#         fig_size = (5, 6)
#         extent = [270, 330, -60, 20]
#         xticks = np.arange(270, 330, 10)
#         yticks = np.arange(-60, 40, 20)
#         contour = False
#
#     elif mapa.lower() == 'hs':
#         fig_size = (9, 3.5)
#         extent = [0, 359, -80, 20]
#         xticks = np.arange(0, 360, 30)
#         yticks = np.arange(-80, 20, 10)
#         contour = True
#
#     levels = [-1, -.8, -.6, -.4, -.2, -.1, 0, .1, .2, .4, .6, .8, 1]
#
#     fig = plt.figure(figsize=fig_size, dpi=dpi)
#     ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
#     crs_latlon = ccrs.PlateCarree()
#
#     ax.set_extent(extent, crs=crs_latlon)
#
#     if data_ctn is not None:
#         levels_ctn = levels.copy()
#         try:
#             if isinstance(levels_ctn, np.ndarray):
#                 levels_ctn = levels_ctn[levels_ctn != 0]
#             else:
#                 levels_ctn.remove(0)
#         except:
#             pass
#
#
#         ax.contour(data.lon.values[::step], data.lat.values[::step],
#                    data_ctn[::step, ::step], linewidths=0.8,
#                    levels=levels_ctn, transform=crs_latlon,
#                    colors='black')
#
#     im = ax.contourf(data.lon.values[::step], data.lat.values[::step],
#                      data[::step, ::step],
#                      levels=levels,
#                      transform=crs_latlon, cmap=cmap, extend='both')
#
#     cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
#
#     cb.ax.tick_params(labelsize=8)
#
#     ax.add_feature(cartopy.feature.LAND, facecolor='white', linewidth=0.5)
#     # ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.2)
#     ax.coastlines(color='k', linestyle='-', alpha=1, linewidth=0.2,
#                   resolution='110m')
#     gl = ax.gridlines(draw_labels=False, linewidth=0.3, linestyle='-',
#                       zorder=20)
#     gl.ylocator = plt.MultipleLocator(20)
#     lon_formatter = LongitudeFormatter(zero_direction_label=True)
#     lat_formatter = LatitudeFormatter()
#     ax.xaxis.set_major_formatter(lon_formatter)
#     ax.yaxis.set_major_formatter(lat_formatter)
#
#     ax.set_extent(extent, crs=crs_latlon)
#     ax.set_xticks(xticks, crs=crs_latlon)
#     ax.set_yticks(yticks, crs=crs_latlon)
#
#     ax.tick_params(width=0.5, pad=1)
#     ax.tick_params(labelsize=7)
#     plt.title(titulo, fontsize=10)
#     plt.tight_layout()
#     if save:
#         plt.savefig(f"{out_dir}{name_fig}.jpg")
#         plt.close()
#     else:
#         plt.show()
# # ---------------------------------------------------------------------------- #
# import xarray as xr
# #from ENSO_IOD_Funciones import SameDateAs
#
# def OpenObsDataSet(name, sa=True, dir=''):
#     aux = xr.open_dataset(dir + name + '.nc')
#     if sa:
#         aux2 = aux.sel(lon=slice(270, 330), lat=slice(15, -60))
#         if len(aux2.lat) > 0:
#             return aux2
#         else:
#             aux2 = aux.sel(lon=slice(270, 330), lat=slice(-60, 15))
#             return aux2
#     else:
#         return aux
#
#

# def Plot_vsP(data, cmap, save, dpi, titulo, name_fig, out_dir,
#              data_ctn=None):
#
#     fig_size = (5, 5)
#
#     #xticks = np.arange(0, 360, 30)
#     xticks = np.arange(-90, -10, 10)
#     contour = False
#
#     levels = [-1, -.8, -.6, -.4, -.2, -.1, 0, .1, .2, .4, .6, .8, 1]
#
#     fig = plt.figure(figsize=fig_size, dpi=dpi)
#     ax = plt.axes()
#
#     if data_ctn is not None:
#         levels_ctn = levels.copy()
#         try:
#             if isinstance(levels_ctn, np.ndarray):
#                 levels_ctn = levels_ctn[levels_ctn != 0]
#             else:
#                 levels_ctn.remove(0)
#         except:
#             pass
#         ax.contour(data.lat.values, data.pressure_level.values,
#                    data_ctn, linewidths=0.8,
#                    levels=levels_ctn, colors='black')
#
#     im = ax.contourf(data.lat.values, data.pressure_level.values,
#                      data,
#                      levels=levels, cmap=cmap, extend='both')
#
#     cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
#     cb.ax.tick_params(labelsize=8)
#
#     ax.set_xticks(xticks)
#     #ax.set_yticks(yticks, crs=crs_latlon)
#
#     lat_formatter = LatitudeFormatter()
#     ax.xaxis.set_major_formatter(lat_formatter)
#     ax.tick_params(labelsize=7)
#
#     plt.yscale('log')
#     ax.set_ylabel("Pressure [hPa]")
#     ax.set_yscale('log')
#     ax.set_ylim(10.*np.ceil(data.pressure_level.values.max()/10.), 30)
#     subs = [1,2,5]
#     if data.pressure_level.values.max()/100 < 30.:
#         subs = [1,2,3,4,5,6,7,8,9, 10,11,12,13,14,15]
#     y1loc = matplotlib.ticker.LogLocator(base=10., subs=subs)
#     ax.yaxis.set_major_locator(y1loc)
#
#     fmt = matplotlib.ticker.FormatStrFormatter("%g")
#     ax.yaxis.set_major_formatter(fmt)
#     ax.grid()
#     plt.title(titulo, fontsize=10)
#     plt.tight_layout()
#
#     if save:
#         plt.savefig(f"{out_dir}{name_fig}.jpg")
#         plt.close()
#     else:
#         plt.show()