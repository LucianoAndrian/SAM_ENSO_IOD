import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs

# def regre(series, intercept, coef=0):
#     df = pd.DataFrame(series)
#     if intercept:
#         X = np.column_stack((np.ones_like(df[df.columns[1]]),
#                              df[df.columns[1:]]))
#     else:
#         X = df[df.columns[1:]].values
#     y = df[df.columns[0]]
#
#     coefs = np.linalg.lstsq(X, y, rcond=None)[0]
#
#     coefs_results = {}
#     for ec, e in enumerate(series.keys()):
#         if intercept and ec == 0:
#             e = 'constant'
#         if e != df.columns[0]:
#             if intercept:
#                 coefs_results[e] = coefs[ec]
#             else:
#                 coefs_results[e] = coefs[ec-1]
#
#     if isinstance(coef, str):
#         return coefs_results[coef]
#     else:
#         return coefs_results

import statsmodels.api as sm
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

#
#
# from scipy.stats import t as t_dist
# def regre(series, intercept, coef=0, filter_significance=True, alpha=0.05):
#     df = pd.DataFrame(series)
#     if intercept:
#         X = sm.add_constant(df[df.columns[1:]])
#     else:
#         X = df[df.columns[1:]]
#     y = df[df.columns[0]]
#
#     model = sm.OLS(y, X).fit()
#     coefs_results = model.params
#     t_values = model.tvalues
#
#     n = len(y)
#     n = len(series[list(series.keys())[0]])
#     p = X.shape[1] - (1 if intercept else 0)  # Número de predictores
#     p = len(series)-1
#     t_critico = t_dist.ppf(1 - alpha/2, df=n - p - 1)
# #    print(t_critico)
#
#     results = {}
#     for col in df.columns[1:]:
#         if filter_significance:
#             if np.abs(t_values[col]) > t_critico:
#                 results[col] = coefs_results[col]
#             else:
#                 results[col] = 0
#         else:
#             results[col] = coefs_results[col]
#
#     if isinstance(coef, str):
#         return results.get(coef, 0)
#     else:
#         return results

def AUX_select_actors(actor_list, set_series, serie_to_set):
    serie_to_set2 = serie_to_set.copy()
    for key in set_series:
        serie_to_set2[key] = actor_list[key]
    return serie_to_set2

def CN_Effect_2(actor_list, set_series_directo, set_series_totales,
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


    print(result_df)
    return result_df



def Plot(data, cmap, mapa, save, dpi, titulo, name_fig, out_dir):

    if mapa.lower() == 'sa':
        fig_size = (5, 6)
        extent = [270, 330, -60, 20]
        xticks = np.arange(270, 330, 10)
        yticks = np.arange(-60, 40, 20)
        contour = False

    elif mapa.lower() == 'hs':
        fig_size = (9, 3.5)
        extent = [0, 359, -80, 20]
        xticks = np.arange(0, 360, 30)
        yticks = np.arange(-80, 20, 10)
        contour = True

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()

    ax.set_extent(extent, crs=crs_latlon)

    im = ax.contourf(data.lon, data.lat, data,
                     levels=[-1, -.8, -.6, -.4, -.2, -.1, 0,
                             .1, .2, .4, .6, .8, 1],
                     transform=crs_latlon, cmap=cmap, extend='both')

    if contour:
        values = ax.contour(data.lon, data.lat, data,
                            levels=[-1, -.8, -.6, -.4, -.2, -.1,
                                    .1, .2, .4, .6, .8, 1],
                            transform=crs_latlon, colors='k', linewidths=1)

    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(xticks, crs=crs_latlon)
    ax.set_yticks(yticks, crs=crs_latlon)
    ax.tick_params(labelsize=7)
    plt.title(titulo, fontsize=10)
    plt.tight_layout()
    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()