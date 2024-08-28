import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import matplotlib
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


def regre_forplot(series, intercept, coef=0, alpha=1):
    #, filter_significance=True, alpha=1):
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

    results_sig = {}
    results_all = {}
    # p_val = {}
    # t_val = {}
    for col in df.columns[1:]:
        if p_values[col] <= alpha:
            results_sig[col] = coefs_results[col]
        else:
            results_sig[col] = None

        results_all[col] = coefs_results[col]

    if isinstance(coef, str):
        return results_sig.get(coef, 0), results_all.get(coef, 0)
    else:
        return results_sig, results_all

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


def Plot(data, cmap, mapa, save, dpi, titulo, name_fig, out_dir,
         step=1, data_ctn=None):

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

    levels = [-1, -.8, -.6, -.4, -.2, -.1, 0, .1, .2, .4, .6, .8, 1]

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()

    ax.set_extent(extent, crs=crs_latlon)

    if data_ctn is not None:
        levels_ctn = levels.copy()
        try:
            if isinstance(levels_ctn, np.ndarray):
                levels_ctn = levels_ctn[levels_ctn != 0]
            else:
                levels_ctn.remove(0)
        except:
            pass


        ax.contour(data.lon.values[::step], data.lat.values[::step],
                   data_ctn[::step, ::step], linewidths=0.8,
                   levels=levels_ctn, transform=crs_latlon,
                   colors='black')

    im = ax.contourf(data.lon.values[::step], data.lat.values[::step],
                     data[::step, ::step],
                     levels=levels,
                     transform=crs_latlon, cmap=cmap, extend='both')

    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)

    cb.ax.tick_params(labelsize=8)

    ax.add_feature(cartopy.feature.LAND, facecolor='white', linewidth=0.5)
    # ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.2)
    ax.coastlines(color='k', linestyle='-', alpha=1, linewidth=0.2,
                  resolution='110m')
    gl = ax.gridlines(draw_labels=False, linewidth=0.3, linestyle='-',
                      zorder=20)
    gl.ylocator = plt.MultipleLocator(20)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_extent(extent, crs=crs_latlon)
    ax.set_xticks(xticks, crs=crs_latlon)
    ax.set_yticks(yticks, crs=crs_latlon)

    ax.tick_params(width=0.5, pad=1)
    ax.tick_params(labelsize=7)
    plt.title(titulo, fontsize=10)
    plt.tight_layout()
    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()
# ---------------------------------------------------------------------------- #
import xarray as xr
from ENSO_IOD_Funciones import SameDateAs

def OpenObsDataSet(name, sa=True, dir=''):
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

def aux2_Setlag(serie_or, serie_lag, serie_set, years_to_remove):
    if serie_lag is not None:
        serie_f = serie_or.sel(
            time=serie_or.time.dt.month.isin([serie_lag]))
        serie_f = serie_f.sel(
            time=serie_f.time.dt.year.isin(serie_set.time.dt.year))
    else:
        serie_f = SameDateAs(serie_or, serie_set)

    serie_f = serie_f / serie_f.std()

    serie_f = serie_f.sel(time=~serie_f.time.dt.year.isin(years_to_remove))

    return serie_f

def auxSetLags_ActorList(lag_target, lag_dmin34, lag_strato,
                         hgt200_anom_or=None, pp_or=None, dmi_or=None,
                         n34_or=None, asam_or=None, ssam_or=None,
                         sam_or=None, u50_or=None, strato_indice=None,
                         years_to_remove=None, asam_lag=None, ssam_lag=None,
                         sam_lag=None, auxdmi_lag=None, auxn34_lag=None,
                         auxstrato_lag=None, auxsam_lag=None, auxssam_lag=None,
                         auxasam_lag=None, auxhgt_lag=None, auxpp_lag=None):

    # lag_target
    if auxhgt_lag is None:
        hgt200_anom = hgt200_anom_or.sel(
            time=hgt200_anom_or.time.dt.month.isin([lag_target]))
    else:
        hgt200_anom = hgt200_anom_or.sel(
            time=hgt200_anom_or.time.dt.month.isin([auxhgt_lag]))

    if strato_indice is not None:
        hgt200_anom = hgt200_anom.sel(
            time=hgt200_anom.time.dt.year.isin([strato_indice.time]))

    if hgt200_anom is not None:
        hgt200_anom = hgt200_anom / hgt200_anom.std()
        hgt200_anom = hgt200_anom.sel(
            time=~hgt200_anom.time.dt.year.isin(years_to_remove))

        if pp_or is not None:
            if auxpp_lag is None:
                pp = SameDateAs(pp_or, hgt200_anom)
            else:
                pp = pp_or.sel(
                    time=pp_or.time.dt.month.isin([auxpp_lag]))
            pp = pp / pp.std()
            pp = pp.sel(time=~pp.time.dt.year.isin(years_to_remove))
            pp2 = pp.values
        else:
            pp = pp2 = None

        if sam_or is not None:
            sam = aux2_Setlag(sam_or, sam_lag, hgt200_anom, years_to_remove)
            aux_sam = aux2_Setlag(sam_or, auxsam_lag, hgt200_anom,
                                  years_to_remove)
            sam2 = sam.values
            aux_sam2 = aux_sam.values
        else:
            sam = sam2 = None
            aux_sam = aux_sam2 = None

        if asam_or is not None:
            asam = aux2_Setlag(asam_or, asam_lag, hgt200_anom, years_to_remove)
            aux_asam = aux2_Setlag(asam_or, auxasam_lag, hgt200_anom,
                                   years_to_remove)
            asam2 = asam.values
            aux_asam2 = aux_asam.values
        else:
            asam = asam2 = None
            aux_asam = aux_asam2 = None

        if ssam_or is not None:
            ssam = aux2_Setlag(ssam_or, ssam_lag, hgt200_anom, years_to_remove)
            aux_ssam = aux2_Setlag(ssam_or, auxssam_lag, hgt200_anom,
                                   years_to_remove)
            ssam2 = ssam.values
            aux_ssam2 = aux_ssam.values
        else:
            ssam = ssam2 = None
            aux_ssam = aux_ssam2 = None

        if dmi_or is not None:
            dmi = aux2_Setlag(dmi_or, lag_dmin34, hgt200_anom, years_to_remove)
            dmi_aux = aux2_Setlag(dmi_or, auxdmi_lag, hgt200_anom,
                                  years_to_remove)
            dmi2 = dmi.values
            dmi_aux2 = dmi_aux.values
        else:
            dmi = dmi2 = None
            dmi_aux = dmi_aux2 = None

        if n34_or is not None:
            n34 = aux2_Setlag(n34_or, lag_dmin34, hgt200_anom, years_to_remove)
            n34_aux = aux2_Setlag(n34_or, auxn34_lag, hgt200_anom,
                                  years_to_remove)
            n342 = n34.values
            n34_aux2 = n34_aux.values
        else:
            n34 = n342 = None
            n34_aux = n34_aux2 = None

        if u50_or is not None:
            u50 = aux2_Setlag(u50_or, lag_strato, hgt200_anom, years_to_remove)
            u50_aux = aux2_Setlag(u50_or, auxstrato_lag, hgt200_anom,
                                  years_to_remove)
            u502 = u50.values
            u50_aux2 = u50_aux.values
        else:
            u50 = u502 = None
            u50_aux = u50_aux2 = None

        if strato_indice is not None:
            strato_indice = strato_indice.sel(
                time=~strato_indice.time.isin(years_to_remove))
            strato_indice2 = strato_indice['var'].values
        else:
            strato_indice = strato_indice2 = None

        actor_list = {'dmi': dmi2, 'n34': n342, 'ssam': ssam2, 'asam': asam2,
                      'strato': strato_indice2, 'sam': sam2, 'u50': u502,
                      'dmi_aux': dmi_aux2, 'n34_aux': n34_aux2,
                      'u50_aux': u50_aux2,
                      'aux_sam': aux_sam2, 'aux_ssam': aux_ssam2,
                      'aux_asam': aux_asam2}

    else:
        print('Error: hgt200_anom es None')
        hgt200_anom =  pp = asam = ssam = u50 = strato_indice = dmi = n34 = \
        actor_list = dmi_aux = n34_aux = u50_aux = aux_sam = aux_ssam = \
        aux_asam = None
        actor_list = None

    return (hgt200_anom, pp, asam, ssam, u50, strato_indice, dmi, n34,\
           actor_list, dmi_aux, n34_aux, u50_aux, aux_sam, aux_ssam,
            aux_asam)


def aux_alpha_CN_Effect_2(actor_list, set_series_directo, set_series_totales,
                          variables, sig, alpha_sig,
                          set_series_directo_particulares=None):
    for i in alpha_sig:
        linea_sig = pd.DataFrame({'v_efecto': ['alpha'], 'b': [str(i)]})

        df = CN_Effect_2(actor_list, set_series_directo,
                         set_series_totales,
                         variables, alpha=i,
                         sig=sig,
                         set_series_directo_particulares=
                         set_series_directo_particulares)

        if i == alpha_sig[0]:
            df_final = pd.concat([linea_sig, df], ignore_index=True)
        else:
            df_final = pd.concat([df_final, linea_sig, df], ignore_index=True)

    return df_final

def Plot_vsP(data, cmap, save, dpi, titulo, name_fig, out_dir,
         step=1, data_ctn=None):

    fig_size = (5, 5)

    #xticks = np.arange(0, 360, 30)
    xticks = np.arange(-90, -10, 10)
    contour = False

    levels = [-1, -.8, -.6, -.4, -.2, -.1, 0, .1, .2, .4, .6, .8, 1]

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.axes()

    if data_ctn is not None:
        levels_ctn = levels.copy()
        try:
            if isinstance(levels_ctn, np.ndarray):
                levels_ctn = levels_ctn[levels_ctn != 0]
            else:
                levels_ctn.remove(0)
        except:
            pass
        ax.contour(data.lat.values, data.pressure_level.values,
                   data_ctn, linewidths=0.8,
                   levels=levels_ctn, colors='black')

    im = ax.contourf(data.lat.values, data.pressure_level.values,
                     data,
                     levels=levels, cmap=cmap, extend='both')



    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)

    ax.set_xticks(xticks)
    #ax.set_yticks(yticks, crs=crs_latlon)

    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)

    plt.yscale('log')
    ax.set_ylabel("Pressure [hPa]")
    ax.set_yscale('log')
    ax.set_ylim(10.*np.ceil(data.pressure_level.values.max()/10.), 30)
    subs = [1,2,5]
    if data.pressure_level.values.max()/100 < 30.:
        subs = [1,2,3,4,5,6,7,8,9, 10,11,12,13,14,15]
    y1loc = matplotlib.ticker.LogLocator(base=10., subs=subs)
    ax.yaxis.set_major_locator(y1loc)


    fmt = matplotlib.ticker.FormatStrFormatter("%g")
    ax.yaxis.set_major_formatter(fmt)
    ax.grid()
    plt.title(titulo, fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig + '.jpg', dpi = dpi)
        plt.close()
    else:
        plt.show()