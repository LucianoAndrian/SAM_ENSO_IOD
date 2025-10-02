import pandas as pd
import numpy as np
import statsmodels.api as sm
import xarray as xr
from funciones.utils import change_name_dim, change_name_variable

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
        time=~variable_target.time.dt.year.isin(years_to_remove))
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
                          directos_particulares=None,
                          to_parallel_run=False):

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

    for e in totales:
        # totales
        if to_parallel_run:
            indice = e.split(':')[0]
            effect_dict['efectos_totales'][indice] = e.split(':')[1]

        else:
            indice = e.split(':')[0]

            aux = []
            for p in e.split(':')[1].split('+'):
                aux.append(p)

            effect_dict['efectos_totales'][indice] = aux

    # directos:
    if to_parallel_run:
        joined = '+'.join(directos)
        effect_dict['efectos_directos'] = {k: joined for k in directos}
    else:
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

def set_data_to_cen(dir_file, interp_2x2=True,
                    select_lat=None, select_lon=None,
                    rolling=False, rl_win=3,
                    purge_extra_dims=False):
    data = xr.open_dataset(dir_file)
    data = change_name_dim(data, 'latitude', 'lat')
    data = change_name_dim(data, 'longitude', 'lon')
    data = change_name_variable(data, 'var')

    if interp_2x2:
        try:
            data = data.interp(lon=np.arange(data.lon[0], data.lon[-1],2),
                               lat=np.arange(data.lat[0], data.lat[-1],2))
        except:
            data = data.interp(lon=np.arange(data.lon[0], data.lon[-1],2),
                               lat=np.arange(data.lat[0], data.lat[-1],-2))

    if select_lon is not None:
        data = data.sel(lon=slice(select_lon[0], select_lon[-1]))

    if select_lat is not None:
        if len(select_lat)>1:
            data = data.sel(lat=slice(select_lat[0], select_lat[-1]))
        else:
            data = data.sel(lat=select_lat[0])

    data_clim = data.sel(time=slice('1979-01-01', '2000-12-01'))

    data_anom_mon = data.groupby('time.month') - \
                    data_clim.groupby('time.month').mean('time')

    if rolling:
        data_anom_mon = data_anom_mon.rolling(time=rl_win, center=True).mean()

    # pesos de esta forma para normalizar para el analisis
    #weights = np.sqrt(np.abs(np.cos(np.radians(data_anom_mon.lat))))
    # data_anom_mon = data_anom_mon * weights

    data_anom_mon = Weights(data_anom_mon)

    if purge_extra_dims:
        extra_dim = [d for d in data_anom_mon['var'].dims
                     if d not in ['lon', 'lat', 'time']]
        for dim in extra_dim:
            first_val = data_anom_mon[dim].values[0]
            data_anom_mon = data_anom_mon.sel({dim: first_val}).drop(dim)
            print(f'drop_dim: {dim}')

    return data_anom_mon
#
def df_linea_lag(lag_key):
    data = {
        'v_efecto': f'Lag {lag_key}',
        'b': '',
        'alpha_0.15': '',
        'alpha_0.1': '',
        'alpha_0.05': ''
    }
    return pd.DataFrame([data])
#
def concat_df(df1=None, df2=None):
    if df1 is None:
        return df2
    else:
        return pd.concat([df1, df2], ignore_index=True)

def apply_cen_1d(variable_referencia, effects_dict, indices, lags, alpha,
                 years_to_remove, verbose=0):
    df = None
    for l_count, lag_key in enumerate(lags.keys()):
        lag_target, indices_lags = identify_lags(lags[lag_key])

        if verbose > 0: print(f'{lag_key}')
        df_linea = df_linea_lag(lag_key)

        variable_target, actor_list = SetLag_to_ActorList(
            variable_target=variable_referencia,
            month_target=lag_target,
            indices=indices,
            lags=indices_lags,
            years_to_remove=years_to_remove,
            verbose=verbose)

        aux_df = apply_CEN_effect(actor_list, effects_dict,
                                  sig=True, alpha_sig=alpha)

        aux_df = concat_df(df_linea, aux_df)
        df = concat_df(df, aux_df)

    if verbose > 0: print('Done')
    return df

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

def apply_cen_2d(variable_target, effects_dict, indices,
                 lags, alpha, years_to_remove, log_level, verbose):
    from cen.cen import CEN_ufunc

    efectos_totales = {}
    efectos_directos = {}
    for l_count, lag_key in enumerate(lags.keys()):
        lag_target, indices_lags = identify_lags(lags[lag_key])

        variable_target_al, actor_list = SetLag_to_ActorList(
            variable_target=variable_target,
            month_target=lag_target,
            indices=indices,
            lags=indices_lags,
            years_to_remove=years_to_remove,
            verbose=verbose)

        cen = CEN_ufunc(actor_list, log_level=log_level)

        regre_efectos_totales, regre_efectos_directos = \
            cen.run_ufunc_cen(variable_target=variable_target_al,
                              effects=effects_dict, alpha=alpha)

        efectos_totales[lag_key] = regre_efectos_totales
        efectos_directos[lag_key] = regre_efectos_directos

    return efectos_totales, efectos_directos
