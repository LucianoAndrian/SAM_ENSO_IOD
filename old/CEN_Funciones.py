import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import pearsonr
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2
from Scales_Cbars import get_cbars

def regre(series, intercept, coef=0):
    df = pd.DataFrame(series)
    if intercept:
        X = np.column_stack((np.ones_like(df[df.columns[1]]),
                             df[df.columns[1:]]))
    else:
        X = df[df.columns[1:]].values
    y = df[df.columns[0]]

    coefs = np.linalg.lstsq(X, y, rcond=None)[0]

    coefs_results = {}
    for ec, e in enumerate(series.keys()):
        if intercept and ec == 0:
            e = 'constant'
        if e != df.columns[0]:
            if intercept:
                coefs_results[e] = coefs[ec]
            else:
                coefs_results[e] = coefs[ec - 1]

    if isinstance(coef, str):
        return coefs_results[coef]
    else:
        return coefs_results


def AUX_select_actors(actor_list, set_series, serie_to_set):
    serie_to_set2 = serie_to_set.copy()
    for key in set_series:
        serie_to_set2[key] = actor_list[key]
    return serie_to_set2


def CN_Effect(actor_list, set_series_directo, set_series_dmi_total,
              set_series_n34_total, set_series_3index_total,
              set_series_dmi_directo=None,
              set_series_n34_directo=None,
              set_series_3index_directo=None,
              name='cn_result'):
    result_df = pd.DataFrame(columns=['v_efecto', 'b'])

    for x, x_name in zip([pp_caja, amd], ['pp_serie', 'amd']):

        pre_serie = {'c': x['var'].values}
        series_directo = AUX_select_actors(actor_list, set_series_directo,
                                           pre_serie)

        if ('asam' in set_series_directo):
            aux_actors = ['dmi', 'n34', 'asam']
            aux_3index = 'asam'
        elif ('ssam' in set_series_directo):
            aux_actors = ['dmi', 'n34', 'ssam']
            aux_3index = 'ssam'
        elif ('strato' in set_series_directo):
            aux_actors = ['dmi', 'n34', 'strato']
            aux_3index = 'strato'
        elif ('sam' in set_series_directo):
            aux_actors = ['dmi', 'n34', 'sam']
            aux_3index = 'sam'
        else:
            aux_actors = ['dmi', 'n34']
            aux_3index = None

        series = {}
        if set_series_dmi_total is not None:
            series_dmi_total = AUX_select_actors(
                actor_list, set_series_dmi_total, pre_serie)
            series['dmi_total'] = series_dmi_total

        if set_series_n34_total is not None:
            series_n34_total = AUX_select_actors(
                actor_list, set_series_n34_total, pre_serie)
            series['n34_total'] = series_n34_total

        if set_series_3index_total is not None:
            series_3index_total = AUX_select_actors(
                actor_list, set_series_3index_total, pre_serie)
            series[aux_3index + '_total'] = series_3index_total

        series_directo_particular = {}
        if set_series_dmi_directo is not None:
            series_dmi_directo = AUX_select_actors(
                actor_list, set_series_dmi_directo, pre_serie)
            series_directo_particular['dmi_directo'] = series_dmi_directo

        if set_series_n34_directo is not None:
            series_n34_directo = AUX_select_actors(
                actor_list, set_series_n34_directo, pre_serie)
            series_directo_particular['n34_directo'] = series_n34_directo

        if set_series_3index_directo is not None:
            series_3index_directo = AUX_select_actors(
                actor_list, set_series_3index_directo, pre_serie)
            series_directo_particular['3index_directo'] = series_3index_directo

        # series = {'dmi_total': series_dmi_total,
        #           'n34_total': series_n34_total,
        #           aux_3index + '_total': series_3index_total}

        for i in aux_actors:

            # Efecto total i --------------------------------------------------#
            i_total = regre(series[f"{i}_total"], True, i)
            result_df = result_df.append({'v_efecto': f"{i}_TOTAL_{x_name}",
                                          'b': i_total},
                                         ignore_index=True)

            # Efecto directo i ------------------------------------------------#
            try:
                i_directo = regre(
                    series_directo_particular[f"{i}_directo"], True, i)
            except:
                i_directo = regre(series_directo, True, i)

            result_df = result_df.append({'v_efecto': f"{i}_DIRECTO_{x_name}",
                                          'b': i_directo},
                                         ignore_index=True)
        # -------------------------------------------------------------------- #
    result_df.to_csv(f"{out_dir}{name}.txt", sep='\t', index=False)
    print(result_df)


def pre_regre_ufunc(x, modelo, coef, modo):
    pre_serie = {'c': x}

    # modelo A1
    if modelo.upper() == 'A1':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'sam': sam.values}
        set_series_directo = ['dmi', 'n34', 'sam']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_3index_total = ['dmi', 'n34', 'sam']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'sam'

    elif modelo.upper() == 'A1ASAM':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'asam': asam.values}
        set_series_directo = ['dmi', 'n34', 'asam']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_3index_total = ['dmi', 'n34', 'asam']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'asam'

    elif modelo.upper() == 'AMD':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'amd': amd['var'].values}
        set_series_directo = ['dmi', 'n34', 'amd']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_3index_total = ['dmi', 'n34', 'amd']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'amd'

    elif modelo.upper() == 'A1SSAM':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'ssam': asam.values}
        set_series_directo = ['dmi', 'n34', 'ssam']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_3index_total = ['dmi', 'n34', 'ssam']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'ssam'

    elif modelo.upper() == 'A1ASAMWSSAM':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'asam': asam.values, 'ssam': ssam.values}
        set_series_directo = ['dmi', 'n34', 'asam', 'ssam']
        set_series_dmi_total = ['dmi', 'n34', 'ssam']
        set_series_n34_total = ['n34', 'ssam']
        set_series_3index_total = ['dmi', 'n34', 'asam', 'ssam']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'asam'

    elif modelo.upper() == 'A1STRATO':
        actor_list = {'dmi': dmi.values, 'n34': n34.values,
                      'strato': strato_indice['var'].values}
        set_series_directo = ['dmi', 'n34', 'strato']
        set_series_dmi_total = ['dmi', 'n34']
        set_series_n34_total = ['n34']
        set_series_3index_total = ['dmi', 'n34', 'strato']
        set_series_n34_directo = None
        set_series_dmi_directo = None
        set_series_3index_directo = None
        aux_3index = 'strato'
    else:
        print('ningun modelo seleccionado')
        return None

    series_directo = AUX_select_actors(actor_list, set_series_directo,
                                       pre_serie)

    series_dmi_total = AUX_select_actors(
        actor_list, set_series_dmi_total, pre_serie)

    series_n34_total = AUX_select_actors(
        actor_list, set_series_n34_total, pre_serie)

    series_3index_total = AUX_select_actors(
        actor_list, set_series_3index_total, pre_serie)

    series_directo_particular = {}
    if set_series_dmi_directo is not None:
        series_dmi_directo = AUX_select_actors(
            actor_list, set_series_dmi_directo, pre_serie)
        series_directo_particular['dmi_directo'] = series_dmi_directo

    if set_series_n34_directo is not None:
        series_n34_directo = AUX_select_actors(
            actor_list, set_series_n34_directo, pre_serie)
        series_directo_particular['n34_directo'] = series_n34_directo

    if set_series_3index_directo is not None:
        series_3index_directo = AUX_select_actors(
            actor_list, set_series_3index_directo, pre_serie)
        series_directo_particular['3index_directo'] = series_3index_directo

    series = {'dmi_total': series_dmi_total,
              'n34_total': series_n34_total,
              aux_3index + '_total': series_3index_total}

    if modo.lower() == 'total':
        efecto = regre(series[f"{coef}_total"], True, coef)

    elif modo.lower() == 'directo':
        try:
            efecto = regre(
                series_directo_particular[f"{coef}_directo"], True, coef)
        except:
            efecto = regre(series_directo, True, coef)

    return efecto


def compute_regression(variable, modelo, coef, modo):

    coef_dataset = xr.apply_ufunc(
        pre_regre_ufunc,
        variable,  modelo, coef, modo,
        input_core_dims=[['time'], [], [], []],
        vectorize=True)
    return coef_dataset

