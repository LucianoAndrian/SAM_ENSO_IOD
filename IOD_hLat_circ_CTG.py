"""
IOD - SAM/U50 ctg
"""
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- #
save = True
sig_thr = 0.05
hgt_vs_p = False
use_pp = False

out_dir = '/pikachu/datos/luciano.andrian/IOD_vs_hLat/ctg/'
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import xarray as xr
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

from ENSO_IOD_Funciones import (DMI2, DMI2_singlemonth, DMI2_twomonths,
                                ChangeLons, SameDateAs)
from cen_funciones import Detrend, Weights, OpenObsDataSet
import dcor
# ---------------------------------------------------------------------------- #
if save:
    dpi = 200
else:
    dpi = 100

# ---------------------------------------------------------------------------- #
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
hgt_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
dir_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_no_detrend/'
# ---------------------------------------------------------------------------- #
def ddn(serie, serie_ref):
    """
    SameDateAs, Detrned, Norm.

    :param serie: xr.dataset or xr.datarray o lista de ellos
    :param serie_ref: xr.dataset or xr.datarray
    :return: xr.dataarray
    """
    output = None
    if isinstance(serie, list):
        output = []
        for s in serie:
            serie_sd = SameDateAs(s, serie_ref)
            serie_sd_d = Detrend(serie_sd, 'time')
            # Da medio igual la resta de la media xq son todas anomalias
            serie_sd_d_n = ((serie_sd_d - serie_sd_d.mean('time')) /
                            serie_sd_d.std('time'))
            output.append(serie_sd_d_n)
        output = tuple(output)
    else:
        serie = SameDateAs(serie, serie_ref)
        serie = Detrend(serie, 'time')
        # Da medio igual la resta de la media xq son todas anomalias
        serie = (serie - serie.mean('time')) / serie.std('time')
        output = serie

    return output

def SelectDMI_SAM_u50(dmi_df, indice_serie, dmi_serie, mm_dmi, mm_ind,
                      use_dmi_df = False):

    output = None

    if use_dmi_df:
        # DMI pos
        dmi_pos = dmi_df[
            dmi_df['DMI'] > 0]  # acá estan los que cumple DMIIndex()
        # DMI neg
        dmi_neg = dmi_df[dmi_df['DMI'] < 0]
    else:
        # dmi pos
        dmi_pos = dmi_serie.where(dmi_serie > .5, drop=True)
        # dmi neg
        dmi_neg = dmi_serie.where(dmi_serie < -.5, drop=True)

    # indice pos
    ind_pos = indice_serie.where(indice_serie > .5, drop=True)
    # indice neg
    ind_neg = indice_serie.where(indice_serie < -.5, drop=True)

    m_count = 0
    for m_dmi, m_ind in zip(mm_dmi, mm_ind):
        ar_result = np.zeros((13,1))

        dmi_serie_sel = dmi_serie.where(dmi_serie.time.dt.month == m_dmi, drop=True)
        indice_serie_sel = indice_serie.where(indice_serie.time.dt.month == m_ind,
                                              drop=True)
        if use_dmi_df:
            dmi_pos_sel = dmi_pos[dmi_pos['Mes'] == m_dmi]
            dmi_neg_sel = dmi_neg[dmi_neg['Mes'] == m_dmi]
        else:

            dmi_pos_sel = dmi_pos.where(dmi_pos.time.dt.month == m_dmi,
                                        drop=True)
            dmi_neg_sel = dmi_neg.where(dmi_neg.time.dt.month == m_dmi,
                                        drop=True)


        ind_pos_sel = ind_pos.where(ind_pos.time.dt.month == m_ind, drop=True)
        ind_neg_sel = ind_neg.where(ind_neg.time.dt.month == m_ind, drop=True)

        if use_dmi_df:
            ind_dmi_pos = ind_pos_sel.where(
                ind_pos_sel.time.dt.year.isin(dmi_pos_sel['Años'].values),
                drop=True)

            ind_dmi_neg = ind_neg_sel.where(
                ind_neg_sel.time.dt.year.isin(dmi_neg_sel['Años'].values),
                drop=True)

            ind_pos_dmi_neg = ind_pos_sel.where(
                ind_pos_sel.time.dt.year.isin(dmi_neg_sel['Años'].values),
                drop=True)

            ind_neg_dmi_pos = ind_neg_sel.where(
                ind_neg_sel.time.dt.year.isin(dmi_pos_sel['Años'].values),
                drop=True)

            dmi_pos_sel_puros = dmi_pos_sel[~dmi_pos_sel['Años'].isin(
                ind_dmi_pos.time.dt.year.values)]
            dmi_pos_sel_puros = dmi_pos_sel_puros[
                ~dmi_pos_sel_puros['Años'].isin(
                    ind_neg_dmi_pos.time.dt.year.values)]

            dmi_neg_sel_puros = dmi_neg_sel[~dmi_neg_sel['Años'].isin(
                ind_dmi_neg.time.dt.year.values)]
            dmi_neg_sel_puros = dmi_neg_sel_puros[
                ~dmi_neg_sel_puros['Años'].isin(
                    ind_pos_dmi_neg.time.dt.year.values)]

        else:

            ind_dmi_pos = ind_pos_sel.where(
                ind_pos_sel.time.dt.year.isin(dmi_pos_sel.time.dt.year.values),
                drop=True)

            ind_dmi_neg = ind_neg_sel.where(
                ind_neg_sel.time.dt.year.isin(dmi_neg_sel.time.dt.year.values),
                drop=True)

            ind_pos_dmi_neg = ind_pos_sel.where(
                ind_pos_sel.time.dt.year.isin(dmi_neg_sel.time.dt.year.values),
                drop=True)

            ind_neg_dmi_pos = ind_neg_sel.where(
                ind_neg_sel.time.dt.year.isin(dmi_pos_sel.time.dt.year.values),
                drop=True)

            dmi_pos_sel_puros = dmi_pos_sel.sel(
                time=~dmi_pos_sel.time.dt.year.isin(
                    ind_dmi_pos.time.dt.year.values))
            dmi_pos_sel_puros = dmi_pos_sel_puros.sel(
                time=~dmi_pos_sel_puros.time.dt.year.isin(
                    ind_neg_dmi_pos.time.dt.year.values))

            dmi_neg_sel_puros = dmi_neg_sel.sel(
                time=~dmi_neg_sel.time.dt.year.isin(
                    ind_dmi_neg.time.dt.year.values))
            dmi_neg_sel_puros = dmi_neg_sel_puros.sel(
                time=~dmi_neg_sel_puros.time.dt.year.isin(
                    ind_pos_dmi_neg.time.dt.year.values))

        ind_pos_sel_puros = ind_pos_sel.sel(
            time=~ind_pos_sel.time.dt.year.isin(
                ind_dmi_pos.time.dt.year.values))
        ind_pos_sel_puros = ind_pos_sel_puros.sel(
            time=~ind_pos_sel_puros.time.dt.year.isin(
                ind_pos_dmi_neg.time.dt.year.values))

        ind_neg_sel_puros = ind_neg_sel.sel(
            time=~ind_neg_sel.time.dt.year.isin(
                ind_dmi_neg.time.dt.year.values))
        ind_neg_sel_puros = ind_neg_sel_puros.sel(
            time=~ind_neg_sel_puros.time.dt.year.isin(
                ind_neg_dmi_pos.time.dt.year.values))

        ind_events = (len(ind_pos_sel_puros.time.values) +
                      len(ind_neg_sel_puros.time.values))

        dmi_events = (len(dmi_pos_sel_puros) +
                      len(dmi_neg_sel_puros))

        in_phase = len(ind_dmi_pos.time.values) + len(ind_dmi_neg.time.values)

        out_phase = (len(ind_pos_dmi_neg.time.values) +
                     len(ind_neg_dmi_pos.time.values))

        ind_neutros = indice_serie_sel.where(abs(indice_serie_sel) < .5,
                                             drop=True)
        if use_dmi_df:
            dmi_neutros = dmi_serie_sel.sel(
                time=~dmi_serie_sel.time.dt.year.isin(
                    dmi_df['Años'].values))
        else:
            dmi_neutros = dmi_serie_sel.sel(
                time=~dmi_serie_sel.time.dt.year.isin(
                    dmi_pos_sel.time.dt.year.values))
            dmi_neutros = dmi_neutros.sel(
                time=~dmi_neutros.time.dt.year.isin(
                    dmi_neg_sel.time.dt.year.values))

        ar_result[0, 0] = len(dmi_pos_sel_puros)
        ar_result[1, 0] = len(dmi_neg_sel_puros)

        ar_result[2, 0] = len(ind_pos_sel_puros.time.values)
        ar_result[3, 0] = len(ind_neg_sel_puros.time.values)

        ar_result[4, 0] = len(ind_dmi_pos.time.values)
        ar_result[5, 0] = len(ind_dmi_neg.time.values)

        ar_result[6, 0] = len(ind_pos_dmi_neg.time.values)
        ar_result[7, 0] = len(ind_neg_dmi_pos.time.values)

        ar_result[8, 0] = ind_events
        ar_result[9, 0] = dmi_events

        ar_result[10, 0] = in_phase
        ar_result[11, 0] = out_phase

        ar_result[12, 0] = (len(ind_neutros.time.values) +
                                  len(dmi_neutros.time.values))
        if m_count == 0:
            m_count = 1
            df = pd.DataFrame(ar_result, columns=[str(m_dmi)])
        else:
            df[str(m_dmi)] = ar_result

    return df

# Las proximas funciones las hice cuando recien empezaba a usar python,
# no me juzguen.
# Tienen actualizaciones...
def ReorderTable(data):

    output = []

    for c in data.columns:
        data_c = data[c]

        result = pd.DataFrame(columns=['0', '1', '2'], dtype=float)

        row1 = [data_c[4], data_c[6], data_c[2]]
        row2 = [data_c[7], data_c[5], data_c[3]]
        row3 = [data_c[0], data_c[1], data_c[12]]

        result = result.append(pd.DataFrame([row1], columns=result.columns))
        result = result.append(pd.DataFrame([row2], columns=result.columns))
        result = result.append(pd.DataFrame([row3], columns=result.columns))

        output.append(result)
    return output

def CtgTest(tabla, alpha=0.05):

    tabla_rnd_output = []
    chi_sq = []
    chi_sq_teo = []

    try:
        tabla.shape
        tabla = [tabla]
    except:
        pass

    for t in tabla:
        tot_row = t.apply(np.sum, axis=1)
        tot_col = t.apply(np.sum, axis=0)
        total = sum(tot_col)

        tabla_rand = pd.DataFrame(columns=t.columns, dtype=float)

        for r in range(0, len(tot_row)):
            aux = tot_row.values[r] * tot_col.values / total

            tabla_rand = tabla_rand.append(
                pd.DataFrame([aux], columns=t.columns))

        # esto es tabla y luego suma para obtener el chi-sq
        est = sum((((t - tabla_rand) ** 2) / tabla_rand).sum())
        import scipy.stats as stats
        est_teo = stats.chi2.ppf(1 - alpha,
                                 (len(tot_row) - 1) * (len(tot_col) - 1))

        if abs(est) > abs(est_teo):
            print(f'Rechazo H0 con un {(1 - alpha) * 100}% de confianza')
        else:
            print(f'No rechazo H0 con un {(1 - alpha) * 100}% de confianza')

        tabla_rnd_output.append(tabla_rand)
        chi_sq.append(est)
        chi_sq_teo.append(est_teo)

    return tabla_rnd_output, chi_sq, chi_sq_teo

def Re_ReorderTable(tabla_rnd, name):

    try:
        tabla_rnd.shape
        tabla_rnd = [tabla_rnd]
    except:
        pass
    retult_output = []
    for t in tabla_rnd:
        t = np.round( t, 2)
        aux = [ t.iloc[2][0],  t.iloc[2][1],  t.iloc[0][2],
                t.iloc[1][2],  t.iloc[0][0],  t.iloc[1][1],
                t.iloc[0][1],  t.iloc[1][0],  t.iloc[2][2]]

        result = pd.DataFrame(columns=[name], dtype=float)

        result = result.append(pd.DataFrame(aux, columns=result.columns))

        retult_output.append(result)

    return retult_output

def combinar_dataframes(lista1, lista2, col_names):
    """
    """
    if len(lista1) != len(lista2):
        raise ValueError("Las dos listas deben tener la misma longitud.")

    data = {}

    for i in range(len(lista1)):
        columna1 = lista1[i].iloc[:,0]
        columna2 = lista2[i].iloc[:,0]

        data[f'{col_names[i]}_obs'] = columna1.values
        data[f'{col_names[i]}_esp'] = columna2.values

    df_final = pd.DataFrame(data)

    return df_final

def TablaFogt(cases, alpha, second_index):
    tabla = ReorderTable(cases)
    tabla_rnd, chi_sq, chi_sq_teo = CtgTest(tabla, alpha)
    tabla_esperado = Re_ReorderTable(tabla_rnd, 'esperado')
    tabla = Re_ReorderTable(tabla, 'observado')
    result = combinar_dataframes(tabla, tabla_esperado, list(cases.columns))
    result.index = ['DMI+', 'DMI-', f'{second_index}+', f'{second_index}-',
                    f'DMI+{second_index}+', f'SIM-{second_index}-',
                    f'DMI-{second_index}+', f'DMI+{second_index}-',
                    'neutro']

    return result, tabla, tabla_esperado, chi_sq, chi_sq_teo

def plot_df(df, title='', name_fig='', save=False, out_dir=out_dir):
    fig, ax = plt.subplots(figsize=(len(df),len(df)/3.33), dpi=200)
    ax.axis('off')  # Ocultar los ejes

    # Crear la tabla
    tabla = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=result.index,
        cellLoc='center',
        loc='center'
    )

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    ax.set_title(title, fontsize=12)

    if save:
        plt.savefig(f'{out_dir}{name_fig}.png', bbox_inches='tight',
                    pad_inches=0.1, dpi=200)
        plt.close('all')
    else:
        plt.tight_layout()
        plt.show()

# ---------------------------------------------------------------------------- #
# HGT ------------------------------------------------------------------------ #
hgt = xr.open_dataset(hgt_dir + 'ERA5_HGT200_40-20.nc')
hgt = hgt.rename({'longitude': 'lon', 'latitude': 'lat', 'z': 'var'})
hgt = hgt.interp(lon=np.arange(0, 360, 2), lat=np.arange(-90, 90, 2))
hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
hgt200_anom_or = (hgt.groupby('time.month') -
                  hgt_clim.groupby('time.month').mean('time'))
weights = np.sqrt(np.abs(np.cos(np.radians(hgt200_anom_or.lat))))
hgt200_anom_or_1rm = hgt200_anom_or * weights

hgt200_anom_or_2rm = hgt200_anom_or_1rm.rolling(time=2, center=True).mean()
hgt200_anom_or_3rm = hgt200_anom_or_1rm.rolling(time=3, center=True).mean()

#
# if hgt_vs_p:
#     hgt_lvls = xr.open_dataset('/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/'
#                                'ERA5_HGT500-10_79-20.mon.nc')
#     hgt_lvls = convertdates(hgt_lvls, 'date', 'time')
#     hgt_lvls = ChangeLons(hgt_lvls, 'longitude')
#     hgt_lvls = hgt_lvls.rename({'latitude': 'lat', 'z': 'var'})
#     hgt_lvls = hgt_lvls.drop('expver')
#     hgt_lvls = hgt_lvls.drop('number')
#
#     # Esto tarda mucho y es mucho peor cuando se selecciona antes una region
#     # mas chica de longitud.
#     # Va por niveles xq ocupa menos ram
#     print('interp...')
#     first = True
#     for l in hgt_lvls.pressure_level.values:
#         print(l)
#         aux = hgt_lvls.sel(pressure_level=l)
#
#         aux = aux.interp(lon=np.arange(hgt_lvls.lon.values[0],
#                                        hgt_lvls.lon.values[-1] + 1, 1),
#                          lat=np.arange(hgt_lvls.lat.values[-1],
#                                        hgt_lvls.lat.values[0] + 1, 1)[::-1])
#         if first:
#             first = False
#             hgt_lvls_interp = aux
#         else:
#             hgt_lvls_interp = xr.concat([hgt_lvls_interp, aux],
#                                         dim='pressure_level')
#
# # PP ------------------------------------------------------------------------- #
# if use_pp:
#     pp_or = OpenObsDataSet(name='pp_pgcc_v2020_1891-2023_1', sa=True,
#                            dir=dir_pp)
#     pp_or = pp_or.rename({'precip': 'var'})
#     pp_or = pp_or.sel(time=slice('1959-01-16', '2020-12-16'))
#
#     pp_or = Weights(pp_or)
#     pp_or = pp_or.sel(lat=slice(20, -60), lon=slice(270, 330))  # SA

# ---------------------------------------------------------------------------- #
# Indices
# ---------------------------------------------------------------------------- #
dmi_or_1rm = DMI2_singlemonth(filter_bwa=False, start_per='1959', end_per='2020',
                  sst_anom_sd=False, opposite_signs_criteria=False)[2]
dmi_or_2rm = DMI2_twomonths(filter_bwa=False, start_per='1959', end_per='2020',
                  sst_anom_sd=False, opposite_signs_criteria=False)[2]
dmi_or_3rm = DMI2(filter_bwa=False, start_per='1959', end_per='2020',
                  sst_anom_sd=False, opposite_signs_criteria=False)[2]

sam_or_1rm = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam_or_2rm = sam_or_1rm.rolling(time=2, center=True).mean()
sam_or_3rm = sam_or_1rm.rolling(time=3, center=True).mean()
sam_or_3rm[-1] = 0

u50_or = xr.open_dataset('/pikachu/datos/luciano.andrian/observado/'
                         'ncfiles/ERA5/downloaded/ERA5_U50hpa_40-20.mon.nc')
u50_or = u50_or.rename({'u': 'var'})
u50_or = u50_or.rename({'longitude': 'lon'})
u50_or = u50_or.rename({'latitude': 'lat'})
u50_or = Weights(u50_or)
u50_or = u50_or.sel(lat=-60)
u50_or = u50_or.sel(expver=1).drop('expver')

u50_or = (u50_or.groupby('time.month') -
              u50_or.groupby('time.month').mean('time'))
u50_or_1rm = u50_or.mean('lon')

u50_or_2rm = u50_or.rolling(time=2, center=True).mean()
u50_or_2rm = u50_or_2rm.mean('lon')
u50_or_3rm = u50_or.rolling(time=3, center=True).mean()
u50_or_3rm = u50_or_3rm.mean('lon')

# ---------------------------------------------------------------------------- #

(dmi_or_1rm, dmi_or_2rm, dmi_or_3rm, sam_or_1rm, sam_or_2rm, sam_or_3rm,
 u50_or_1rm, u50_or_2rm, u50_or_3rm, hgt200_anom_or_1rm, hgt200_anom_or_2rm,
 hgt200_anom_or_3rm) = ddn([dmi_or_1rm, dmi_or_2rm, dmi_or_3rm, sam_or_1rm,
                            sam_or_2rm, sam_or_3rm, u50_or_1rm, u50_or_2rm,
                            u50_or_3rm, hgt200_anom_or_1rm, hgt200_anom_or_2rm,
                            hgt200_anom_or_3rm], dmi_or_1rm)

# ---------------------------------------------------------------------------- #
meses_dmi = [7, 8, 9, 10]

dmis = [dmi_or_1rm, dmi_or_2rm, dmi_or_3rm]
sams = [sam_or_1rm, sam_or_2rm, sam_or_3rm]
u50s = [u50_or_1rm['var'], u50_or_2rm['var'], u50_or_3rm['var']]

for rl in [0,1,2]:
    aux_indices = [sams[rl], u50s[rl]]
    aux_dmi = dmis[rl]

    for index_name, index in zip(['sam', 'u50'], aux_indices):
        for lag in [0, 1, 2, 3, 4]:
            meses_ind = [x - lag for x in meses_dmi]

            cases = SelectDMI_SAM_u50(None, index, aux_dmi, meses_dmi,
                                      meses_ind, use_dmi_df=False)

            result, tabla, tabla_esperado, chi_sq, chi_sq_teo = (
                TablaFogt(cases, 0.1, index_name.upper()))

            aux_fila = []
            for chi, teo in zip(chi_sq, chi_sq_teo):
                aux_fila.extend([np.round(chi, 1), np.round(teo, 1)])

            result.loc["chi_sq_test"] = aux_fila

            title = f'DMI vs {index_name.upper()} lag. {lag} - rm-{rl+1}'
            name_fig = f'ctg_dmi_vs_{index_name}_lag-{lag}_rl-{rl+1}'
            plot_df(result, title=title, name_fig=name_fig, save=save)
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #