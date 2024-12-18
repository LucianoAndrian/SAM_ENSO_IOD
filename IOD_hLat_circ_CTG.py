"""
IOD - SAM/U50 ctg
"""
# ---------------------------------------------------------------------------- #
save = True
sig_thr = 0.05

out_dir = '/pikachu/datos/luciano.andrian/IOD_vs_hLat/ctg/'
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import IOD_hLat_cic_SET_VAR
# ---------------------------------------------------------------------------- #
if save:
    dpi = 200
else:
    dpi = 100
# ---------------------------------------------------------------------------- #
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

        if use_dmi_df:
            ar_result[0, 0] = len(dmi_pos_sel_puros)
            ar_result[1, 0] = len(dmi_neg_sel_puros)
        else:
            ar_result[0, 0] = len(dmi_pos_sel_puros.time.values)
            ar_result[1, 0] = len(dmi_neg_sel_puros.time.values)

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
        data[f'{col_names[i]}_dif_esp'] = columna2.values

    df_final = pd.DataFrame(data)

    return df_final

def TablaFogt(cases, alpha, second_index):
    tabla_or = ReorderTable(cases)
    tabla_rnd, chi_sq, chi_sq_teo = CtgTest(tabla_or, alpha)
    tabla_esperado = Re_ReorderTable(tabla_rnd, 'esperado')
    tabla = Re_ReorderTable(tabla_or, 'observado')

    dif = []
    for t, e in zip(tabla, tabla_esperado):
        df = np.round(pd.DataFrame(t.values - e.values),1)
        dif.append(df.rename(columns={df.columns[0]: 'dif'}))

    result = combinar_dataframes(tabla, dif, list(cases.columns))
    result.index = ['DMI+', 'DMI-', f'{second_index}+', f'{second_index}-',
                    f'DMI+{second_index}+', f'DMI-{second_index}-',
                    f'DMI-{second_index}+', f'DMI+{second_index}-',
                    'neutro']

    return result, tabla, tabla_esperado, chi_sq, chi_sq_teo, tabla_or

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

def plot_contingency_matrices(result, first_index='DMI', second_index='U50',
                              cmap='Blues', cmap_esp='Reds', save=False,
                              out_dir=None, name_fig='fig', dpi=100):
    """
    Genera un gráfico que contiene matrices de contingencia para las columnas
    "obs" y "esp".

    Parameters:
        result (DataFrame): DataFrame con columnas que contienen "obs" y "esp".
        cmap (str): Paleta de colores para el gráfico (por defecto 'Blues').
    """

    # columnas "obs" y "esp"
    obs_cols = [col for col in result.columns if "obs" in col]
    esp_cols = [col for col in result.columns if "esp" in col]

    # aux
    aux = result.loc[result.index != "neutro", obs_cols]
    max_obs = aux.abs().values.max()
    aux = result.loc[result.index != "neutro", esp_cols]
    max_esp = aux.abs().values.max()

    row_col_order = [f"{first_index}+{second_index}-",
                     f"{first_index}+", f"{first_index}+{second_index}+",
                     f"{second_index}-",
                     "...",
                     f"{second_index}+",
                     f"{first_index}-{second_index}-",
                     f"{first_index}-",
                     f"{first_index}-{second_index}+"]

    if len(obs_cols) == 1:
        fig, axes = plt.subplots(1, len(obs_cols)+1,
                                 figsize=(8 * len(obs_cols), 4),
                                 squeeze=False, dpi=dpi)
    else:
        fig, axes = plt.subplots(2, len(obs_cols),
                                 figsize=(4 * len(obs_cols), 8),
                                 squeeze=False, dpi=dpi)

    ims = []
    ims_esp = []
    for i, (obs_col, esp_col) in enumerate(zip(obs_cols, esp_cols)):
        obs_matrix = np.zeros((3, 3))
        esp_matrix = np.zeros((3, 3))

        for r in range(3):
            for c in range(3):
                try:
                    obs_matrix[r, c] = result.loc[
                        row_col_order[r * 3 + c], obs_col]
                    esp_matrix[r, c] = result.loc[
                        row_col_order[r * 3 + c], esp_col]
                except:
                    obs_matrix[r, c] = None
                    esp_matrix[r, c] = None

        ax_obs = axes[0, i]
        ax_obs2 = ax_obs.twinx()
        ax_obs3 = ax_obs2.twiny()

        if len(obs_cols) == 1:
            ax_esp = axes[0, i+1]
        else:
            ax_esp = axes[1, i]

        ax_esp2 = ax_esp.twinx()
        ax_esp3 = ax_esp2.twiny()

        im_obs = ax_obs.imshow(obs_matrix, cmap=cmap, aspect='auto',
                               vmin=0, vmax=max_obs)
        ims.append(im_obs) # Guardar la imagen para la barra de color comun
        for (row, col), value in np.ndenumerate(obs_matrix):
            if ~np.isnan(value):
                if value > max_obs / 2:
                    color = 'white'
                else:
                    color = 'black'
                ax_obs.text(col, row, value, ha='center', va='center',
                            color=color)
            else:
                ax_obs.text(col, row, 'neutro', ha='center', va='center',
                            color='black')

        ax_obs.set_title(f"{obs_col}")
        ax_obs.set_xlabel(f"{first_index}-")
        ax_obs.set_ylabel(f"{second_index}-")
        ax_obs2.set_ylabel(f"{second_index}+")
        ax_obs3.set_xlabel(f"{first_index}+")

        ax_obs.set_xticks([])
        ax_obs.set_yticks([])
        ax_obs.xaxis.set_ticks_position('none')

        ax_obs2.set_xticks([])
        ax_obs2.set_yticks([])
        ax_obs2.xaxis.set_ticks_position('none')

        ax_obs3.set_xticks([])
        ax_obs3.set_yticks([])
        ax_obs3.xaxis.set_ticks_position('none')

        im_esp = ax_esp.imshow(np.abs(esp_matrix), cmap=cmap_esp, aspect='auto',
                               vmin=0, vmax=max_esp)
        ims_esp.append(im_esp) # Guardar la imagen para la barra de color comun
        for (row, col), value in np.ndenumerate(np.abs(esp_matrix)):
            if ~np.isnan(value):
                if value > max_esp / 2:
                    color = 'white'
                else:
                    color = 'black'
                ax_esp.text(col, row, value, ha='center', va='center',
                            color=color)
            else:
                ax_esp.text(col, row, 'neutro', ha='center', va='center',
                            color='black')

        ax_esp.set_title(f"{esp_col}")
        ax_esp.set_xlabel(f"{first_index}-")
        ax_esp.set_ylabel(f"{second_index}-")
        ax_esp2.set_ylabel(f"{second_index}+")
        ax_esp3.set_xlabel(f"{first_index}+")

        ax_esp.set_xticks([])
        ax_esp.set_yticks([])
        ax_esp.xaxis.set_ticks_position('none')

        ax_esp2.set_xticks([])
        ax_esp2.set_yticks([])
        ax_esp2.xaxis.set_ticks_position('none')

        ax_esp3.set_xticks([])
        ax_esp3.set_yticks([])
        ax_esp3.xaxis.set_ticks_position('none')

    plt.subplots_adjust(hspace=0.3)
    if save:
        plt.savefig(f'{out_dir}{name_fig}.png', bbox_inches='tight',
                    pad_inches=0.1, dpi=200)
        plt.close('all')
    plt.show()
# ---------------------------------------------------------------------------- #
(dmi_or_1rm, dmi_or_2rm, dmi_or_3rm, sam_or_1rm, sam_or_2rm, sam_or_3rm,
 u50_or_1rm, u50_or_2rm, u50_or_3rm, hgt200_anom_or_1rm, hgt200_anom_or_2rm,
 hgt200_anom_or_3rm, hgt_1rm, hgt_2rm, hgt_3rm) = IOD_hLat_cic_SET_VAR.compute()
# ---------------------------------------------------------------------------- #
# meses_dmi = [9, 10]
#
# dmis = [dmi_or_1rm, dmi_or_2rm, dmi_or_3rm]
# sams = [sam_or_1rm, sam_or_2rm, sam_or_3rm]
# u50s = [u50_or_1rm['var'], u50_or_2rm['var'], u50_or_3rm['var']]
#
# for rl in [0,2]:
#     aux_indices = [sams[rl], u50s[rl]]
#     aux_dmi = dmis[rl]
#
#     for index_name, index in zip(['sam', 'u50'], aux_indices):
#         for lag in [0, 1, 2, 3, 4]:
#             meses_ind = [x - lag for x in meses_dmi]
#
#             cases = SelectDMI_SAM_u50(None, index, aux_dmi, meses_dmi,
#                                       meses_ind, use_dmi_df=False)
#
#             result, tabla, tabla_esperado, chi_sq, chi_sq_teo, tabla_or = (
#                 TablaFogt(cases, 0.1, index_name.upper()))
#
#
#
#             aux_fila = []
#             for chi, teo in zip(chi_sq, chi_sq_teo):
#                 aux_fila.extend([np.round(chi, 1), np.round(teo, 1)])
#
#             result.loc["chi_sq_test"] = aux_fila
#
#             title = f'DMI vs {index_name.upper()} lag. {lag} - rm-{rl+1}'
#             name_fig = f'ctg_dmi_vs_{index_name}_lag-{lag}_rl-{rl+1}'
#             plot_df(result, title=title, name_fig=name_fig, save=save)


# U50 Ago --> DMI SON
meses_ind = [8]
meses_dmi = [10]
index = u50_or_1rm['var']
index_name = 'U50'
aux_dmi = dmi_or_3rm

cases = SelectDMI_SAM_u50(None, index, aux_dmi, meses_dmi,
                          meses_ind, use_dmi_df=False)

result, tabla, tabla_esperado, chi_sq, chi_sq_teo, tabla_or = (
                TablaFogt(cases, 0.1, index_name.upper()))

aux_fila = []
for chi, teo in zip(chi_sq, chi_sq_teo):
    aux_fila.extend([np.round(chi, 1), np.round(teo, 1)])
    result.loc["chi_sq_test"] = aux_fila

title = f'DMI - SON vs {index_name.upper()} - Ago.'
name_fig = f'CTG_dmi-son_vs_{index_name}_ago'
plot_df(result, title=title, name_fig=name_fig, save=save)
plot_contingency_matrices(result, save=save, out_dir=out_dir,
                          name_fig=f'PLOT_{name_fig}', dpi=dpi)
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #