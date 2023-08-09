"""
Scatter IOD-SAM identificando los IOD según su ocurrencia con el ENSO
"""
################################################################################
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/scatter/'
out_dir_dataframe = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'

################################################################################
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from ENSO_IOD_Funciones import DMI, Nino34CPC, SameDateAs, DMI2
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
################################################################################
save = True
seasons = ['JJA', 'SON']
mmonth = [7, 10]

if save:
    dpi = 300
else:
    dpi = 100
################################################################################
def NormSD(serie):
    return serie / serie.std('time')

def auxScatter(n34, n34_3, dmi, dmi_3, sam, s):
    """
    Hace lo mismo que en ENSO_IOD_Satter_OBS.py pero ahora devuelve los valores
    de SAM para cada case de IOD. (en lugar de los valores de N34)

    :param n34:
    :param n34_3:
    :param dmi:
    :param dmi_3:
    :param sam:
    :param s:
    :return:
    """
    dmi_todos = dmi_3.sel(time=dmi_3.time.dt.month.isin([s]))
    dmi_criteria_y = dmi.where((dmi.Mes == s)).Años.dropna().values

    n34_todos = n34.sel(time=n34.time.dt.month.isin([s]))
    n34_criteria_y = n34_3.where((n34_3.Mes == s)).Años.dropna().values

    sam_todos = sam.sel(time=sam.time.dt.month.isin([s]))
    # -------------------------------------------------------------------------#
    sim_y = np.intersect1d(n34_criteria_y, dmi_criteria_y)

    dmi_sim = dmi_todos.sel(time=dmi_todos.time.dt.year.isin(sim_y))
    n34_sim = n34_todos.sel(time=n34_todos.time.dt.year.isin(sim_y))

    dmi_sim_pos = dmi_sim.where(dmi_sim > 0)
    n34_sim_pos = n34_sim.where(n34_sim > 0)
    dmi_sim_pos_sam_values = sam_todos.sel(
        time=sam_todos.time.isin(dmi_sim_pos.time))
    # Esto solo para los casos de fases opuestas entre ENSO-IOD
    n34_sim_pos_sam_values = sam_todos.sel(
        time=sam_todos.time.isin(n34_sim_pos.time))


    dmi_sim_neg = dmi_sim.where(dmi_sim < 0)
    n34_sim_neg = n34_sim.where(n34_sim < 0)
    dmi_sim_neg_sam_values = sam_todos.sel(
        time=sam_todos.time.isin(dmi_sim_neg.time))
    n34_sim_neg_sam_values = sam_todos.sel(
        time=sam_todos.time.isin(n34_sim_neg.time))

    dmi_pos_n34_neg = dmi_sim_pos.where(~np.isnan(n34_sim_neg.values))
    dmi_neg_n34_pos = dmi_sim_neg.where(~np.isnan(n34_sim_pos.values))

    dmi_dates_ref = dmi_todos.time.dt.year
    mask = np.in1d(dmi_dates_ref, dmi_criteria_y)
    aux_dmi = dmi_todos.sel(
        time=dmi_todos.time.dt.year.isin(dmi_dates_ref[mask]))

    n34_dates_ref = n34_todos.time.dt.year
    mask = np.in1d(n34_dates_ref, n34_criteria_y)
    aux_n34 = n34_todos.sel(
        time=n34_todos.time.dt.year.isin(n34_dates_ref[mask]))

    aux_dates_ref = aux_dmi.time.dt.year
    mask = np.in1d(aux_dates_ref, sim_y, invert=True)
    dmi_un = aux_dmi.sel(time=aux_dmi.time.dt.year.isin(aux_dates_ref[mask]))

    dmi_un_pos = dmi_un.where(dmi_un > 0)
    dmi_un_pos_sam_values = sam_todos.sel(
        time=sam_todos.time.isin(dmi_un_pos.time))

    dmi_un_neg = dmi_un.where(dmi_un < 0)
    dmi_un_neg_sam_values = sam_todos.sel(
        time=sam_todos.time.isin(dmi_un_neg.time))

    return dmi_un_pos, dmi_un_pos_sam_values, \
           dmi_un_neg, dmi_un_neg_sam_values, \
           dmi_sim_pos, dmi_sim_pos_sam_values, n34_sim_pos_sam_values,\
           dmi_sim_neg, dmi_sim_neg_sam_values, n34_sim_neg_sam_values,\
           dmi_todos, sam_todos, \
           dmi_pos_n34_neg, dmi_neg_n34_pos

def ToCombinedDataframe(dmi, sam, out_dir, name):
    dmi_df = dmi.to_dataframe(name='dmi')
    sam_df = sam.to_dataframe(name='sam')

    combined_df = pd.concat([dmi_df, sam_df], axis=1)
    # Hay valores con NaN x como se calculan en auxScatter()
    combined_df = combined_df.dropna(subset=['dmi'])

    output_file = f"{out_dir}/{name}.txt"
    combined_df.to_csv(output_file, sep='\t')

################################################################################
# indices
sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
#
# dmi = DMI(filter_bwa=False, start_per='1920', end_per='2020')[2]
# n34 = Nino34CPC( xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc"),
#                  start=1920, end=2020)[0]

dmi, dmi_2, dmi_3 = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
                         sst_anom_sd=False, opposite_signs_criteria=False)

aux = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")
n34, n34_2, n34_3 = Nino34CPC(aux, start=1920, end=2020)

del n34_2, dmi_2 # no importan

sam = sam.rolling(time=3, center=True).mean()
sam = NormSD(sam)
dmi_3 = NormSD(SameDateAs(dmi_3, sam))
n34 = NormSD(SameDateAs(n34, sam))
#------------------------------------------------------------------------------#
for s, mm in zip(seasons, mmonth):
    print(s + ' auxScatter...')
    dmi_un_pos, dmi_un_pos_sam_values, dmi_un_neg, dmi_un_neg_sam_values, \
    dmi_sim_pos, dmi_sim_pos_sam_values, n34_sim_pos_sam_values, \
    dmi_sim_neg, dmi_sim_neg_sam_values, n34_sim_neg_sam_values, \
    dmi_todos, sam_todos, dmi_pos_n34_neg, dmi_neg_n34_pos = \
        auxScatter(n34, n34_3, dmi, dmi_3, sam, mm)

    # Dataframe de fechas -----------------------------------------------------#
    ToCombinedDataframe(dmi_un_neg, dmi_un_neg_sam_values, out_dir_dataframe,
                        'dmi_un_neg_vs_sam_' + s)

    ToCombinedDataframe(dmi_un_pos, dmi_un_pos_sam_values, out_dir_dataframe,
                        'dmi_un_pos_vs_sam_' + s)

    ToCombinedDataframe(dmi_sim_pos, dmi_sim_pos_sam_values, out_dir_dataframe,
                        'dmi_sim_pos_vs_sam_' + s)

    ToCombinedDataframe(dmi_sim_neg, dmi_sim_neg_sam_values, out_dir_dataframe,
                        'dmi_sim_neg_vs_sam_' + s)
    # -------------------------------------------------------------------------#

    print('plot...')
    fig, ax = plt.subplots(dpi=dpi)
    # todos
    plt.scatter(x=sam_todos, y=dmi_todos, marker='.', label='SAM vs DMI',
                s=20, edgecolor='k', color='dimgray', alpha=1)
    # dmi puros
    plt.scatter(x=dmi_un_pos_sam_values.values, y=dmi_un_pos.values, marker='>',
                s=70, edgecolor='firebrick', facecolor='firebrick', alpha=1,
                label='IOD puro +')

    plt.scatter(y=dmi_un_neg.values, x=dmi_un_neg_sam_values.values, marker='<',
                s=70, facecolor='limegreen', edgecolor='limegreen', alpha=1,
                label='IOD puro -')

    # sim
    plt.scatter(y=dmi_sim_pos.values, x=n34_sim_pos_sam_values.values,
                marker='s', s=50,
                edgecolor='red', color='red', alpha=1, label='Niño & IOD+')
    plt.scatter(y=dmi_sim_neg.values, x=n34_sim_neg_sam_values.values,
                marker='s', s=50,
                edgecolor='deepskyblue', color='deepskyblue', alpha=1,
                label='Niña & IOD-')

    # sim opp. sing
    plt.scatter(x=dmi_pos_n34_neg.values, y=n34_sim_neg_sam_values.values,
                marker='s', s=50,
                edgecolor='orange', color='orange', alpha=1,
                label='Niña & IOD+')
    plt.scatter(x=dmi_neg_n34_pos.values, y=n34_sim_neg_sam_values.values,
                marker='s', s=50,
                edgecolor='gold', color='gold', alpha=1, label='Niño & IOD-')

    plt.legend(loc=(.7, .60))

    plt.ylim((-5, 5))
    plt.xlim((-5, 5))
    plt.axhspan(-.5, .5, alpha=0.2, color='black', zorder=0)
    plt.axvspan(-.5, .5, alpha=0.2, color='black', zorder=0)
    ax.grid(True)
    fig.set_size_inches(6, 6)
    plt.xlabel('SAM', size=15)
    plt.ylabel('DMI', size=15)

    plt.text(-4.8, 4.6, 'SAM-/IOD+', dict(size=15))
    plt.text(-.5, 4.6, 'IOD+', dict(size=15))
    plt.text(+2.5, 4.6, 'SAM+/IOD+', dict(size=15))
    plt.text(+3.5, -.1, 'SAM+', dict(size=15))
    plt.text(+2.5, -4.9, 'IOD-/SAM+', dict(size=15))
    plt.text(-.5, -4.9, 'IOD-', dict(size=15))
    plt.text(-4.8, -4.9, ' IOD-/SAM-', dict(size=15))
    plt.text(-4.8, -.1, 'SAM-', dict(size=15))
    plt.title(s + ' - ' + 'OBS')
    plt.tight_layout()
    if save:
        plt.savefig(
            out_dir + 'IOD_SAM_scatter' + s + '_OBS.jpg')
    else:
        plt.show()
#------------------------------------------------------------------------------#
print('#######################################################################')
print('done')
print('#######################################################################')
################################################################################