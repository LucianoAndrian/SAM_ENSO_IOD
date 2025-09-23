"""
Scatter N34-SAM identificando los IOD según su ocurrencia con el ENSO
"""
################################################################################
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir_dataframe = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
dir_results = 'scatter'
################################################################################
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2, CreateDirectory, \
    DirAndFile

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
################################################################################
save = True
dataframes = True
seasons = ['JJA', 'SON']
mmonth = [7, 10]
CreateDirectory(out_dir, dir_results)
if save:
    dpi = 300
else:
    dpi = 100
################################################################################
def NormSD(serie):
    return serie / serie.std('time')

def RemoveYear(data1, data2):
    return data1.sel(
        time=data2.time[xr.ufuncs.isnan(data2.values)])

def auxScatter(n34, n34_3, dmi, dmi_3, sam, s):
    """
    Hace lo mismo que en ENSO_IOD_Satter_OBS.py pero ahora devuelve los valores
    de SAM para cada case de N34.

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

    n34_pos_dmi_neg = n34_sim_pos.where(~np.isnan(dmi_sim_neg.values))
    n34_neg_dmi_pos = n34_sim_neg.where(~np.isnan(dmi_sim_pos.values))

    try:
        n34_sim_pos =RemoveYear(n34_sim_pos, n34_pos_dmi_neg)
        n34_sim_pos_sam_values = RemoveYear(n34_sim_pos_sam_values, n34_pos_dmi_neg)

        n34_sim_neg =RemoveYear(n34_sim_neg, n34_neg_dmi_pos)
        n34_sim_neg_sam_values = RemoveYear(n34_sim_neg_sam_values, n34_neg_dmi_pos)

    except:
        pass

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

    aux_dates_ref = n34.time.dt.year
    mask = np.in1d(aux_dates_ref, sim_y, invert=True)
    n34_un = aux_n34.sel(time=aux_n34.time.dt.year.isin(aux_dates_ref[mask]))

    n34_un_pos = n34_un.where(n34_un > 0)
    n34_un_pos_sam_values = sam_todos.sel(
        time=sam_todos.time.isin(n34_un_pos.time))
    n34_un_neg = n34_un.where(n34_un < 0)
    n34_un_neg_sam_values = sam_todos.sel(
        time=sam_todos.time.isin(n34_un_neg.time))

    return n34_un_pos, n34_un_pos_sam_values, \
           n34_un_neg, n34_un_neg_sam_values, \
           n34_sim_pos, n34_sim_pos_sam_values, dmi_sim_pos_sam_values,\
           n34_sim_neg, n34_sim_neg_sam_values, dmi_sim_neg_sam_values,\
           n34_todos, sam_todos, \
           n34_pos_dmi_neg, n34_neg_dmi_pos

def ToCombinedDataframe(n34, sam, out_dir, name):
    n34_df = n34.to_dataframe(name='n34')
    sam_df = sam.to_dataframe(name='sam')

    combined_df = pd.concat([n34_df, sam_df], axis=1)
    # Hay valores con NaN x como se calculan en auxScatter()
    combined_df = combined_df.dropna(subset=['n34'])

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
dmi_3 = NormSD(SameDateAs(dmi_3, sam))
n34 = NormSD(SameDateAs(n34, sam))
#------------------------------------------------------------------------------#
for sam_component in ['sam', 'asam', 'ssam']:
    print('###################################################################')
    print(sam_component)
    print('###################################################################')

    sam = xr.open_dataset(sam_dir + sam_component +'_700.nc')['mean_estimate']
    sam = sam.rolling(time=3, center=True).mean()
    sam = NormSD(sam)

    for s, mm in zip(seasons, mmonth):

        print(s + ' auxScatter...')
        n34_un_pos, n34_un_pos_sam_values, n34_un_neg, n34_un_neg_sam_values, \
        n34_sim_pos, n34_sim_pos_sam_values, dmi_sim_pos_sam_values, \
        n34_sim_neg, n34_sim_neg_sam_values, dmi_sim_neg_sam_values, \
        n34_todos, sam_todos, n34_pos_dmi_neg, n34_neg_dmi_pos = \
            auxScatter(n34, n34_3, dmi, dmi_3, sam, mm)

        # Dataframe de fechas -------------------------------------------------#
        if dataframes:
            ToCombinedDataframe(n34_un_neg, n34_un_neg_sam_values,
                                out_dir_dataframe,
                                'n34_un_neg_vs_' + sam_component + '_' + s)

            ToCombinedDataframe(n34_un_pos, n34_un_pos_sam_values,
                                out_dir_dataframe,
                                'n34_un_pos_vs_' + sam_component + '_' + s)

            ToCombinedDataframe(n34_sim_pos, n34_sim_pos_sam_values,
                                out_dir_dataframe,
                                'n34_sim_pos_vs_' + sam_component + '_' + s)

            ToCombinedDataframe(n34_sim_neg, n34_sim_neg_sam_values,
                                out_dir_dataframe,
                                'n34_sim_neg_vs_' + sam_component + '_' + s)
        # ---------------------------------------------------------------------#

        print('plot...')
        fig, ax = plt.subplots(dpi=dpi)
        # todos
        plt.scatter(x=sam_todos, y=n34_todos, marker='.', label='SAM vs N34',
                    s=20, edgecolor='k', color='dimgray', alpha=1)
        # dmi puros
        plt.scatter(x=n34_un_pos_sam_values.values, y=n34_un_pos.values,
                    marker='^',
                    s=70, edgecolors='navy', facecolor='navy', alpha=1,
                    label='El Niño puro')

        plt.scatter(y=n34_un_neg.values, x=n34_un_neg_sam_values.values,
                    marker='v',
                    s=70, edgecolors='deeppink', facecolor='deeppink', alpha=1,
                    label='La Niña puro')

        # sim
        plt.scatter(y=n34_sim_pos.values, x=n34_sim_pos_sam_values.values,
                    marker='s', s=50,
                    edgecolor='red', color='red', alpha=1, label='Niño & IOD+')
        plt.scatter(y=n34_sim_neg.values, x=n34_sim_neg_sam_values.values,
                    marker='s', s=50,
                    edgecolor='deepskyblue', color='deepskyblue', alpha=1,
                    label='Niña & IOD-')

        # sim opp. sing
        plt.scatter(x=n34_pos_dmi_neg.values, y=dmi_sim_neg_sam_values.values,
                    marker='s', s=50,
                    edgecolor='orange', color='orange', alpha=1,
                    label='Niña & IOD+')
        plt.scatter(x=n34_neg_dmi_pos.values, y=dmi_sim_neg_sam_values.values,
                    marker='s', s=50,
                    edgecolor='gold', color='gold', alpha=1,
                    label='Niño & IOD-')

        plt.legend(loc=(.7, .60))

        plt.ylim((-5, 5))
        plt.xlim((-5, 5))
        plt.axhspan(-.5, .5, alpha=0.2, color='black', zorder=0)
        plt.axvspan(-.5, .5, alpha=0.2, color='black', zorder=0)
        ax.grid(True)
        fig.set_size_inches(6, 6)
        sname = sam_component.upper()
        plt.xlabel(sname, size=15)
        plt.ylabel('N34', size=15)
        plt.text(-4.8, 4.6, sname + '-/El Niño', dict(size=15))
        plt.text(-.8, 4.6, 'El Niño', dict(size=15))
        plt.text(+2.1, 4.6, sname + '+/El Niño', dict(size=15))
        plt.text(+3.5, -.1, sname + '+', dict(size=15))
        plt.text(+2.1, -4.9, 'La Niña/' + sname + '+', dict(size=15))
        plt.text(-.8, -4.9, 'La Niña', dict(size=15))
        plt.text(-4.8, -4.9, 'La Niña/' + sname + '-', dict(size=15))
        plt.text(-4.8, -.1, sname + '-', dict(size=15))
        plt.title(sname + ' vs N34 - ' + s)
        plt.tight_layout()
        if save:
            plt.savefig(DirAndFile(out_dir, dir_results, s,
                                   ['N34', sam_component]))
        else:
            plt.show()
# -----------------------------------------------------------------------------#
print('#######################################################################')
print('done')
print('out_dir = ' + out_dir)
print('#######################################################################')
################################################################################