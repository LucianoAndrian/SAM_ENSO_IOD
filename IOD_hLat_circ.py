"""
IOD - SAM/U50
"""
# ---------------------------------------------------------------------------- #
save = False
sig_thr = 0.05

out_dir = '/pikachu/datos/luciano.andrian/IOD_vs_hLat/'
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import dcor
import IOD_hLat_cic_SET_VAR
# ---------------------------------------------------------------------------- #
if save:
    dpi = 200
else:
    dpi = 100
# ---------------------------------------------------------------------------- #
def SelectMonths(data, months_to_select, years_to_remove=None):
    """
    Selecciona meses
    :param series: xr.dataset o xr.dataarray o lista de ellos
    :param months_to_select: int o lista de meses
    :param years_to_remove: int o lista de años a quitar, default None
    :return: data con meses seleccionados
    """
    output = None
    if isinstance(data, list):
        output = []
        for d in data:
            d_aux = d.sel(time=d.time.dt.month.isin(months_to_select))
            if years_to_remove is not None:
                d_aux = d_aux.sel(time=~d_aux.time.dt.year.isin(years_to_remove))
            output.append(d_aux)
        output = tuple(output)
    else:
        output = data.sel(time=data.time.dt.month.isin(months_to_select))
        output = output.sel(time=~output.time.dt.year.isin(years_to_remove))

    return output

# Función para calcular el p-valor usando permutaciones
def permutation_test(x, y, num_permutations=500):
    # Correlación real
    real_corr = dcor.distance_correlation(x, y)

    # Generar correlaciones permutadas
    permuted_corrs = []
    for _ in range(num_permutations):
        y_permuted = np.random.permutation(y)
        permuted_corrs.append(dcor.distance_correlation(x, y_permuted))

    # Calcular el p-valor (frecuencia de permutaciones >= correlación real)
    p_value = np.mean(np.abs(permuted_corrs) >= np.abs(real_corr))
    return real_corr, p_value

def CorrelationMatrix(data1, data2, years_to_remove=[2002, 2019]):

    from scipy.stats import pearsonr, spearmanr

    pearson_corr = np.zeros((12, 12))
    pearson_corr_pv = np.zeros((12, 12))
    spearman_corr = np.zeros((12, 12))
    spearman_corr_pv = np.zeros((12, 12))
    distance_corr = np.zeros((12, 12))
    distance_corr_pv = np.zeros((12, 12))

    for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        data1_m = SelectMonths(data1, m, years_to_remove)
        for m2 in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            data2_m = SelectMonths(data2, m2, years_to_remove)

            pr = pearsonr(data1_m, data2_m)
            pearson_corr[m-1, m2-1] = pr[0]
            pearson_corr_pv[m-1, m2-1] = pr[1]

            sp = spearmanr(data1_m, data2_m)
            spearman_corr[m-1, m2-1] = sp[0]
            spearman_corr_pv[m-1, m2-1] = sp[1]

            dc = permutation_test(data1_m, data2_m)
            distance_corr[m-1, m2-1] = dc[0]
            distance_corr_pv[m-1, m2-1] = dc[1]

    return (pearson_corr, pearson_corr_pv, spearman_corr, spearman_corr_pv,
            distance_corr, distance_corr_pv)


def PlotMatrix(matrix, matrix_sig, sig_thr=0.05, cmap='RdBu_r', title='Corr',
               save=save, name_fig='fig_matrix', scale=(-1,1), x_label='x',
               y_label='y'):
    import matplotlib.pyplot as plt
    fig = plt.figure(dpi=dpi, figsize=(8,7))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, cmap=cmap, vmin=min(scale), vmax=max(scale))

    for i in range(0, 12):
        for j in range(0, 12):
            if np.abs(matrix_sig[i,j])<=sig_thr:
                ax.text(j, i, np.round(matrix[i, j], 2),
                        ha='center', va='center', color='k')

    ax.set_ylabel(f'Meses - {y_label}', fontsize=11)
    ax.set_xlabel(f'Meses - {x_label}', fontsize=11)
    fig.suptitle(title, size=12)

    ax.set_xticks(np.arange(0, 12), major=True)
    ax.set_yticks(np.arange(0, 12), major=True)
    ax.set_xticklabels(np.arange(1, 13))
    ax.set_yticklabels(np.arange(1, 13))
    ax.margins(0)

    ax.set_xticks(np.arange(0, 12, 0.5), minor=True)
    ax.set_yticks(np.arange(0, 12, 0.5), minor=True)

    ax.grid(which='minor', alpha=0.5, color='k')

    plt.tight_layout()
    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

def PlotScatter(x, y, x_label='x', y_label='y', title='', save=save,
                name_fig='fig_scatter', text=True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(dpi=dpi, figsize=(7.08661, 7.08661))
    in_label_size = 13
    label_legend_size = 12
    tick_label_size = 11
    scatter_size_fix = 3
    # todos
    ax.scatter(x=x, y=y, marker='o', label=f'{x_label} vs {y_label}',
               s=30 * scatter_size_fix, edgecolor='k', color='dimgray', alpha=1)

    if text:
        for i in range(0, len(x)):
            ax.text(x[i], y[i], x.time.dt.year.values[i],
                    verticalalignment='bottom', horizontalalignment='center')

    ax.legend(fontsize=label_legend_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_label_size, pad=1)
    ax.set_ylim((-5, 5))
    ax.set_xlim((-5, 5))
    ax.axhspan(-.5, .5, alpha=0.2, color='black', zorder=0)
    ax.axvspan(-.5, .5, alpha=0.2, color='black', zorder=0)
    ax.set_xlabel(x_label, size=in_label_size)
    ax.set_ylabel(y_label, size=in_label_size)
    ax.set_title(title, size=label_legend_size)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig, dpi=dpi, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()


def PlotAllMatrices(pearson_1rm, pearson_sig_1rm, spearman_1rm,
                    spearman_sig_1rm, distance_1rm, distance_sig_1rm,
                    pearson_2rm, pearson_sig_2rm, spearman_2rm,
                    spearman_sig_2rm, distance_2rm, distance_sig_2rm,
                    pearson_3rm, pearson_sig_3rm, spearman_3rm,
                    spearman_sig_3rm, distance_3rm, distance_sig_3rm,
                    sig_thr=0.05, cmap='RdBu_r',
                    save=False, name_fig='all_matrices', scale=(-1, 1),
                    title='', y_label='x', x_label='x', dpi=dpi,
                    out_dir=out_dir):

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(3, 3, figsize=(17.5, 17), dpi=dpi)
    matrices = [
        (pearson_1rm, pearson_sig_1rm, 'Pearson Correlation (1M)'),
        (spearman_1rm, spearman_sig_1rm, 'Spearman Correlation (1M)'),
        (distance_1rm, distance_sig_1rm, 'Distance Correlation (1M)'),
        (pearson_2rm, pearson_sig_2rm, 'Pearson Correlation (2M)'),
        (spearman_2rm, spearman_sig_2rm, 'Spearman Correlation (2M)'),
        (distance_2rm, distance_sig_2rm, 'Distance Correlation (2M)'),
        (pearson_3rm, pearson_sig_3rm, 'Pearson Correlation (3M)'),
        (spearman_3rm, spearman_sig_3rm, 'Spearman Correlation (3M)'),
        (distance_3rm, distance_sig_3rm, 'Distance Correlation (3M)')]

    for i, ax in enumerate(axes.flatten()):
        matrix, matrix_sig, label = matrices[i]
        im = ax.imshow(matrix, cmap=cmap,
                       vmin=min(scale),
                       vmax=max(scale))
        ax.set_title(label, fontsize=10)
        ax.set_xticks(np.arange(12), major=True)
        ax.set_yticks(np.arange(12), major=True)
        ax.set_xticklabels(np.arange(1, 13))
        ax.set_yticklabels(np.arange(1, 13))
        ax.set_ylabel(f'Meses - {y_label}', fontsize=11)
        ax.set_xlabel(f'Meses - {x_label}', fontsize=11)

        ax.margins(0)

        ax.set_xticks(np.arange(0, 12, 0.5), minor=True)
        ax.set_yticks(np.arange(0, 12, 0.5), minor=True)

        ax.grid(which='minor', alpha=0.5, color='k')

        for x in range(12):
            for y in range(12):
                if np.abs(matrix_sig[x, y]) <= sig_thr:
                    ax.text(y, x, f'{matrix[x, y]:.2f}',
                            ha='center', va='center', color='k')


    fig.suptitle(title, fontsize=16, y=0.97)

    # Agregar la barra de color
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.subplots_adjust(bottom=0, hspace=0.05, left=0.05, top=0.95)

    # Guardar o mostrar la figura
    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

# ---------------------------------------------------------------------------- #
(dmi_or_1rm, dmi_or_2rm, dmi_or_3rm, sam_or_1rm, sam_or_2rm, sam_or_3rm,
 u50_or_1rm, u50_or_2rm, u50_or_3rm, hgt200_anom_or_1rm, hgt200_anom_or_2rm,
 hgt200_anom_or_3rm) = IOD_hLat_cic_SET_VAR.compute()
# ---------------------------------------------------------------------------- #
# from ENSO_IOD_Funciones import (Nino34CPC, Nino34CPC_singlemonth,
#                                 Nino34CPC_twomonths)
#
# sst_aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/"
#                       "sst.mnmean.nc")
#
# sst_aux = sst_aux.sel(time=slice('1920-01-01', '2020-12-01'))
# n34_or_3rm = Nino34CPC(sst_aux, start=1920, end=2020)[0]
# n34_or_1rm = Nino34CPC_singlemonth(sst_aux, start=1920, end=2020)[0]
# n34_or_2rm = Nino34CPC_twomonths(sst_aux, start=1920, end=2020)[0]
#
# n34_or_1rm, n34_or_2rm, n34_or_3rm = ddn([n34_or_1rm, n34_or_2rm, n34_or_3rm],
#                                          dmi_or_1rm)

# ---------------------------------------------------------------------------- #
# Que onda la relacion entre SAM/U50 y el IOD?
# ---------------------------------------------------------------------------- #
# U50 vs SAM ----------------------------------------------------------------- #
# es como usar lo mismo?
(pearson_corr, pearson_corr_pv, spearman_corr, spearman_corr_pv,
 distance_corr, distance_corr_pv) = CorrelationMatrix(u50_or_1rm['var'],
                                                      sam_or_1rm)

(pearson_corr_2rm, pearson_corr_pv_2rm, spearman_corr_2rm, spearman_corr_pv_2rm,
 distance_corr_2rm, distance_corr_pv_2rm) = CorrelationMatrix(u50_or_2rm['var'],
                                                      sam_or_2rm)

(pearson_corr_3rm, pearson_corr_pv_3rm, spearman_corr_3rm, spearman_corr_pv_3rm,
 distance_corr_3rm, distance_corr_pv_3rm) = CorrelationMatrix(u50_or_3rm['var'],
                                                      sam_or_3rm)

PlotAllMatrices(pearson_corr, pearson_corr_pv, spearman_corr,
                spearman_corr_pv, distance_corr, distance_corr_pv,
                pearson_corr_2rm, pearson_corr_pv_2rm, spearman_corr_2rm,
                spearman_corr_pv_2rm, distance_corr_2rm, distance_corr_pv_2rm,
                pearson_corr_3rm, pearson_corr_pv_3rm, spearman_corr_3rm,
                spearman_corr_pv_3rm, distance_corr_3rm, distance_corr_pv_3rm,
                sig_thr=sig_thr, cmap='RdBu_r', save=save,
                name_fig='sam_vs_u50', title='SAM vs U50', x_label='SAM',
                y_label='U50')

# SAM vs DMI ----------------------------------------------------------------- #
(pearson_corr, pearson_corr_pv, spearman_corr, spearman_corr_pv,
 distance_corr, distance_corr_pv) = CorrelationMatrix(sam_or_1rm,
                                                      dmi_or_1rm)

(pearson_corr_2rm, pearson_corr_pv_2rm, spearman_corr_2rm, spearman_corr_pv_2rm,
 distance_corr_2rm, distance_corr_pv_2rm) = (
    CorrelationMatrix(sam_or_2rm, dmi_or_2rm))

(pearson_corr_3rm, pearson_corr_pv_3rm, spearman_corr_3rm, spearman_corr_pv_3rm,
 distance_corr_3rm, distance_corr_pv_3rm) = (
    CorrelationMatrix(sam_or_3rm, dmi_or_3rm))

PlotAllMatrices(pearson_corr, pearson_corr_pv, spearman_corr,
                spearman_corr_pv, distance_corr, distance_corr_pv,
                pearson_corr_2rm, pearson_corr_pv_2rm, spearman_corr_2rm,
                spearman_corr_pv_2rm, distance_corr_2rm, distance_corr_pv_2rm,
                pearson_corr_3rm, pearson_corr_pv_3rm, spearman_corr_3rm,
                spearman_corr_pv_3rm, distance_corr_3rm, distance_corr_pv_3rm,
                sig_thr=sig_thr, cmap='RdBu_r', save=save,
                name_fig='dmi_vs_sam', title='DMI vs SAM', x_label='DMI',
                y_label='SAM')

# U50 vs DMI ----------------------------------------------------------------- #
(pearson_corr, pearson_corr_pv, spearman_corr, spearman_corr_pv,
 distance_corr, distance_corr_pv) = CorrelationMatrix(u50_or_1rm['var'],
                                                      dmi_or_1rm)

(pearson_corr_2rm, pearson_corr_pv_2rm, spearman_corr_2rm, spearman_corr_pv_2rm,
 distance_corr_2rm, distance_corr_pv_2rm) = (
    CorrelationMatrix(u50_or_2rm['var'], dmi_or_2rm))

(sampearson_corr_3rm, pearson_corr_pv_3rm, spearman_corr_3rm, spearman_corr_pv_3rm,
 distance_corr_3rm, distance_corr_pv_3rm) = (
    CorrelationMatrix(u50_or_3rm['var'], dmi_or_3rm))

PlotAllMatrices(pearson_corr, pearson_corr_pv, spearman_corr,
                spearman_corr_pv, distance_corr, distance_corr_pv,
                pearson_corr_2rm, pearson_corr_pv_2rm, spearman_corr_2rm,
                spearman_corr_pv_2rm, distance_corr_2rm, distance_corr_pv_2rm,
                pearson_corr_3rm, pearson_corr_pv_3rm, spearman_corr_3rm,
                spearman_corr_pv_3rm, distance_corr_3rm, distance_corr_pv_3rm,
                sig_thr=sig_thr, cmap='RdBu_r', save=save,
                name_fig='dmi_vs_u50', title='DMI vs U50', x_label='DMI',
                y_label='U50')

# Para control, N34 - DMI ---------------------------------------------------- #
# (pearson_corr, pearson_corr_pv, spearman_corr, spearman_corr_pv,
#  distance_corr, distance_corr_pv) = CorrelationMatrix(n34_or_1rm,
#                                                       dmi_or_1rm)
#
# (pearson_corr_2rm, pearson_corr_pv_2rm, spearman_corr_2rm, spearman_corr_pv_2rm,
#  distance_corr_2rm, distance_corr_pv_2rm) = (
#     CorrelationMatrix(n34_or_2rm, dmi_or_2rm))
#
# (pearson_corr_3rm, pearson_corr_pv_3rm, spearman_corr_3rm, spearman_corr_pv_3rm,
#  distance_corr_3rm, distance_corr_pv_3rm) = (
#     CorrelationMatrix(n34_or_3rm, dmi_or_3rm))
#
# PlotAllMatrices(pearson_corr, pearson_corr_pv, spearman_corr,
#                 spearman_corr_pv, distance_corr, distance_corr_pv,
#                 pearson_corr_2rm, pearson_corr_pv_2rm, spearman_corr_2rm,
#                 spearman_corr_pv_2rm, distance_corr_2rm, distance_corr_pv_2rm,
#                 pearson_corr_3rm, pearson_corr_pv_3rm, spearman_corr_3rm,
#                 spearman_corr_pv_3rm, distance_corr_3rm, distance_corr_pv_3rm,
#                 sig_thr=sig_thr, cmap='RdBu_r', save=save,
#                 name_fig='dmi_vs_n34', title='DMI vs N34', x_label='DMI',
#                 y_label='N34')
# ---------------------------------------------------------------------------- #

aux_x = SelectMonths(sam_or_1rm, 8, [2002,2019])
aux_y = SelectMonths(u50_or_1rm['var'], 8, [2002,2019])
PlotScatter(aux_x, aux_y, x_label='SAM', y_label='U50', save=False)

# ---------------------------------------------------------------------------- #
from PCMCI import PCMCI

series = {'u50':u50_or_1rm['var'].values, 'sam':sam_or_1rm.values,
         'dmi':dmi_or_1rm.values}

for mm in [7,8,9,10,11]:
    print(f'mm: {mm}')
    links = PCMCI(series=series, tau_max=6, pc_alpha=0.1, mci_alpha=0.1,
                  mm=mm, w=0,
                  autocorr=False)

    print(links)

# ---------------------------------------------------------------------------- #