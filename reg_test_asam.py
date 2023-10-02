"""
Scatter IOD-SAM identificando los IOD según su ocurrencia con el ENSO
"""
################################################################################
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/reg_test/'
era5_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
################################################################################
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2, CreateDirectory, \
    LinearReg1_D
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
################################################################################
save = True

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

def regression_func(target_values, *predictor_values):
    predictors_list = list(predictor_values)
    predictors = np.column_stack(predictors_list)
    model = sm.OLS(target_values, predictors)
    results = model.fit()
    return results.params, results.pvalues

def compute_regression(variable, *index):
    input_core_dims = [['time']] + [['time']] * len(index)

    coef_dataset, pvalues = xr.apply_ufunc(
        regression_func,
        variable, *index,
        input_core_dims=input_core_dims,
        output_core_dims=[['coefficients'], ['pvalues']],
        output_dtypes=[float, float],
        vectorize=True
    )
    return coef_dataset, pvalues

def Count(index):
    return len(np.where(index>0)[0]), len(np.where(index<0)[0])

def PlotLt(index, index_wo, index_wo_name, title, name_fig, dpi, save):
    fig = plt.figure(figsize=(6, 4), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(index)
    ax.plot(index_wo, label='without ' + index_wo_name)
    plt.xticks(np.arange(0, 8),
               [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010])
    plt.ylim(-1, 1)
    plt.title(title)
    plt.hlines(y=0, xmin=0, xmax=8, color='k')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

def CorrReg(periodos, sam, dmi, n34, mm, plot, *args):
    mm_n34 = mm_dmi = mm_sam = mm
    try:
        len(args)
        index_lagged, mm_lag = args[0].split('_')
        mm_lag = int(mm_lag)
        print(index_lagged + ' lag: ' + str(mm_lag))

        if index_lagged == 'sam':
            mm_sam = mm + mm_lag
        elif index_lagged == 'dmi':
            mm_dmi = mm + mm_lag
        elif index_lagged == 'n34':
            mm_n34 = mm + mm_lag
    except:
        index_lagged = ''
        mm_lag = 0

    sam = NormSD(sam.sel(time=sam.time.dt.month.isin(mm_sam)))
    dmi = NormSD(dmi.sel(time=dmi.time.dt.month.isin(mm_dmi)))
    n34 = NormSD(n34.sel(time=n34.time.dt.month.isin(mm_n34)))

    sam_dmi_corr = []
    dmi_n34_corr = []
    n34_sam_corr = []
    sam_dmi_corr_wo_n34 = []
    dmi_n34_corr_wo_sam = []
    n34_sam_corr_wo_dmi = []
    for p in periodos:
        print('###############################################################')
        print('Period: ' + str(p[0]) + ' - ' + str(p[1]))
        print('###############################################################')

        aux_sam = sam.sel(time=slice(str(p[0]) + '-' + str(mm_sam) + '-01',
                                     str(p[1]) + '-' + str(mm_sam) + '-01'))

        aux_n34 = n34.sel(time=slice(str(p[0]) + '-' + str(mm_n34) + '-01',
                                     str(p[1]) + '-' + str(mm_n34) + '-01'))

        aux_dmi = dmi.sel(time=slice(str(p[0]) + '-' + str(mm_dmi) + '-01',
                                     str(p[1]) + '-' + str(mm_dmi) + '-01'))

        # print('Distribución del índice --------------------------------------)
        # sam_pos, sam_neg = Count(aux_sam)
        # n34_pos, n34_neg = Count(aux_n34)
        # dmi_pos, dmi_neg = Count(aux_dmi)
        # print('SAM > 0: ' + str(sam_pos) + ' | SAM <0: ' + str(sam_neg))
        # print('N34 > 0: ' + str(n34_pos) + ' | N34 <0: ' + str(n34_neg))
        # print('DMI > 0: ' + str(dmi_pos) + ' | DMI <0: ' + str(dmi_neg))

        print('Correlaciones -------------------------------------------------')
        r_dmi_n34, pv_dmi_n34 = pearsonr(aux_dmi, aux_n34)
        r_dmi_sam, pv_dmi_sam = pearsonr(aux_dmi, aux_sam)
        r_sam_n34, pv_sam_n34 = pearsonr(aux_sam, aux_n34)

        print('dmi vs n34 r = ' + str(np.round(r_dmi_n34, 2)) +
              ', p-value = ' + str(np.round(pv_dmi_n34, 3)))
        print('dmi vs sam r = ' + str(np.round(r_dmi_sam, 2)) +
              ', p-value = ' + str(np.round(pv_dmi_sam, 3)))
        print('sam vs n34 r = ' + str(np.round(r_sam_n34, 2)) +
              ', p-value = ' + str(np.round(pv_sam_n34, 3)))

        print('---------------------------------------------------------------')
        print('Correlaciones sin el 3er indice -------------------------------')
        n34_wo_dmi, dmi_wo_n34 = LinearReg1_D(aux_dmi, aux_n34)
        sam_wo_dmi, dmi_wo_sam = LinearReg1_D(aux_dmi, aux_sam)
        sam_wo_n34, n34_wo_sam = LinearReg1_D(aux_n34, aux_sam)

        r_dmi_n34_wosam, pv_dmi_n34_wosam = pearsonr(dmi_wo_sam, n34_wo_sam)
        r_dmi_sam_won34, pv_dmi_sam_won34 = pearsonr(dmi_wo_n34, sam_wo_n34)
        r_sam_n34_wodmi, pv_sam_n34_wodmi = pearsonr(sam_wo_dmi, n34_wo_dmi)

        print('dmi vs n34 sin sam r = ' + str(np.round(r_dmi_n34_wosam, 2)) +
              ', p-value = ' + str(np.round(pv_dmi_n34_wosam, 3)))
        print('dmi vs sam sin n34 r = ' + str(np.round(r_dmi_sam_won34, 2)) +
              ', p-value = ' + str(np.round(pv_dmi_sam_won34, 3)))
        print('sam vs n34 sin dmi r = ' + str(np.round(r_sam_n34_wodmi, 2)) +
              ', p-value = ' + str(np.round(pv_sam_n34_wodmi, 3)))

        dmi_n34_corr.append(r_dmi_n34)
        dmi_n34_corr_wo_sam.append(r_dmi_n34_wosam)

        n34_sam_corr.append(r_sam_n34)
        n34_sam_corr_wo_dmi.append(r_sam_n34_wodmi)

        sam_dmi_corr.append(r_dmi_sam)
        sam_dmi_corr_wo_n34.append(r_dmi_sam_won34)

        print(' MLR ----------------------------------------------------------')
        predictor_values = [aux_n34, aux_dmi]
        reg = sm.OLS(aux_sam.values,
                     np.column_stack(list(predictor_values))).fit()
        print('SAM = ' + str(np.round(reg.params[0], 3)) + '* N34 ' +
              str(np.round(reg.params[1], 3)) + '* DMI')

    if plot:
        PlotLt(dmi_n34_corr, dmi_n34_corr_wo_sam, 'SAM',
               'Correlación Niño3.4 vs DMI por decadas' + '\n' + index_lagged +
               'Lag = ' + str(mm_lag),
               'lt_r_n34_dmi_lag' + index_lagged + str(mm_lag), dpi,
               save)

        PlotLt(n34_sam_corr, n34_sam_corr_wo_dmi, 'DMI',
               'Correlación Niño3.4 vs asam por decadas' + '\n' + index_lagged +
               'Lag = ' + str(mm_lag),
               'lt_r_n34_asam_lag' + index_lagged + str(mm_lag), dpi,
               save)

        PlotLt(sam_dmi_corr, sam_dmi_corr_wo_n34, 'Niño 3.4',
               'Correlación asam vs DMI por decadas' + '\n' + index_lagged +
               'Lag = ' + str(mm_lag),
               'lt_r_asam_dmi_lag' + index_lagged + str(mm_lag), dpi, save)
################################################################################
# indices
sam = xr.open_dataset(sam_dir + 'asam_700.nc')['mean_estimate']
sam = sam.rolling(time=3, center=True).mean()

dmi, dmi_2, dmi_3 = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
                         sst_anom_sd=False, opposite_signs_criteria=False)

aux = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")
n34, n34_2, n34_3 = Nino34CPC(aux, start=1920, end=2020)
del n34_2, dmi_2 # no importan
dmi = dmi_3
################################################################################
periodos = [[1940, 2020], [1958, 1978], [1983, 2004], [1970, 1989],
            [1990, 2009]]

periodos_dec = [[1940, 1949], [1950, 1959], [1960, 1969], [1970, 1979],
                [1980, 1989], [1990, 1999], [2000, 2009], [2010, 2020]]

CorrReg(periodos_dec, sam, dmi, n34, 10, True)
print('-----------------------------------------------------------------------')
print('done')
print('-----------------------------------------------------------------------')














