"""
Scatter IOD-SAM identificando los IOD según su ocurrencia con el ENSO
"""
################################################################################
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
era5_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
out_dir_dataframe = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
dir_results = 'mlr'
################################################################################
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2, CreateDirectory, \
    DirAndFile, LinearReg1_D, PlotReg

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
################################################################################
save = False
# seasons = ['JJA', 'SON']
# mmonth = [7, 10]
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
################################################################################
# indices
sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam = sam.rolling(time=3, center=True).mean()

dmi, dmi_2, dmi_3 = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
                         sst_anom_sd=False, opposite_signs_criteria=False)

aux = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")
n34, n34_2, n34_3 = Nino34CPC(aux, start=1920, end=2020)
del n34_2, dmi_2 # no importan

sam = NormSD(sam.sel(time=sam.time.dt.month.isin(10))[1:-1])
dmi = NormSD(SameDateAs(dmi_3, sam))
n34 = NormSD(SameDateAs(n34, sam))
#------------------------------------------------------------------------------#
hgt = SameDateAs(xr.open_dataset(era5_dir + 'HGT200_SON_mer_d_w.nc'), sam)
################################################################################
print('Correlaciones ---------------------------------------------------------')
r_dmi_n34, pv_dmi_n34 = pearsonr(dmi, n34)
r_dmi_sam, pv_dmi_sam = pearsonr(dmi, sam)
r_sam_n34, pv_sam_n34 = pearsonr(sam, n34)

print('dmi vs n34 r = ' + str(np.round(r_dmi_n34, 2)) +
      ', p-value = ' + str(np.round(pv_dmi_n34, 3)))
print('dmi vs sam r = ' + str(np.round(r_dmi_sam, 2)) +
      ', p-value = ' + str(np.round(pv_dmi_sam, 3)))
print('sam vs n34 r = ' + str(np.round(r_sam_n34, 2)) +
      ', p-value = ' + str(np.round(pv_sam_n34, 3)))

print('-----------------------------------------------------------------------')
print('Correlaciones sin el 3er indice ---------------------------------------')
n34_wo_dmi, dmi_wo_n34 = LinearReg1_D(dmi, n34)
sam_wo_dmi, dmi_wo_sam = LinearReg1_D(dmi, sam)
sam_wo_n34, n34_wo_sam = LinearReg1_D(n34, sam)

r_dmi_n34_wosam, pv_dmi_n34_wosam = pearsonr(dmi_wo_sam, n34_wo_sam)
r_dmi_sam_won34, pv_dmi_sam_won34 = pearsonr(dmi_wo_n34, sam_wo_n34)
r_sam_n34_wodmi, pv_sam_n34_wodmi = pearsonr(sam_wo_dmi, n34_wo_dmi)

print('dmi vs n34 sin sam r = ' + str(np.round(r_dmi_n34_wosam, 2)) +
      ', p-value = ' + str(np.round(pv_dmi_n34_wosam, 3)))
print('dmi vs sam sin n34 r = ' + str(np.round(r_dmi_sam_won34, 2)) +
      ', p-value = ' + str(np.round(pv_dmi_sam_won34, 3)))
print('sam vs n34 sin dmi r = ' + str(np.round(r_sam_n34_wodmi, 2)) +
      ', p-value = ' + str(np.round(pv_sam_n34_wodmi, 3)))

print('-----------------------------------------------------------------------')
predictor_values = [sam]
reg = sm.OLS(dmi.values, np.column_stack(list(predictor_values))).fit()
print(reg.params)
plt.plot(sam, label='sam')
plt.plot(reg.predict(), label='predict')
plt.legend()
plt.show()