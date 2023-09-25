"""
Scatter IOD-SAM identificando los IOD según su ocurrencia con el ENSO
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
save = False
dataframes = True
seasons = ['JJA', 'SON']
mmonth = [7, 10]
#CreateDirectory(out_dir, dir_results)
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
sam = NormSD(sam.sel(time=sam.time.dt.month.isin(10))[1:-1])
dmi = NormSD(SameDateAs(dmi_3, sam))
n34 = NormSD(SameDateAs(n34, sam))

#------------------------------------------------------------------------------#
era5_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
hgt = SameDateAs(xr.open_dataset(era5_dir + 'HGT200_SON_mer_d_w.nc'), sam)
import xarray as xr
import numpy as np
import statsmodels.api as sm

aux_hgt = hgt.sel(lon=slice(270, 330), lat=slice(-20, -90))

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

coef_dataset, pvalues = compute_regression(hgt, n34, dmi, sam)
################################################################################
