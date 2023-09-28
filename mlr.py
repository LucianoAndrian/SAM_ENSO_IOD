"""
Scatter IOD-SAM identificando los IOD según su ocurrencia con el ENSO
"""
################################################################################
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
era5_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
out_dir_dataframe = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/mlr/'
################################################################################
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2, CreateDirectory, \
    PlotReg

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
        #dask='parallelized' +21 seg
    )
    return coef_dataset, pvalues

def MakerMaskSig(data, pvalue):
    mask_sig = data.where(data <= pvalue)
    mask_sig = mask_sig.where(np.isnan(mask_sig), 1)

    return mask_sig



def Plot(data, data_pv, pvalue, title, name_fig, dpi, save, i=0):
    aux = data * MakerMaskSig(data_pv, pvalue)
    aux = aux['var'][:,:,i,i]
    aux2 = data['var'][:,:,i]

    from matplotlib import colors
    cbar = colors.ListedColormap(['#9B1C00', '#B9391B', '#CD4838', '#E25E55',
                                  '#F28C89', '#FFCECC',
                                  'white',
                                  '#B3DBFF', '#83B9EB', '#5E9AD7', '#3C7DC3',
                                  '#2064AF', '#014A9B'][::-1])
    cbar.set_over('#641B00')
    cbar.set_under('#012A52')
    cbar.set_bad(color='white')
    scale = [-150, -100, -75, -50, -25, -15, 0, 15, 25, 50, 75, 100, 150]

    PlotReg(data=aux,
            data_cor=n34_reg_pv, SA=False,
            levels=scale, sig=False,
            two_variables=True, data2=aux2, sig2=False,
            levels2=scale, title=title, name_fig=name_fig,
            out_dir=out_dir, save=save, cmap=cbar, dpi=dpi,
            color_map='grey', sig_point=False, color_sig='k')
################################################################################
# indices
sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam = sam.rolling(time=3, center=True).mean()

dmi, dmi_2, dmi_3 = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
                         sst_anom_sd=False, opposite_signs_criteria=False)

aux = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")
n34, n34_2, n34_3 = Nino34CPC(aux, start=1920, end=2020)
del n34_2, dmi_2 # no importan

sam_or = NormSD(sam.sel(time=sam.time.dt.month.isin(10))[1:-1])
dmi_or = NormSD(SameDateAs(dmi_3, sam))
n34_or = NormSD(SameDateAs(n34, sam))
#------------------------------------------------------------------------------#
hgt_or = xr.open_dataset(era5_dir + 'HGT200_SON_mer_d_w.nc')
################################################################################
periodos = [[1940, 2020], [1970, 1989], [1990, 2009], [2010, 2020],
            [1958, 1978], [1983, 2004]]

for p in periodos:
    print('###################################################################')
    print('Period: '+ str(p[0]) + ' - ' + str(p[1]))
    print('###################################################################')

    sam = sam_or.sel(time=slice(str(p[0])+'-10-01', str(p[1])+'-10-01'))
    n34 = SameDateAs(n34_or, sam)
    dmi = SameDateAs(dmi_or, sam)
    hgt = SameDateAs(hgt_or, sam)

    print('Regression --------------------------------------------------------')
    print('Solo N34--------------------')
    n34_reg, n34_reg_pv = compute_regression(hgt, n34)
    Plot(n34_reg, n34_reg_pv, 0.1, 'z200 - N34 - ' + str(p), 
         'z200_N34_full_' + str(p[0]) + '-' + str(p[1]), dpi, save)

    print('Solo DMI--------------------')
    dmi_reg, dmi_reg_pv = compute_regression(hgt, dmi)
    Plot(dmi_reg, dmi_reg_pv, 0.1, 'z200 - DMI - ' + str(p), 
         'z200_DMI_full_' + str(p[0]) + '-' + str(p[1]), dpi, save)

    print('Solo SAM--------------------')
    sam_reg, sam_reg_pv = compute_regression(hgt, sam)
    Plot(sam_reg, sam_reg_pv, 0.1, 'z200 - SAM - ' + str(p), 
         'z200_SAM_full_' + str(p[0]) + '-' + str(p[1]), dpi, save)

    print('MLR ------------------------')
    mlr_reg, mlr_reg_pv = compute_regression(hgt, n34, dmi, sam)
    Plot(mlr_reg, mlr_reg_pv, 0.1, 'z200 - N34|dmi_sam - ' + str(p),
         'z200_N34_wo_dmi_sam_' + str(p[0]) + '-' + str(p[1]), dpi, save, i=0)
    Plot(mlr_reg, mlr_reg_pv, 0.1, 'z200 - DMI|n34_sam - ' + str(p),
         'z200_DMI_wo_n34_sam_' + str(p[0]) + '-' + str(p[1]), dpi, save, i=1)
    Plot(mlr_reg, mlr_reg_pv, 0.1, 'z200 - SAM|dmi_n34 - ' + str(p),
         'z200_SAM_wo_dmi_n34_' + str(p[0]) + '-' + str(p[1]), dpi, save, i=2)

    print('-------------------------------------------------------------------')
    print('done')
    print('-------------------------------------------------------------------')








