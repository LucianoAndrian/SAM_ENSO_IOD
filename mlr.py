"""
MLR ENSO, SAM y IOD
"""
################################################################################
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/mlr/'
era5_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
t_pp_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_obs_d_w_c/'
################################################################################
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2, CreateDirectory, \
    PlotReg
import Scales_Cbars
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

def Plot(data, data_pv, pvalue, title, name_fig, dpi, save, i=0,
         VarName='hgt200'):

    if VarName.lower() == 'hgt200':
        SA_map = False
        pvalue_mask = 0.1
        sig = False
        sig_point = False
        two_variables = True
    else:
        SA_map = True
        pvalue_mask = 1
        sig = True
        sig_point = True
        two_variables = False

    aux = data * MakerMaskSig(data_pv, pvalue_mask)
    aux = aux['var'][:,:,i,i]
    aux2 = data['var'][:,:,i]

    cbar = Scales_Cbars.get_cbars(VarName)
    scale = Scales_Cbars.get_scales(VarName)

    PlotReg(data=aux,
            data_cor=data_pv.sel(pvalues=i), SA=SA_map,
            levels=scale, sig=sig,
            two_variables=two_variables, data2=aux2, sig2=False,
            levels2=scale, title=title, name_fig=name_fig,
            out_dir=out_dir, save=save, cmap=cbar, dpi=dpi,
            color_map='grey', sig_point=sig_point, color_sig='k', r_crit=pvalue)

def SelectSeason_and_Variable(sam, dmi, n34, VarName, mm):

    # Seleccion de seaons en los índices
    sam_season = NormSD(sam.sel(time=sam.time.dt.month.isin(mm)))
    dmi_season = NormSD(SameDateAs(dmi, sam_season))
    n34_season = NormSD(SameDateAs(n34, sam_season))

    # Variable en la season
    if mm==10:
        season_name = 'SON'
    elif mm==7:
        season_name = 'JJA'
    else:
        return print('Season solo JJA o SON')

    if VarName.lower() == 'hgt200':
        file = 'HGT200_' + season_name + '_mer_d_w.nc'
        path = era5_dir
    elif VarName.lower() == 'pp':
        file = 'ppgpcc_w_c_d_1_' + season_name + '.nc'
        path = t_pp_dir
    elif VarName.lower() == 't':
        file = 'tcru_w_c_d_0.25_' + season_name + '.nc'
        path = t_pp_dir

    var_season = xr.open_dataset(path + file)

    return sam_season, dmi_season, n34_season, var_season
################################################################################
# indices
sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam = sam.rolling(time=3, center=True).mean()

dmi = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

aux = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")
n34 = Nino34CPC(aux, start=1920, end=2020)[0]

################################################################################
# periodos = [[1940, 2020],[1958, 1978], [1983, 2004], [1970, 1989], [1990, 2009],
#             [1990,2020]]
periodos = [[1940, 2020],[1970, 1989], [1990, 2009], [1990,2020]]

for VarName in ['hgt200', 'pp', 't']:
    for mm, s_name in zip([7, 10], ['JJA', 'SON']):
        for p in periodos:
            print('###########################################################')
            print('Period: ' + str(p[0]) + ' - ' + str(p[1]))
            print('###########################################################')

            sam_season, dmi_season, n34_season, var_season = \
                SelectSeason_and_Variable(sam, dmi, n34, VarName, mm)

            sam_p = sam_season.sel(
                time=slice(str(p[0]) + '-10-01', str(p[1]) + '-10-01'))
            n34_p = SameDateAs(n34_season, sam_p)
            dmi_p = SameDateAs(dmi_season, sam_p)
            try:
                var_p = SameDateAs(var_season, sam_p)
            except:
                var_p = var_season.sel(
                    time=slice(str(p[0]) + '-10-01', str(p[1]) + '-10-31'))
                var_p['time'] = sam_p.time.values

            print('Regression ------------------------------------------------')
            print('Solo N34--------------------')
            n34_reg, n34_reg_pv = compute_regression(var_p, n34_p)
            Plot(n34_reg, n34_reg_pv, 0.1,
                 VarName +' - N34 - ' + s_name + ' - ' + str(p),
                 VarName + '_N34_full_'+ s_name + '_' + str(p[0]) + '-' +
                 str(p[1]), dpi, save, 0, VarName)

            print('Solo DMI--------------------')
            dmi_reg, dmi_reg_pv = compute_regression(var_p, dmi_p)
            Plot(dmi_reg, dmi_reg_pv, 0.1,
                 VarName +' - DMI - ' + s_name + ' - ' + str(p),
                 VarName + '_DMI_full_' + s_name + '_' + str(p[0]) + '-'
                 + str(p[1]), dpi, save, 0, VarName)

            print('Solo SAM--------------------')
            sam_reg, sam_reg_pv = compute_regression(var_p, sam_p)
            Plot(sam_reg, sam_reg_pv, 0.1,
                 VarName +' - SAM - ' + s_name + ' - ' + str(p),
                 VarName + '_SAM_full_' + s_name + '_' + str(p[0]) + '-' +
                 str(p[1]), dpi, save, 0, VarName)

            print('MLR ------------------------')
            mlr_reg, mlr_reg_pv = compute_regression(var_p, n34_p, dmi_p, sam_p)
            Plot(mlr_reg, mlr_reg_pv, 0.1,
                 VarName +' - N34|dmi_sam - ' + s_name + ' - ' + str(p),
                 VarName + '_N34_wo_dmi_sam_' + s_name + '_' + str(p[0]) + '-' +
                 str(p[1]), dpi, save, 0, VarName)

            Plot(mlr_reg, mlr_reg_pv, 0.1,
                 VarName +' DMI|n34_sam - ' + s_name + ' - ' + str(p),
                 VarName + '_DMI_wo_n34_sam_' + s_name + '_' + str(p[0]) + '-' +
                 str(p[1]), dpi, save, 1, VarName)

            Plot(mlr_reg, mlr_reg_pv, 0.1,
                 VarName +' SAM|dmi_n34 - ' + s_name + ' - ' + str(p),
                 VarName + '_SAM_wo_dmi_n34_' + s_name + '_' + str(p[0]) + '-' +
                 str(p[1]), dpi, save, 2, VarName)

            print('-----------------------------------------------------------')
            print('done')
            print('-----------------------------------------------------------')










