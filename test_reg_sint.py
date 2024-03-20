"""
test campos sinteticos a partir de mlr
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
save = False
if save:
    dpi = 300
else:
    dpi = 100
################################################################################
def NormSD(serie):
    return serie / serie.std('time')


def RemoveEffect(predictand, predictor):
    model = sm.OLS(predictand, predictor)
    result = model.fit()
    predictand_res = predictand - result.predict()
    return predictand_res


def regression_func(target_values, *predictor_values):
    predictors_list = list(predictor_values)
    predictors = np.column_stack(predictors_list)
    predictors = sm.add_constant(predictors)
    model = sm.OLS(target_values, predictors, hasconst=False)
    results = model.fit()
    return results.params, results.predict(), results.pvalues

def compute_regression(variable, *index):
    input_core_dims = [['time']] + [['time']] * len(index)

    coef_dataset, predict, pvalues = xr.apply_ufunc(
        regression_func,
        variable, *index,
        input_core_dims=input_core_dims,
        output_core_dims=[['predict'],['coefficients'], ['pvalues']],
        output_dtypes=[float, float, float],
        vectorize=True
        #dask='parallelized' +21 seg
    )
    return coef_dataset, predict, pvalues


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

aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
n34 = Nino34CPC(aux, start=1920, end=2020)[0]

################################################################################
# periodos = [[1940, 2020],[1958, 1978], [1983, 2004], [1970, 1989], [1990, 2009],
#             [1990,2020]]
periodos = [[1940, 2020]]

# for VarName in ['hgt200']:
#     for mm, s_name in zip([10], ['SON']):
#         for p in periodos:
#
VarName= 'hgt200'
mm = 10
s_name = 'SON'
p = periodos[0]

print('###########################################################')
print('Period: ' + str(p[0]) + ' - ' + str(p[1]))
print('###########################################################')

sam_season, dmi_season, n34_season, var_season = \
    SelectSeason_and_Variable(sam, dmi, n34, VarName, mm)

sam_p = sam_season.sel(time=slice(str(p[0]) + '-10-01', str(p[1]) + '-10-01'))
n34_p = SameDateAs(n34_season, sam_p)
dmi_p = SameDateAs(dmi_season, sam_p)
try:
    var_p = SameDateAs(var_season, sam_p)
except:
    var_p = var_season.sel(time=slice(str(p[0]) + '-10-01',
                                      str(p[1]) + '-10-31'))
    var_p['time'] = sam_p.time.values

var_p = var_p.interp(lon=np.arange(0,360,.5), lat=np.arange(-80, 20, .5)[::-1])

print('Regression ------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print('N34 full ---------------------------')
n34_full_params, n34_full_pred, n34_full_pv = compute_regression(var_p, n34_p)

print('N34 without DMI --------------------')
aux = RemoveEffect(n34_p.values, dmi_p.values)
n34_wodmi_params, n34_wodmi_pred, n34_wodmi_pv = compute_regression(var_p, aux)

print('N34 without SAM --------------------')
aux = RemoveEffect(n34_p.values, sam_p.values)
n34_wosam_params, n34_wosam_pred, n34_wosam_pv = compute_regression(var_p, aux)

print('-----------------------------------------------------------------------')
print('DMI full ---------------------------')
dmi_full_params, dmi_full_pred, dmi_full_pv = compute_regression(var_p, dmi_p)

print('DMI without N34 --------------------')
aux = RemoveEffect(dmi_p.values, n34_p.values)
dmi_won34_params, dmi_won34_pred, dmi_won34_pv = compute_regression(var_p, aux)

print('DMI without SAM --------------------')
aux = RemoveEffect(dmi_p.values, sam_p.values)
dmi_wosam_params, dmi_wosam_pred, dmi_wosam_pv = compute_regression(var_p, aux)

print('-----------------------------------------------------------------------')
print('SAM full ---------------------------')
sam_full_params, sam_full_pred, sam_full_pv = compute_regression(var_p, sam_p)

print('SAM without N34 --------------------')
aux = RemoveEffect(sam_p.values, n34_p.values)
sam_won34_params, sam_won34_pred, sam_won34_pv = compute_regression(var_p, aux)

print('SAM without DMI --------------------')
aux = RemoveEffect(sam_p.values, sam_p.values)
sam_wodmi_params, sam_wodmi_pred, sam_wodmi_pv = compute_regression(var_p, aux)

print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print('DMI + N34 --------------------------')
mlr_full_params, mlr_full_pred, mlr_full_pv = \
    compute_regression(var_p, dmi_p, n34_p)

print('DMI + N34 wo. --------------------------')
dmi_wo_n34 = RemoveEffect(dmi_p.values, n34_p.values)
n34_wo_dmi = RemoveEffect(n34_p.values, dmi_p.values)
mlr_wo_params, mlr_wo_pred, mlr_wo_pv = \
    compute_regression(var_p, dmi_wo_n34, n34_wo_dmi)

print('-----------------------------------------------------------')
print('done')
print('-----------------------------------------------------------')

#            Plot(n34_reg, n34_reg_pv, 0.1,
#                  VarName +' - N34 - ' + s_name + ' - ' + str(p),
#                  VarName + '_N34_full_'+ s_name + '_' + str(p[0]) + '-' +
#                  str(p[1]), dpi, save, 0, VarName)

