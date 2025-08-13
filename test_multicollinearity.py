"""
Testo simple de multicolinealidad
La idea no es usar VIF sino ver el cambio en los coeficientes

De los resultados previos un caso claro es con:
DMI, N34, U50 en ASO
con:
N34 -> DMI
DMI -> U50
N34 -> U50
"""
# ---------------------------------------------------------------------------- #
# Seteos generales de Trinity
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import numpy as np
import xarray as xr
import pandas as pd
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
from ENSO_IOD_Funciones import Nino34CPC, DMI2, ChangeLons, DMI2_singlemonth, \
    Nino34CPC_singlemonth, DMI2_twomonths, Nino34CPC_twomonths, MakeMask
from cen_funciones import OpenObsDataSet, Detrend, Weights, \
    auxSetLags_ActorList, aux_alpha_CN_Effect_2
from CEN_ufunc import CEN_ufunc
from ENSO_IOD_Funciones import SameDateAs
import statsmodels.api as sm
from cen_funciones import aux2_Setlag
# ---------------------------------------------------------------------------- #
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
hgt_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
dir_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_no_detrend/'

#
def regre(series, intercept, coef=0, filter_significance=True, alpha=1):
    df = pd.DataFrame(series)
    if intercept:
        X = sm.add_constant(df[df.columns[1:]])
    else:
        X = df[df.columns[1:]]
    y = df[df.columns[0]]

    model = sm.OLS(y, X).fit()
    coefs_results = model.params
    p_values = model.pvalues
    # t_values = model.tvalues

    results = {}
    for col in df.columns[1:]:
        if filter_significance:
            if p_values[col] <= alpha:
                results[col] = coefs_results[col]
            else:
                results[col] = None
        else:
            results[col] = coefs_results[col]

    if isinstance(coef, str):
        return results.get(coef, 0)
    else:
        return results

def beta_from_cov_MLR(y, x1, x2):
    # Varianzas
    var_x1 = np.var(x1, ddof=1)
    var_x2 = np.var(x2, ddof=1)

    # Covarianzas
    cov_yx1 = np.cov(y, x1, ddof=1)[0, 1]
    cov_yx2 = np.cov(y, x2, ddof=1)[0, 1]
    cov_x1x2 = np.cov(x1, x2, ddof=1)[0, 1]

    # b1
    termino_num_x2 = (cov_yx2 * cov_x1x2)/var_x2
    num = cov_yx1  - termino_num_x2

    termino_den_x2 = (cov_x1x2 ** 2)/var_x2
    den = var_x1  - termino_den_x2


    print(f'numerador = {np.round(cov_yx1,3)} - {np.round(termino_num_x2,3)}')
    print('Numerador:', np.round(num,3))

    print(f'denominador = {np.round(var_x1,3)} - {np.round(termino_den_x2,3)}')
    print('Denominador:', np.round(den,3))

    b1_rs = cov_yx1/var_x1
    b1_mlr = num/den

    print('Coeficiente regresion simple: ', np.round(b1_rs,3))
    print('Coeficiente regresion multiple: ', np.round(b1_mlr, 3))

    if np.abs(b1_rs) < np.abs(b1_mlr):
        aux = np.round(b1_mlr/b1_rs)
        print(f'Coef. MLR es ~{aux} veces > Coef. Regresion Simple')


    return b1_rs, b1_mlr

def vif(X):
    """
    Calcula el VIF para cada predictor en X.

    Parámetros:
    -----------
    X : pd.DataFrame
        DataFrame con las variables predictoras.

    Retorna:
    --------
    pd.DataFrame
        DataFrame con columnas: 'variable' y 'VIF'
    """

    X = pd.DataFrame(X)
    vif_data = []
    for i in range(X.shape[1]):
        X_i = X.iloc[:, i]
        X_others = X.drop(X.columns[i], axis=1)

        X_others_const = sm.add_constant(X_others)
        model = sm.OLS(X_i, X_others_const).fit()
        r_squared = model.rsquared
        vif = 1 / (1 - r_squared)
        vif_data.append({'variable': X.columns[i], 'VIF': vif})

    result = pd.DataFrame(vif_data)
    print(result)
    return result

def aux2_Setlag(serie_or, serie_lag, serie_set, years_to_remove):
    if serie_lag is not None:
        serie_f = serie_or.sel(
            time=serie_or.time.dt.month.isin([serie_lag]))
        serie_f = serie_f.sel(
            time=serie_f.time.dt.year.isin(serie_set.time.dt.year))
    else:
        serie_f = SameDateAs(serie_or, serie_set)

    serie_f = serie_f.sel(time=~serie_f.time.dt.year.isin(years_to_remove))

    serie_f = serie_f / serie_f.std()

    return serie_f

# Indices -------------------------------------------------------------------- #
year_start = 1959
year_end = 2020
dmi = DMI2(filter_bwa=False, start_per=str(year_start), end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

sst_aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/"
                      "sst.mnmean.nc")
sst_aux = sst_aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34 = Nino34CPC(sst_aux, start=1920, end=2020)[0]


u50_or = xr.open_dataset('/pikachu/datos/luciano.andrian/observado/'
                         'ncfiles/ERA5/downloaded/ERA5_U50hpa_40-20.mon.nc')
u50_or = u50_or.rename({'u': 'var'})
u50_or = u50_or.rename({'longitude': 'lon'})
u50_or = u50_or.rename({'latitude': 'lat'})
u50_or = Weights(u50_or)
u50_or = u50_or.sel(lat=-60)
#u50_or = u50_or - u50_or.mean('time')
u50_or = (u50_or.groupby('time.month') -
          u50_or.groupby('time.month').mean('time'))

u50 = u50_or.rolling(time=1, center=True).mean()
u50 = Detrend(u50, 'time')
u50 = u50.sel(expver=1).drop('expver')
u50 = u50.mean('lon')
u50 = xr.DataArray(u50['var'].drop('lat'))

# ASO ------------------------------------------------------------------------ #
u50 = u50.drop('month')
u50 = u50.sel(time=u50.time.dt.year.isin(range(year_start, year_end+1)))

# ---------------------------------------------------------------------------- #

dmi_l = aux2_Setlag(dmi, 10, dmi, years_to_remove=[2002, 2019])
n34_l = aux2_Setlag(n34, 10, dmi, years_to_remove=[2002, 2019])
u50_l = aux2_Setlag(u50, 9, dmi, years_to_remove=[2002, 2019])

# Regresion simple
series = {'u50':u50_l.values, 'n34':n34_l.values}
b_total_n34 = regre(series, True, 'n34', False, 0.05)

series = {'u50':u50_l.values, 'dmi':dmi_l.values}
b_total_dmi = regre(series, True, 0, False, 0.05)

# Control, como es dmi = b*n34
series = {'dmi':dmi_l.values,'n34':n34_l.values}
b_n34_dmi = regre(series, True, 'n34', False, 0.05)

# MLR
series = {'u50': u50_l.values, 'n34':n34_l.values, 'dmi':dmi_l.values}
b_directo_n34 = regre(series, True, 'n34', False, 0.05)
b_directo_dmi = regre(series, True, 'dmi', False, 0.05)
print('')
aux = b_directo_n34 + b_n34_dmi*b_directo_dmi
print('b_directo_n34 + b_n34_dmi * b_directo_dmi:', aux)
print('b_total_n34:', b_total_n34)
print('')
series = {'n34':n34_l.values, 'dmi':dmi_l.values}
vif(series)
print('')
# coeficientes "a mano" ------------------------------------------------------ #
b1_rs, b1_mlr = beta_from_cov_MLR(u50_l.values, n34_l.values, dmi_l.values)
print('')
b2_rs, b2_mlr = beta_from_cov_MLR(u50_l.values, dmi_l.values, n34_l.values)
# ---------------------------------------------------------------------------- #