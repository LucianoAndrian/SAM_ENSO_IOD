"""
TEST Algoritmo de descubrimiento causal
en UN punto de grilla
"""

################################################################################
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
import pandas as pd
pd.options.mode.chained_assignment = None
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import xarray as xr
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
################################################################################
################################################################################
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/mlr/'
era5_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
t_pp_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_obs_d_w_c/'
# ---------------------------------------------------------------------------- #
path = '/pikachu/datos/luciano.andrian/cases_fields/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/eof/'
################################################################################
################################################################################
ruta = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
hgt = xr.open_dataset(ruta + 'ERA5_HGT200_40-20.nc')
hgt = hgt.rename({'longitude': 'lon', 'latitude': 'lat', 'z': 'var'})
hgt = hgt.sel(lat=slice(-20, -90))
hgt = hgt.interp(lon=np.arange(0,360,.5), lat=np.arange(-90, 90, .5))

hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
hgt_anom = hgt.groupby('time.month') - \
           hgt_clim.groupby('time.month').mean('time')

weights = np.sqrt(np.abs(np.cos(np.radians(hgt_anom.lat))))
hgt_anom = hgt_anom * weights

hgt_anom = hgt_anom.sel(lat=slice(None, -20))
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam = sam.rolling(time=3, center=True).mean()

dmi = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

aux = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")
n34 = Nino34CPC(aux, start=1920, end=2020)[0]
################################################################################
c = hgt_anom.sel(lon=270, lat=-30)
#c = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
dmi2 = SameDateAs(dmi, c)
n342 = SameDateAs(n34, c)

c = c/c.std()
dmi = dmi2/dmi2.std()
n34 = n342/n342.std()
#------------------------------------------------------------------------------#
################################################################################

# array con series temporales, lista, diccionario
# tau max
# variable target
# alpha pc
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
# Funcion que calcule la correlación entre dos series temporales sin el efecto
# de una 3era. Agregando, si es que hay una 3era sino. La correlación entre
# esas variables.

# Funciones ####################################################################
def Test_PartialCorrelation(x, y, z=None, ty=0, tz=0, res=False):

    if len(x) != len(y):
        print('Test_PartialCorrelation: len(x) != len(y)')
        return
    else:
        if z is not None:
            if len(x) != len(z):
                print('Test_PartialCorrelation: len(x) != len(z)')
                return

    if z is None:
        # Only Correlation
        return pearsonr(x, y)

    elif z is not None:
        l = len(x)
        # Partial Correlation
        df = pd.DataFrame({'x':x[tz:], 'z':z[:l-tz]})
        models = smf.ols(formula='x~z', data=df).fit()
        x_pred_by_z = models.params[1] * z[:l-tz] + models.params[0]

        x_res = x[tz:] - x_pred_by_z

        if tz>ty:
            tyaux = tz - ty
            tzaux = 0
        elif ty>tz:
            tzaux = ty - tz
            tyaux=0
        else:
            tyaux=0
            tzaux=0

        df = pd.DataFrame({'y': y[tyaux:l-ty], 'z': z[tzaux:l-tz]})

        models = smf.ols(formula='y~z', data=df).fit()
        y_pred_by_z = models.params[1] * z[tzaux:l-tz] + models.params[0]

        y_res = y[tyaux:l-ty] - y_pred_by_z

        if res:
            return pearsonr(x_res[tzaux:], y_res[:l-tyaux]), \
                   x_res[tzaux:], y_res[:l-tyaux]
        else:
            return pearsonr(x_res[tzaux:], y_res[:l-tyaux])

    else:
        print('Error en TestPartialCorrelation')

def SetParents(parents, pc_alpha):
    parents = parents[parents['pval'] < pc_alpha]
    parents = parents.query('r < 0.9999')
    parents = parents.iloc[parents['r'].abs().argsort()[::-1]]
    return parents

################################################################################
# argumentos de una posible funcion
tau_max = 2
series = {'c':c['var'].values, 'dmi':dmi.values, 'n34':n34.values}
target = 'dmi' #incluida en "series"
pc_alpha = 0.05

# Set preliminary parents
taus = np.arange(1, tau_max+1)
len_series = len(series[target])

first = True
for k in series.keys():
    for t in taus:

        r, pv = Test_PartialCorrelation(series[target][t:],
                                        series[k][:len_series-t])

        d = {'pparents':k +'_lag_' + str(t), 'r': [r], 'pval':[pv]}
        if first:
            first = False
            parents0 = pd.DataFrame(d)
        else:
            parents0 = pd.concat([parents0, pd.DataFrame(d)], axis=0)

parents = SetParents(parents0, pc_alpha)

#------------------------------------------------------------------------------#
if len(parents)>2:

    strong_parents = parents['pparents'].head(2).tolist()

    first = True
    for p in parents['pparents']:

        if p == strong_parents[0]:
            sp = strong_parents[1]
        else:
            sp = strong_parents[0]

        serie_p = p.split('_lag_')[0]
        t_p = np.int(p.split('_lag_')[1])

        serie_sp = sp.split('_lag_')[0]
        t_sp = np.int(sp.split('_lag_')[1])

        r, pv = Test_PartialCorrelation(series[target], series[serie_p],
                                        z=series[serie_sp], ty=t_p, tz=t_sp)

        d = {'pparents':serie_p +'_lag_' + str(t_p), 'r': [r], 'pval':[pv]}
        if first:
            first = False
            parents1 = pd.DataFrame(d)
        else:
            parents1 = pd.concat([parents1, pd.DataFrame(d)], axis=0)

    parents = SetParents(parents1, pc_alpha)


    # REVISAR a partir de aca, lo anterior esta bien.
    if len(parents) > 3:

        strong_parents = parents['pparents'].head(3).tolist()

        first = True
        for p in parents['pparents']:

            aux_strong_parents = strong_parents[:2] if \
                all(parent != p for parent in strong_parents) else \
                [parent for parent in strong_parents if parent != p]

            sp1 = aux_strong_parents[0]
            sp2 = aux_strong_parents[1]

            serie_p = p.split('_lag_')[0]
            t_p = np.int(p.split('_lag_')[1])

            serie_sp1 = sp1.split('_lag_')[0]
            t_sp1 = np.int(sp1.split('_lag_')[1])

            serie_sp2 = sp2.split('_lag_')[0]
            t_sp2 = np.int(sp2.split('_lag_')[1])


            rpv, x_res, y_res = \
                Test_PartialCorrelation(series[target], series[serie_p],
                                        z=series[serie_sp1], ty=t_p, tz=t_sp1,
                                        res=True)

            t_aux = max([t_p, t_sp1])
            r, pv = Test_PartialCorrelation(x_res, y_res,
                                            z=series[serie_sp2][t_aux:],
                                            ty=t_p, tz=t_sp2)

            d = {'pparents': serie_p + '_lag_' + str(t_p), 'r': [r],
                 'pval': [pv]}
            if first:
                first = False
                parents2 = pd.DataFrame(d)
            else:
                parents2 = pd.concat([parents2, pd.DataFrame(d)], axis=0)

        parents = SetParents(parents2, pc_alpha)


print(parents)

