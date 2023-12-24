"""
TEST Algoritmo de descubrimiento causal
en UN punto de grilla
"""

################################################################################
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import xarray as xr
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
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
c = hgt_anom.sel(lon=270, lat=-60)#.sel(time=hgt_anom.time.dt.year.isin(np.arange(1980,2020)))
#c = c.sel(time=c.time.dt.year.isin([np.arange(2010,2018)]))
#c = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
dmi2 = SameDateAs(dmi, c)
n342 = SameDateAs(n34, c)
sam2 = SameDateAs(sam, c)

c = c/c.std()
dmi3 = dmi2/dmi2.std()
n343 = n342/n342.std()
sam3 = sam2/sam2.std()
#------------------------------------------------------------------------------#
################################################################################
# Funciones ####################################################################
def Test_PartialCorrelation(x, y, ylag=0, parents=None, series=None,
                            retestmode=False):

    if len(x) != len(y):
        print('Test_PartialCorrelation Error: len(x) != len(y)')
        return
    if parents is None:
        print('Test_PartialCorrelation Error: parents is None')
        return
    if type(ylag)!=np.int:
        print('Test_PartialCorrelation Error: ylag must be int')
        return
    len_series = len(x)
    ty = ylag
    # ------------------------------------------------------------------------ #
    # first iter
    z = series[parents[0].split('_lag_')[0]]
    tz = np.int(parents[0].split('_lag_')[1])

    df = pd.DataFrame({'x':x[tz:], 'z':z[:-tz]})
    models = smf.ols(formula='x~z', data=df).fit()
    x_pred_by_z = models.params[1] * z[:-tz] + models.params[0]
    x_res = x[tz:] - x_pred_by_z
    del models

    if tz>ty:
        tyaux = tz - ty
        tzaux = 0
    elif ty>tz:
        tzaux = ty - tz
        tyaux=0
    else:
        tyaux=0
        tzaux=0

    df = pd.DataFrame({'y': y[tyaux:len_series-ty], 'z': z[tzaux:-tz]})
    models = smf.ols(formula='y~z', data=df).fit()
    y_pred_by_z = models.params[1] * z[tzaux:-tz] + models.params[0]
    y_res = y[tyaux:len_series-ty] - y_pred_by_z
    del models

    if retestmode==False:
        return pearsonr(x_res[tzaux:], y_res)

    for i in range(1 , len(parents)):
        # N step PC
        if i==1:
            #print('i=1')
            t_prima = max([ty, tz])
            x = x_res[tzaux:]
            y = y_res

            z2 = series[parents[i].split('_lag_')[0]]
            tz2 = np.int(parents[i].split('_lag_')[1])
            z = z2[t_prima:]

            df = pd.DataFrame({'x': x[tz2:], 'z': z[:-tz2]})
            models = smf.ols(formula='x~z', data=df).fit()
            x_pred_by_z = models.params[1] * z[:-tz2] + models.params[0]
            x_res = x[tz2:] - x_pred_by_z
            del models

            df = pd.DataFrame({'y': y[tz2:], 'z': z[:-tz2]})
            models = smf.ols(formula='y~z', data=df).fit()
            y_pred_by_z = models.params[1] * z[:-tz2] + models.params[0]
            y_res = y[tz2:] - y_pred_by_z
            del models
            #print(pearsonr(x_res, y_res))

        elif retestmode==True:
            #print('i=n')
            x = x_res
            y = y_res

            if len(x)==0:
                print('xerror')
            if len(y)==0:
                print('yerror')
            try:
                zn = series[parents[i].split('_lag_')[0]]
                tzn = np.int(parents[i].split('_lag_')[1])
                z = zn[:-tzn]
            except:
                pass
            if len(z)==0:
                return print('zerror')

            taux=1
            while(len(x)!=len(z)):
                if len(z)>len(x):
                    z = z[taux:]
                else:
                    x = x[taux:]

                if len(x)==0:
                    print('xerror2')
                if len(z) == 0:
                    print('zerror2')
                    return z

            while (len(y) != len(z)):
                if len(z)>len(y):
                    z = z[taux:]
                else:
                    y = y[taux:]

                if len(z) == 0:
                    print('zerror2a')
                    return z

            df = pd.DataFrame({'x': x, 'z': z})
            models = smf.ols(formula='x~z', data=df).fit()
            x_pred_by_z = models.params[1] * z + models.params[0]
            x_res = x - x_pred_by_z
            del models

            df = pd.DataFrame({'y': y, 'z': z})
            models = smf.ols(formula='y~z', data=df).fit()
            y_pred_by_z = models.params[1] * z + models.params[0]
            y_res = y - y_pred_by_z
            del models

    try:
        return pearsonr(x_res, y_res)
    except:
        print('error lenght')

def SetParents(parents, pc_alpha, withtarget=False):
    parents = parents[parents['pval'] < pc_alpha]
    parents = parents.query('r < 0.99')
    if withtarget:
        parents = parents.assign(abs_r=parents['r'].abs()).sort_values(
            by=['Target', 'abs_r'], ascending=[True, False])
        parents = parents.drop(columns=['abs_r'])
    else:
        parents = parents.iloc[parents['r'].abs().argsort()[::-1]]
    parents['r'] = parents['r'].round(3)
    parents['pval'] = parents['pval'].round(3)
    return parents

def PC(series, target, tau_max, pc_alpha):
    taus = np.arange(1, tau_max + 1)
    len_series = len(series[target])

    # Set preliminary parents ------------------------------------------------ #
    # Correlation
    first = True
    for k in series.keys():
        for t in taus:
            r, pv = pearsonr(series[target][t:], series[k][:len_series - t])
            d = {'pparents': k + '_lag_' + str(t), 'r': [r], 'pval': [pv]}

            if first:
                first = False
                parents0 = pd.DataFrame(d)
            else:
                parents0 = pd.concat([parents0, pd.DataFrame(d)], axis=0)

    parents = SetParents(parents0, pc_alpha)
    # ------------------------------------------------------------------------ #
    # Partial correlation
    if len(parents) > 2:
        # Strong parents for partial correlation
        strong_parents = parents['pparents'].head(2).tolist()

        first = True
        for p in parents['pparents']:
            if p == strong_parents[0]:
                sp = strong_parents[1]
            else:
                sp = strong_parents[0]

            serie_p = p.split('_lag_')[0]
            t_p = np.int(p.split('_lag_')[1])

            r,pv = Test_PartialCorrelation(x=series[target],
                                    y=series[serie_p],
                                    ylag=t_p,
                                    series=series,
                                    parents=[sp],
                                    retestmode=False)

            d = {'pparents': serie_p + '_lag_' + str(t_p),
                 'r': [r], 'pval': [pv]}
            if first:
                first = False
                parents1 = pd.DataFrame(d)
            else:
                parents1 = pd.concat([parents1, pd.DataFrame(d)], axis=0)

        parents = SetParents(parents1, pc_alpha)
        print('parents1')
        print(parents)
        # -------------------------------------------------------------------- #
        # Partial correlation
        # 2nd step PC
        if len(parents) > 2:
            # Strong parents for partial correlation
            strong_parents = parents['pparents'].head(3).tolist()

            first = True
            for p in parents['pparents']:
                # Select 2 strong parents for partial correlation
                aux_strong_parents = strong_parents[:2] if \
                    all(parent != p for parent in strong_parents) else \
                    [parent for parent in strong_parents if parent != p]

                serie_p = p.split('_lag_')[0]
                t_p = np.int(p.split('_lag_')[1])

                r, pv = Test_PartialCorrelation(x=series[target],
                                                 y=series[serie_p],
                                                 ylag=t_p,
                                                 series=series,
                                                 parents=aux_strong_parents,
                                                 retestmode=True)

                d = {'pparents': serie_p + '_lag_' + str(t_p), 'r': [r],
                     'pval': [pv]}
                if first:
                    first = False
                    parents2 = pd.DataFrame(d)
                else:
                    parents2 = pd.concat([parents2, pd.DataFrame(d)], axis=0)

            parents = SetParents(parents2, pc_alpha)

    parents_name = []
    for p in parents['pparents']:
        parents_name.append(p)

    return parents_name

def add_lag(parents, plus_lag=1):
    parents_add_lag = []
    for p in parents:
        pre, lag = p.split('_lag_')
        lag = int(lag) + plus_lag
        parents_add_lag.append(pre + '_lag_' + str(lag))

    return parents_add_lag
################################################################################
# argumentos de una posible funcion
tau_max = 2
series = {'c':c['var'].values, 'dmi':dmi3.values, 'n34':n343.values}
pc_alpha = 0.05

################################################################################
# mci step
c_parents = PC(series, 'c', tau_max, pc_alpha)
dmi_parents = PC(series, 'dmi', tau_max, pc_alpha)
n34_parents = PC(series, 'n34', tau_max, pc_alpha)
# ---------------------------------------------------------------------------- #

################################################################################
# probar
targets = ['n34', 'dmi', 'c']
lags = [0, 1, 2]
parents = {'c':c_parents, 'dmi':dmi_parents, 'n34':n34_parents}

first = True
for target in targets:
    print('target:' + target)
    for l in lags:
        print('Lag:' + str(l))
        target_parents = parents[target].copy()

        for a in targets:
            print(a)
            actor_parents = parents[a].copy()

            actor_as_parent = a + '_lag_' + str(l)

            if actor_as_parent in target_parents:
                target_parents.remove(actor_as_parent)

            target_actor_parents = target_parents + add_lag(actor_parents, l)
            # unique_target_actor_parents = list(set(target_actor_parents))
            unique_set = set()
            unique_target_actor_parents = []

            for item in target_actor_parents:
                if item not in unique_set:
                    unique_set.add(item)
                    unique_target_actor_parents.append(item)
            if l == 0:
                print(unique_target_actor_parents)
           # target_actor_parents = list(set(target_actor_parents))
            r, pv = Test_PartialCorrelation(x=series[target],
                                    y=series[a],
                                    ylag=l, series=series,
                                    parents=unique_target_actor_parents,
                                    retestmode=True)

            d = {'Target': target, 'Actor': a + '_lag_' + str(l),
                 'r': [r], 'pval': [pv]}

            if first:
                first = False
                parents_f = pd.DataFrame(d)
            else:
                parents_f = pd.concat([parents_f, pd.DataFrame(d)], axis=0)

print(SetParents(parents_f, 0.5, True))

c_parents
dmi_parents
rpv, dmi_res = Test_PartialCorrelation(x=series['n34'],
                                y=series['dmi'],
                                ylag=1, series=series,
                                parents=n34_parents[:2]+add_lag(dmi_parents,1),
                                retestmode=True)

rpv, n34_res = Test_PartialCorrelation(x=series['dmi'],
                                y=series['n34'],
                                ylag=0, series=series,
                                parents=dmi_parents+n34_parents[:2],
                                retestmode=True)

x = series['n34']
y = series['dmi']
ylag = 1
series = series
parents = n34_parents + add_lag(dmi_parents, 1)
retestmode = True

parents_f.sort_values(by=['Target', 'r'], ascending=[True, False])

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
import itertools
import matplotlib.pyplot as plt

def partial_correlation(x, y, parents):
    # Inicializar los residuos de x e y
    x_resid = x.copy()
    y_resid = y.copy()

    # Calcular la correlación parcial entre x e y quitando el efecto de los parents
    partial_corr, _ = pearsonr(x_resid, y_resid)

    return partial_corr

def average_partial_correlation(x, y, parents):
    # Obtener todas las permutaciones posibles de los parents
    parent_permutations = list(itertools.permutations(parents))

    # Calcular la correlación parcial para cada permutación y tomar el promedio
    avg_partial_corr = np.max([
        partial_correlation(x, y, list(permutation))
        for permutation in parent_permutations])

    return avg_partial_corr

# Generar datos de ejemplo
np.random.seed(42)
n = 100
time = np.arange(971)


# Generar parents
num_parents = 3
parents = [series['dmi'][:-1], series['n34'][:-1]]

# Calcular la correlación parcial promedio
avg_partial_corr = average_partial_correlation(x, y, parents)

# Visualizar las series temporales
plt.figure(figsize=(12, 6))
plt.plot(time, x, label='X')
plt.plot(time, y, label='Y')
plt.title('Series Temporales X e Y')

# Mostrar la correlación parcial promedio
print(f"Correlación parcial promedio: {avg_partial_corr}")






import numpy as np
import pandas as pd
import statsmodels.api as sm
x = series['n34'][2:]  # Serie temporal X con tendencia lineal
y = series['dmi'][2:]
# Generar parents
num_parents = 3
parents = [series['dmi'][1:-1], series['n34'][:-2],
           series['dmi'][:-2], series['n34'][1:-1]]


# Crear un DataFrame con los datos
data = pd.DataFrame({'x': x, 'y': y,
                     'parent1': parents[0],
                     'parent2': parents[1],
                     'parent3': parents[2],
                     'parent4': parents[3]})

# Realizar regresión múltiple para quitar efecto de los parents en x
model_x = sm.OLS(data['x'], sm.add_constant(data[['parent1', 'parent2',
                                                  'parent3', 'parent4']])).fit()
x_resid = model_x.resid

# Realizar regresión múltiple para quitar efecto de los parents en y
model_y = sm.OLS(data['y'], sm.add_constant(data[['parent1', 'parent2',
                                                  'parent3', 'parent4']])).fit()
y_resid = model_y.resid

# Calcular la correlación entre los residuos
correlation_residuals, _ = np.corrcoef(x_resid, y_resid)

print(f"Correlación entre los residuos: {correlation_residuals[1]}")
