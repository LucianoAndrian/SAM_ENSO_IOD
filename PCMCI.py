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
def SetParents(parents, alpha, withtarget=False):
    parents = parents[parents['pval'] < alpha]
    #parents = parents.query('r < 0.99')
    if withtarget:
        parents = parents.assign(abs_r=parents['r'].abs()).sort_values(
            by=['Target', 'abs_r'], ascending=[True, False])
        parents = parents.drop(columns=['abs_r'])
    else:
        parents = parents.iloc[parents['r'].abs().argsort()[::-1]]
    parents['r'] = parents['r'].round(3)
    parents['pval'] = parents['pval'].round(3)
    return parents

def select_positions(serie, pos, window, lag=0):
    selected_values = []
    length = len(serie)
    step = 12
    pos -= lag

    for i in range(0, length//12 + 1):
        selected_values.extend(serie[pos-window-1+step*i:pos+window+step*i])

    return selected_values

def SetLags2(x, y, ty, series, parents, mm, w):
    x = select_positions(x, mm, w, 0)
    y = select_positions(y, mm, w, ty)

    z_series = []
    for i, p  in enumerate(parents):
        zn = series[p.split('_lag_')[0]]
        tz = np.int(p.split('_lag_')[1])

        z_series.append(select_positions(zn, mm, w, tz))

    z_columns = [f'parent{i}' for i in range(1, len(z_series) + 1)]
    z_data = {column: z_series[i] for i, column in enumerate(z_columns)}

    return pd.DataFrame({'x':x, 'y':y, **z_data})

def PartialCorrelation(df):
    x = df['x'].values
    y = df['y'].values

    X = np.column_stack((np.ones_like(x), df[df.columns[2:]].values))
    beta_x = np.linalg.lstsq(X, x, rcond=None)[0]
    x_res = x - np.dot(X, beta_x)

    beta_y = np.linalg.lstsq(X, y, rcond=None)[0]
    y_res = y - np.dot(X, beta_y)
    r, pv = pearsonr(x_res, y_res)

    return r, pv

def PC(series, target, tau_max, pc_alpha, mm, w):
    taus = np.arange(1, tau_max + 1)
    # Set preliminary parents ------------------------------------------------ #
    # Correlation
    first = True
    target_serie = select_positions(series[target], mm, w, 0)
    for k in series.keys():
            for t in taus:
                if t == 0 and k==target:
                    # NO si mismo.
                    pass
                else:
                    k_serie = select_positions(series[k], mm, w, t)

                    r, pv = pearsonr(target_serie, k_serie)

                    d = {'pparents': k + '_lag_' + str(t), 'r': [r],
                         'pval': [pv]}

                    if first:
                        first = False
                        parents0 = pd.DataFrame(d)
                    else:
                        parents0 = pd.concat([parents0, pd.DataFrame(d)],
                                             axis=0)
    parents = SetParents(parents0, pc_alpha)
    # ------------------------------------------------------------------------ #
    # Partial correlation
    i = 0
    while len(parents) > 2:
        strong_parents = parents['pparents'].head(i+2).tolist()
        first = True
        for p in parents['pparents']:

            # Select strong parent/s for partial correlation
            aux_strong_parents = strong_parents[:i+1] if \
                all(parent != p for parent in strong_parents) else \
                [parent for parent in strong_parents if parent != p]

            serie_p = p.split('_lag_')[0]
            t_p = np.int(p.split('_lag_')[1])

            df = SetLags2(series[target], series[serie_p], ty=t_p,
                          series=series, parents=aux_strong_parents,
                          mm=mm, w=w)

            r, pv = PartialCorrelation(df)

            d = {'pparents': serie_p + '_lag_' + str(t_p),
                 'r': [r], 'pval': [pv]}

            if first:
                first = False
                parents1 = pd.DataFrame(d)
            else:
                parents1 = pd.concat([parents1, pd.DataFrame(d)], axis=0)

        parents = SetParents(parents1, pc_alpha)
        i += 1
        if i > 5:
            break

    parents_name=[]
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

def MCI(series, targets, tau_max, parents, mci_alpha, mm, w):

    lags = np.arange(0, tau_max + 1)
    first = True
    for target in targets:
        target_parents_original = parents[target].copy()

        for l in lags:
            for a in targets:
                if l == 0 and a == target:
                    # NO si mismo.
                    pass
                else:
                    actor_parents = parents[a].copy()
                    actor_as_target_parent = a + '_lag_' + str(l)

                    target_parents = target_parents_original.copy()

                    # esto esta OK.
                    if actor_as_target_parent in target_parents_original:
                        target_parents.remove(actor_as_target_parent)

                    target_actor_parents = target_parents + \
                                           add_lag(actor_parents, l)

                    # Test
                    target_actor_parents = list(set(target_actor_parents))

                    # esto es debido a la longitud de las series en los casos
                    # donde el add_lag se va a la mierda falla pero no tienen
                    # mucho sentido por ahora esos lags
                    try:
                        df = SetLags2(series[target], series[a], ty=l,
                                      series=series,
                                      parents=target_actor_parents, mm=mm, w=w)
                        r, pv = PartialCorrelation(df)
                    except:
                        r, pv = 0, 1

                    d = {'Target': target, 'Actor': a + '_lag_' + str(l),
                         'r': [r], 'pval': [pv]}

                    if first:
                        first = False
                        parents_f = pd.DataFrame(d)
                    else:
                        parents_f = pd.concat([parents_f, pd.DataFrame(d)],
                                              axis=0)


    links = SetParents(parents_f, mci_alpha, True)
    #print(links)
    return links

def PCMCI(series, tau_max, pc_alpha, mci_alpha, mm, w):
    targets = []
    targets_parents= {}
    # PC --------------------------------------------------------------------- #
    for s in series.keys():
        targets.append(s)
        targets_parents.update({s:PC(series, s, tau_max, pc_alpha, mm, w)})
    # MCI -------------------------------------------------------------------- #
    links = MCI(series,targets, tau_max, targets_parents, mci_alpha, mm, w)
    return links

# series = {'x':x.values, ...}
# print(PCMCI(series=series, tau_max=3, pc_alpha=.1, mci_alpha=.1, mm=10, w=0))