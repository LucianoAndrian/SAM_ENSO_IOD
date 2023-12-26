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

def recursive_cut(s1, s2):
    i=1
    while(len(s1)!=len(s2)):
        i += 1
        if len(s1)>len(s2):
            s1 = s1[1:]
        else:
            s2 = s2[1:]

        if i > 100:
            print('Error recursive_cut +100 iter.')
            return

    return s1, s2

def min_len(st):
    min_len = float('inf')
    min_st = None

    for s in st:
        cu_len = len(s)
        if cu_len < min_len:
            min_len = cu_len
            min_st = s

    return min_st

def SetLags(x, y, ty, series, parents):

    z_series = []
    for i, p  in enumerate(parents):

        zn = series[p.split('_lag_')[0]]
        tz = np.int(p.split('_lag_')[1])
        len_series = len(zn)

        if i==0:
            ty_aux = max(0, tz - ty)
            tz_aux = max(0, ty - tz)

            x = x[tz+tz_aux:]
            y = y[ty_aux:len_series-ty]
            z_series.append(zn[tz_aux:len_series-tz])
            tz_prev = tz

        if i>0:
            if i>1:
                t_aux = 0
            else:
                t_aux = max(ty, tz_prev)

            z_aux = zn[t_aux:len_series - tz]
            z_series.append(z_aux)

            aux = min_len([x, y, z_aux])

            x,_ = recursive_cut(x, aux)
            y,_ = recursive_cut(y, aux)

            for iz, zs in enumerate(z_series):
                z_series[iz] ,_ = recursive_cut(zs, aux)

    z_columns = [f'parent{i}' for i in range(1, len(z_series) + 1)]
    z_data = {column: z_series[i] for i, column in enumerate(z_columns)}

    return pd.DataFrame({'x':x, 'y':y, **z_data})

def PartialCorrelation(df):
    import statsmodels.api as sm

    x_model = sm.OLS(df['x'], sm.add_constant(df[df.columns[2:]])).fit()
    x_res = x_model.resid

    y_model = sm.OLS(df['y'], sm.add_constant(df[df.columns[2:]])).fit()
    y_res = y_model.resid

    return pearsonr(x_res, y_res)

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

            df = SetLags(series[target], series[serie_p], ty=t_p, series=series,
                         parents=[sp])

            r, pv = PartialCorrelation(df)

            d = {'pparents': serie_p + '_lag_' + str(t_p),
                 'r': [r], 'pval': [pv]}
            if first:
                first = False
                parents1 = pd.DataFrame(d)
            else:
                parents1 = pd.concat([parents1, pd.DataFrame(d)], axis=0)

        parents = SetParents(parents1, pc_alpha)
        # print('parents1')
        # print(parents)
        # -------------------------------------------------------------------- #
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

                df = SetLags(series[target], series[serie_p], ty=t_p,
                             series=series,
                             parents=aux_strong_parents)

                r, pv = PartialCorrelation(df)

                d = {'pparents': serie_p + '_lag_' + str(t_p), 'r': [r],
                     'pval': [pv]}
                if first:
                    first = False
                    parents2 = pd.DataFrame(d)
                else:
                    parents2 = pd.concat([parents2, pd.DataFrame(d)],
                                         axis=0)

            parents = SetParents(parents2, pc_alpha)

    #print(parents)
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

def MCI(series, targets, tau_max, parents, mci_alpha):

    lags = np.arange(0, tau_max + 1)
    first = True
    for target in targets:
        target_parents_original = parents[target].copy()

        for l in lags:
            for a in targets:
                actor_parents = parents[a].copy()
                actor_as_target_parent = a + '_lag_' + str(l)

                target_parents = target_parents_original.copy()
                if actor_as_target_parent in target_parents_original:
                    target_parents.remove(actor_as_target_parent)

                target_actor_parents = target_parents + \
                                       add_lag(actor_parents, l)
                # Test
                target_actor_parents = list(set(target_actor_parents))

                df = SetLags(series[target], series[a], ty=l, series=series,
                             parents=target_actor_parents)

                r, pv = PartialCorrelation(df)

                d = {'Target': target, 'Actor': a + '_lag_' + str(l),
                     'r': [r], 'pval': [pv]}

                if first:
                    first = False
                    parents_f = pd.DataFrame(d)
                else:
                    parents_f = pd.concat([parents_f, pd.DataFrame(d)], axis=0)

    links = SetParents(parents_f, mci_alpha, True)
    #print(links)
    return links

def PCMCI(series, tau_max, pc_alpha, mci_alpha):
    targets = []
    targets_parents= {}
    # PC --------------------------------------------------------------------- #
    for s in series.keys():
        targets.append(s)
        targets_parents.update({s:PC(series, s, tau_max, pc_alpha)})
    # MCI -------------------------------------------------------------------- #
    links = MCI(series,targets, tau_max, targets_parents, mci_alpha)
    return links