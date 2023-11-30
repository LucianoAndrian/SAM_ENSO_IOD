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

sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam = sam.rolling(time=3, center=True).mean()

dmi = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

aux = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")
n34 = Nino34CPC(aux, start=1920, end=2020)[0]
################################################################################
# Funciones ####################################################################
from scipy.stats import pearsonr
def ComputeAndAdd(x, y, name, *df0):
    r, pv = pearsonr(x,y)
    d = {'name': name, 'r': [r], 'pv': [pv]}

    try:
        df = df0[0]
        dfs = pd.concat([df, pd.DataFrame(d)], axis=0)
    except:
        dfs = pd.DataFrame(d)

    return dfs


def RegRes(df, n, y):
    import statsmodels.formula.api as smf
    formula = y+'~'+n
    result = smf.ols(formula=formula, data=df).fit()
    y_pred_x = result.params[1] * df[n] + result.params[0]

    return df[y] - y_pred_x

def IdentifyTau(ns):
    try:
        t = -1*int(ns[1])
    except:
        t = None

    return t

def SetTaus(df, n, n_sp, t, t_sp):
    c = df['c'].tolist()
    i_n = df[n][:t].tolist()
    i_n_sp = df[n_sp][:t_sp].tolist()

    t_aux0 = np.abs(t_sp-0)
    if(len(c)>len(i_n_sp)):
        c = c[t_aux0:]

    c_df = pd.DataFrame({'c':c, n_sp:i_n_sp})

    if t is None:
        t_aux1 = np.abs(t_sp)
    elif t_sp is None:
        t_aux1 = np.abs(t)
    elif (t_sp is None) and (t is None):
        t_aux1 = 0
    else:
        t_aux1 = np.abs(t_sp-t)

    if (len(i_n)>len(i_n_sp)):
        i_n = i_n[t_aux1:]
    elif (len(i_n)<len(i_n_sp)):
        i_n_sp = i_n_sp[t_aux1:]

    if n == n_sp:
        n_sp_aux = n_sp + '_sp'
    else:
        n_sp_aux = n_sp

    i_n_df = pd.DataFrame({n:i_n, n_sp_aux:i_n_sp})

    t_aux = 0 if (len(c_df)==len(i_n_df)) else t_aux1

    return c_df, i_n_df, t_aux

################################################################################
c = hgt_anom.sel(lon=270, lat=-60)
dmi = SameDateAs(dmi, c)
n34 = SameDateAs(n34, c)
#------------------------------------------------------------------------------#

df = pd.DataFrame({'indice1' : n34.values,
                   'indice2': dmi.values,
                   'c': c['var'].values})

taus = [None, -1, -2]
taus0 = [None, 1, 2]
first = True
for i in ['indice1', 'indice2', 'c']:
    for t, t0 in zip(taus, taus0):
        if (i!='c' or t!=None):
            if first:
                first = False
                corrs_1 = ComputeAndAdd(df['c'], df[i], i)
            else:
                t_aux = '' if t == None else t
                corrs_1 = ComputeAndAdd(df['c'][t0:],
                                        df[i][:t], i + str(t_aux),
                                        corrs_1)


parents0 = corrs_1[corrs_1['pv'] < 0.05]

if len(parents0)>2:
    parents0 = parents0.iloc[parents0['r'].abs().argsort()[::-1]]

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
strong_parents = parents0['name'].head(2).tolist()
# partial correlation
first=True
globals().pop('corrs_2', None) if 'corrs_2' in globals() else None
for n in parents0['name']:

    n_sp = strong_parents[1] if n == strong_parents[0] else strong_parents[0]

    ns = n.split('-')
    ns_sp = n_sp.split('-')

    t = IdentifyTau(ns)
    t_sp = IdentifyTau(ns_sp)

    n = ns[0]
    n_sp = ns_sp[0]

    c_df, i_n_df, t_aux = SetTaus(df, n, n_sp, t, t_sp)

    c_wo_n_sp = RegRes(c_df, n_sp, 'c')
    i_wo_n_sp = RegRes(i_n_df, i_n_df.columns[1], n)


    if first:
        first = False
        corrs_2 = ComputeAndAdd(c_wo_n_sp[t_aux:], i_wo_n_sp,
                                n+str(t)+'_wo_'+n_sp+str(t_sp))
    else:
        t_sp = '' if t == None else t
        t = '' if t == None else t
        corrs_2 =ComputeAndAdd(c_wo_n_sp[t_aux:], i_wo_n_sp,
                                n+str(t)+'_wo_'+n_sp+str(t_sp), corrs_2)

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#

parents1 = corrs_2[corrs_2['pv'] < 0.05]
if len(parents1)>3:
    parents1 = parents1.iloc[parents1['r'].abs().argsort()[::-1]]
strong_parents = parents1['name'].head(3).tolist()

# REVEER #######################################################################
def SetTaus_Nphase(r, df, n, n_sp2, t_sp1, t_sp2):
    i_n_sp2 = df[n_sp2][:t_sp2].tolist()

    if t_sp1 is None:
        t_aux1 = np.abs(t_sp2)
    elif t_sp2 is None:
        t_aux1 = np.abs(t_sp1)
    elif (t_sp2 is None) and (t_sp1 is None):
        t_aux1 = 0
    else:
        print('test')
        t_aux1 = np.abs(t_sp2-t_sp1)

    if (len(r)>len(i_n_sp2)):
        r = r[t_aux1:]
    elif (len(r)<len(i_n_sp2)):
        i_n_sp = i_n_sp2[t_aux1:]

    if n == n_sp:
        n_sp_aux = n_sp + '_spN'
    else:
        n_sp_aux = n_sp

    print(t_aux1)
    print(len(r))
    print(len(i_n_sp))

    i_n_df = pd.DataFrame({n:r, n_sp_aux:i_n_sp})

    t_aux = 0 if (len(c_df)==len(i_n_df)) else t_aux1

    return i_n_df, t_aux
first = True
globals().pop('corrs_3', None) if 'corrs_3' in globals() else None
for n in parents1['name']:

    aux_strong_parents = strong_parents[:2] if \
        all(parent != n for parent in strong_parents) else \
        [parent for parent in strong_parents if parent != n]

    n_sp = aux_strong_parents[0]

    ns = n.split('_wo_')[0].split('-')
    t = IdentifyTau(ns)

    ns_sp = n_sp.split('_wo_')[0].split('-')
    t_sp1 = IdentifyTau(ns_sp)

    n = ns[0]
    n_sp = ns_sp[0]

    c_df, i_n_df, t_aux = SetTaus(df, n, n_sp, t, t_sp1)

    c_wo_n_sp = RegRes(c_df, n_sp, 'c')
    i_wo_n_sp = RegRes(i_n_df, i_n_df.columns[1], n)

    n_sp2 = aux_strong_parents[1]
    ns_sp2 = n_sp2.split('_wo_')[0].split('-')
    t_sp2 = IdentifyTau(ns_sp2)
    ns_sp2  = ns_sp2[0]

    i_n_df, t_aux = SetTaus_Nphase(c_wo_n_sp.tolist(),
                                    df, n, ns_sp2, t_sp1, t_sp2)
    c_wo_n_sp2 = RegRes(i_n_df, i_n_df.columns[1], i_n_df.columns[0])

    i_n_df, t_aux = SetTaus_Nphase(i_wo_n_sp.tolist(),
                                    df, n, ns_sp2, t_sp1, t_sp2)
    i_wo_n_sp2 = RegRes(i_n_df, i_n_df.columns[1], i_n_df.columns[0])


    if first:
        first = False
        corrs_3 = ComputeAndAdd(c_wo_n_sp2[t_aux:], i_wo_n_sp2,
                                n+str(t)+'_wo_2p')
    else:
        corrs_3 =ComputeAndAdd(c_wo_n_sp2[t_aux:], i_wo_n_sp2,
                                n+str(t)+'_wo_2p', corrs_2)

#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#