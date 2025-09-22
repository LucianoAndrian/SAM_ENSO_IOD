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

def RegRes(df, y, n):
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

def NoneToZero(t):
    if t is None:
        return 0
    else:
        return t

def SetLags_and_Reg(c, i, sp1, sp2, t, t_sp1, t_sp2, control, mode_single):
    t = NoneToZero(t)
    t_sp1 = NoneToZero(t_sp1)
    t_sp2 = NoneToZero(t_sp2)

    # c ---------------------------------------------------------------------- #
    c = c[np.abs(t_sp1):]
    rc = pd.DataFrame({'c': c, 'sp1': sp1})

    # 1 ---------------------------------------------------------------------- #
    rc = RegRes(rc, 'sp1', 'c')

    # 2 ---------------------------------------------------------------------- #
    if mode_single is False:
        c1 = False
        c2 = False
        if np.abs(t_sp1) > np.abs(t_sp2):
            t_prima = np.abs(t_sp1) - np.abs(t_sp2)
            sp2aux = sp2[t_prima:]
            rrc = pd.DataFrame({'rc': rc, 'sp2': sp2aux})
            c1 = True

        elif np.abs(t_sp1) < np.abs(t_sp2):
            t_prima = np.abs(t_sp2) - np.abs(t_sp1)
            rc = rc[t_prima:]
            rrc = pd.DataFrame({'rc': rc, 'sp2': sp2})
            c2 = True

        elif np.abs(t_sp1) == np.abs(t_sp2):

            rrc = pd.DataFrame({'rc': rc, 'sp2': sp2})
            c1 = True

        rrc = RegRes(rrc, 'sp2', 'rc')

    # ------------------------------------------------------------------------ #
    # i
    # 1 ---------------------------------------------------------------------- #
    a = False
    b = False

    if np.abs(t) > np.abs(t_sp1):
        t_prima = np.abs(t) - np.abs(t_sp1)
        sp1 = sp1[t_prima:]
        ri = pd.DataFrame({'i': i, 'sp1': sp1})
        a = True

    elif np.abs(t) < np.abs(t_sp1):
        t_prima = np.abs(t_sp1) - np.abs(t)
        i = i[t_prima:]
        ri = pd.DataFrame({'i': i, 'sp1': sp1})
        b = True

    elif np.abs(t) == np.abs(t_sp1):
        ri = pd.DataFrame({'i': i, 'sp1': sp1})
        a = True

    ri = RegRes(ri, 'sp1', 'i')

    if mode_single:
        if a:
            t_control = np.abs(t) - np.abs(t_sp1)
            if t_control !=0:
                rc = rc[t_control:]
        return rc, ri
    # 2 ---------------------------------------------------------------------- #
    i = False
    ii = False
    iii = False
    if a:
        if np.abs(t_sp2) > np.abs(t):
            t_prima = np.abs(t_sp2) - np.abs(t)
            ri = ri[t_prima:]
            rri = pd.DataFrame({'ri': ri, 'sp2': sp2})
            i = True

        elif np.abs(t_sp2) < np.abs(t):
            t_prima = np.abs(t) - np.abs(t_sp2)
            sp2aux2 = sp2[t_prima:]
            rri = pd.DataFrame({'ri': ri, 'sp2': sp2aux2})
            ii = True

        elif np.abs(t_sp2) == np.abs(t):

            rri = pd.DataFrame({'ri': ri, 'sp2': sp2})
            i = True

    elif b:
        if np.abs(t_sp2) > np.abs(t_sp1):
            i = True
            t_prima = np.abs(t_sp2) - np.abs(t_sp1)
            ri = ri[t_prima:]
            rri = pd.DataFrame({'ri': ri, 'sp2': sp2})

        elif np.abs(t_sp2) < np.abs(t_sp1):
            iii = True
            t_prima = np.abs(t_sp1) - np.abs(t_sp2)
            sp2aux3 = sp2[t_prima:]
            rri = pd.DataFrame({'ri': ri, 'sp2': sp2aux3})

        elif np.abs(t_sp2) == np.abs(t_sp1):
            i = True
            rri = pd.DataFrame({'ri': ri, 'sp2': sp2})


    rri = RegRes(rri, 'sp2', 'ri')

    if c1 and a and ii:
        t_control = np.abs(t) - np.abs(t_sp1)
        if t_control !=0:
            rrc = rrc[t_control:]
    elif c2 and a and ii:
        t_control = np.abs(t) - np.abs(t_sp2)
        if t_control !=0:
            rrc = rrc[t_control:]

    if control:
        print('Done SetLag_and_Reg')
    return rrc, rri

################################################################################
c = hgt_anom.sel(lon=270, lat=-30)
#c = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
dmi2 = SameDateAs(dmi, c)
n342 = SameDateAs(n34, c)

c = c/c.std()
dmi = dmi2/dmi2.std()
n34 = n342/n342.std()
#------------------------------------------------------------------------------#
def Phase1(indice1, indice2, indice3, mode=1):
    if mode==1:
        indices = [indice1, indice2, indice3]
    elif mode==2:
        indices = [indice2, indice3, indice1]
    elif mode==3:
        indices = [indice3, indice1, indice2]

    df = pd.DataFrame({'indice1':indices[0],
                       'indice2':indices[1],
                       'indice3':indices[2]})
    control = False
    # globals().pop('parents1', None) if 'parents1' in globals() else None
    # globals().pop('corrs_2', None) if 'corrs_2' in globals() else None
    # globals().pop('parents2', None) if 'parents2' in globals() else None
    # globals().pop('corrs_3', None) if 'corrs_3' in globals() else None
    print('test0')
    taus = [None, -1, -2]
    taus0 = [None, 1, 2]
    first = True
    for i in ['indice1', 'indice2', 'indice3']:
        for t, t0 in zip(taus, taus0):
            if (i != 'indice3' or t != None):
                if first:
                    first = False
                    corrs_1 = ComputeAndAdd(df['indice3'], df[i], i)
                else:
                    t_aux = '' if t == None else t
                    corrs_1 = ComputeAndAdd(df['indice3'][t0:],
                                            df[i][:t], i + str(t_aux),
                                            corrs_1)

    parents0 = corrs_1[corrs_1['pv'] < 0.05]
    parents = parents0
    print('test1')
    # ------------------------------------------------------------------------ #
    if len(parents0) > 2:

        parents0 = parents0.iloc[parents0['r'].abs().argsort()[::-1]]

        strong_parents0 = parents0['name'].head(2).tolist()
        first = True

        for n in parents0['name']:

            n_sp = strong_parents0[1] if n == strong_parents0[0] else \
            strong_parents0[0]

            ns = n.split('-')
            t = IdentifyTau(ns)

            ns_sp = n_sp.split('-')
            t_sp = IdentifyTau(ns_sp)

            n = ns[0]
            n1 = ns_sp[0]

            c = df['indice3'].tolist()
            i = df[n].tolist()[:t]
            sp = df[n1].tolist()[:t_sp]

            rc, ri = SetLags_and_Reg(c, i, sp, None, t, t_sp, None, control,
                                     True)

            t_sp = '' if t == None else t
            t = '' if t == None else t
            if first:
                first = False
                corrs_2 = ComputeAndAdd(rc, ri,
                                        n + str(t) + '_wo_' + n_sp + str(t_sp))
            else:
                corrs_2 = ComputeAndAdd(rc, ri,
                                        n + str(t) + '_wo_' + n_sp + str(t_sp),
                                        corrs_2)

        parents1 = corrs_2[corrs_2['pv'] < 0.05]
        parents = parents1
        # -------------------------------------------------------------------- #

        if len(parents1) > 3:
            parents1 = parents1.iloc[parents1['r'].abs().argsort()[::-1]]
            strong_parents = parents1['name'].head(3).tolist()

            first = True
            for n in parents1['name']:

                aux_strong_parents = strong_parents[:2] if \
                    all(parent != n for parent in strong_parents) else \
                    [parent for parent in strong_parents if parent != n]

                n_sp1 = aux_strong_parents[0]
                n_sp2 = aux_strong_parents[1]

                ns = n.split('_wo_')[0].split('-')
                t = IdentifyTau(ns)

                ns_sp1 = n_sp1.split('_wo_')[0].split('-')
                t_sp1 = IdentifyTau(ns_sp1)

                ns_sp2 = n_sp2.split('_wo_')[0].split('-')
                t_sp2 = IdentifyTau(ns_sp2)

                n = ns[0]
                n1 = ns_sp1[0]
                n2 = ns_sp2[0]

                c = df['indice3'].tolist()
                i = df[n].tolist()[:t]
                sp1 = df[n1].tolist()[:t_sp1]
                sp2 = df[n2].tolist()[:t_sp2]

                rrc, rri = SetLags_and_Reg(c, i, sp1, sp2, t, t_sp1, t_sp2,
                                           control, False)

                t = '' if t == None else t
                if first:
                    first = False
                    corrs_3 = ComputeAndAdd(rrc, rri, n + str(t) + '_wo_2p')
                else:
                    corrs_3 = ComputeAndAdd(rrc, rri, n + str(t) + '_wo_2p',
                                            corrs_3)

            parents2 = corrs_3[corrs_3['pv'] < 0.05]
            parents = parents2

    return parents, df
# 2
def Phase2(x,y, px, py, x_df, y_df, t):
    # control
    if x is not None:
        t = 0
        t_max = 0
    elif y is not None:
        print('test')
        t = t
        x = y
        if t !=0:
            x = x[:-t]
            t_max = t
        else:
            t_max = 0
    elif x is not None and y is not None:
        print('x or y must by None')
        print('return')
    elif x is None and y is None:
        print('x or y must by not None')
        print('return')


    for parents, p_df in zip([px, py], [x_df, y_df]):
        for n in parents['name']:
            # set t
            aux = n.split('_wo_')[0].split('-')
            n_indice = aux[0]

            try:
                t_p = int(aux[1])
                p = p_df[n_indice].tolist()[:-t_p]
            except:
                t_p = 0
                p = p_df[n_indice].tolist()

            if t_p == t_max:
                pass
            elif t_p > t_max:
                t_prima = t_p - t_max
                x = x[t_prima:]

            elif t_p < t_max:
                t_prima = t_max - t_p
                p = p[t_prima:]

            t_max = t_p if t_p > t_max else t_max

            df = pd.DataFrame({'x': x, 'p': p})

            x = RegRes(df, 'p', 'x')

    return x
#------------------------------------------------------------------------------#
c_parents, c_df = Phase1(c['var'].values, dmi.values, n34.values, mode=1)
dmi_parents, dmi_df = Phase1(c['var'].values, dmi.values, n34.values, mode=2)
n34_parents, n34_df = Phase1(c['var'].values, dmi.values, n34.values, mode=3)

# todos contra c
def ComputePhase2(c_name, c, c_parents, c_df, *args):
    # estructura a_name, a, a_parents, a_df
    first = True

    for n in args:

        index_name = n[0]
        index_values = n[1]
        index_parents = n[2]
        index_df = n[3]

        c_f = Phase2(c, None, c_parents, index_parents, c_df,
                            index_df, 0)
        for t in [0, 1, 2]:
            index_f = Phase2(None, index_values, index_parents, c_parents,
                             index_df, c_df, t)
            if first:
                first = False
                corrs = ComputeAndAdd(c_f, index_f,
                                        c_name + '_w_' + index_name + '_' +
                                        str(t))
            else:
                corrs = ComputeAndAdd(c_f, index_f,
                                        c_name + '_w_' + index_name + '_' +
                                        str(t), corrs)
    corrs = corrs[corrs['pv'] < 0.05]
    corrs = corrs.iloc[corrs['r'].abs().argsort()[::-1]]
    corrs = corrs.query('r < 0.999')

    return corrs

corrs = ComputePhase2('c', c.values, c_parents, c_df,
              ['dmi', dmi.values, dmi_parents, dmi_df],
              ['n34', n34.values, n34_parents, n34_df])

corrs = ComputePhase2('c', c.values, c_parents, c_df,
              ['dmi', dmi.values, dmi_parents, dmi_df],
              ['n34', n34.values, n34_parents, n34_df],
              ['c', c.values, c_parents, c_df])

corrs = ComputePhase2('dmi', dmi.values, dmi_parents, dmi_df,
              ['dmi', dmi.values, dmi_parents, dmi_df],
              ['n34', n34.values, n34_parents, n34_df],
              ['c', c['var'].values, c_parents, c_df])
################################################################################
# Testeo #######################################################################
c1 = hgt_anom.sel(lon=250, lat=-50, time=hgt_anom.time.dt.year.isin([2018,2020]))
c2 = SameDateAs(dmi, c1)
c3 = SameDateAs(n34, c1)
c1 = c1['var'].values
c2 = c2.values
c3 = c3.values

c1_parents, c1_df = Phase1(c1, c2, c3, mode=1)
c2_parents, c2_df = Phase1(c1, c2, c3, mode=2)
c3_parents, c3_df = Phase1(c1, c2, c3, mode=3)

ComputePhase2('c1', c1, c1_parents, c1_df,
              ['dmi', c2, c2_parents, c2_df],
              ['n34', c3, c3_parents, c3_df],
              ['c1', c1, c1_parents, c1_df])
# ok
################################################################################