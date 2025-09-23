# ENSO vs SAM Fogt. 2010
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

w_dir = '/home/luciano.andrian/doc/salidas/SAM/'

sst = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")

########################################################################################################################

def Preproc(aux):
    aux2 = []
    for i in range(1, 13):
        recta = np.polyfit(range(0, len(aux[i])), aux[i], deg=1)
        detrend = aux[i] - recta[0] * range(0, len(aux[i]))
        aux2.append(xr.DataArray(detrend, dims=['time']))

    aux = xr.concat(aux2, dim='time')

    index_month_mean = aux.groupby('time.month').mean()
    index_month_std = aux.groupby('time.month').std()

    aux3 = []
    for i in range(1, 13):
        aux2 = ((aux.groupby('time.month')[i] - index_month_mean.groupby('month')[i])) \
               / index_month_std.groupby('month')[i]

        aux3.append(xr.DataArray(aux2, dims=['time']))

    result = xr.concat(aux3, dim='time')

    return(result)

def Sam(data):
    p40 = data.sel(lat=-40, method='nearest').mean(dim='lon')
    #trend = np.polyfit(range(0, len(p40.psl)), p40.psl, deg=1)
    p40 = p40.psl #- trend[0] * range(0, len(p40.psl))

    p65 = data.sel(lat=-65, method='nearest').mean(dim='lon')
    #trend = np.polyfit(range(0, len(p65.psl)), p65.psl, deg=1)
    p65 = p65.psl #- trend[0] * range(0, len(p65.psl))

    # index
    sam = (p40 - p40.mean(dim='time')) / p40.std(dim='time') - (p65 - p65.mean(dim='time')) / p65.std(dim='time')

    return sam

def ENSOSAMTable(sam, ninio34_f=ninio34_f):
    seasons = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]

    import pandas as pd
    cases = pd.DataFrame(columns=['DJF', 'MAM', 'JJA', 'SON', 'Anual'], dtype=float)
    names = ['DJF', 'MAM', 'JJA', 'SON', 'Anual']
    aux = np.empty_like(names, dtype=float)
    aux2 = np.empty_like(names, dtype=float)
    aux3 = np.empty_like(names, dtype=float)
    aux4 = np.empty_like(names, dtype=float)
    aux5 = np.empty_like(names, dtype=float)
    aux6 = np.empty_like(names, dtype=float)
    aux7 = np.empty_like(names, dtype=float)
    aux8 = np.empty_like(names, dtype=float)
    aux9 = np.empty_like(names, dtype=float)
    for i in range(len(seasons)):
        aux_sam = sam.where(sam.time.dt.month.isin(seasons[i]), drop=True)
        aux_ninio34_f = ninio34_f.where(ninio34_f.time.dt.month.isin(seasons[i]), drop=True)

        #### SAM + only ####
        time_sam = aux_sam.where(aux_sam > 0.5, drop=True).time
        time_n34 = aux_ninio34_f.where(aux_ninio34_f < abs(0.5), drop=True).time

        l = len(time_sam.where(time_n34 == time_sam, drop=True))
        aux[i] = l

        #### SAM - only ####
        time_sam = aux_sam.where(aux_sam < -0.5, drop=True).time

        l = len(time_sam.where(time_n34 == time_sam, drop=True))
        aux2[i] = l

        #### La Niña only #### (>0.5, -1*n34)
        time_sam = aux_sam.where(aux_sam < abs(0.5), drop=True).time
        time_n34 = aux_ninio34_f.where(aux_ninio34_f > 0.5, drop=True).time

        l = len(time_sam.where(time_n34 == time_sam, drop=True))
        aux3[i] = l

        #### El Niño only #### (<-0.5, -1*n34)
        time_n34 = aux_ninio34_f.where(aux_ninio34_f < -0.5, drop=True).time

        l = len(time_sam.where(time_n34 == time_sam, drop=True))
        aux4[i] = l

        #### La Niña/Sam+ ####
        time_sam = aux_sam.where(aux_sam > 0.5, drop=True).time
        time_n34 = aux_ninio34_f.where(aux_ninio34_f > 0.5, drop=True).time

        l = len(time_sam.where(time_n34 == time_sam, drop=True))
        aux5[i] = l

        #### El Niño/Sam- ####
        time_sam = aux_sam.where(aux_sam < 0.5, drop=True).time
        time_n34 = aux_ninio34_f.where(aux_ninio34_f < 0.5, drop=True).time

        l = len(time_sam.where(time_n34 == time_sam, drop=True))
        aux6[i] = l

        #### La Niña/Sam- ####
        time_sam = aux_sam.where(aux_sam < 0.5, drop=True).time
        time_n34 = aux_ninio34_f.where(aux_ninio34_f > 0.5, drop=True).time

        l = len(time_sam.where(time_n34 == time_sam, drop=True))
        aux7[i] = l

        #### El Niño/Sam+ ####
        time_sam = aux_sam.where(aux_sam > 0.5, drop=True).time
        time_n34 = aux_ninio34_f.where(aux_ninio34_f < 0.5, drop=True).time

        l = len(time_sam.where(time_n34 == time_sam, drop=True))
        aux8[i] = l

        #### climatology ####
        time_sam = aux_sam.where(aux_sam < abs(0.5), drop=True).time
        time_n34 = aux_ninio34_f.where(aux_ninio34_f < abs(0.5), drop=True).time

        l = len(time_sam.where(time_n34 == time_sam, drop=True))
        aux9[i] = l

    cases = cases.append(pd.DataFrame([aux], columns=names))
    cases = cases.append(pd.DataFrame([aux2], columns=names))
    cases = cases.append(pd.DataFrame([aux3], columns=names))
    cases = cases.append(pd.DataFrame([aux4], columns=names))
    cases = cases.append(pd.DataFrame([aux5], columns=names))
    cases = cases.append(pd.DataFrame([aux6], columns=names))
    cases = cases.append(pd.DataFrame([aux7], columns=names))
    cases = cases.append(pd.DataFrame([aux8], columns=names))
    cases = cases.append(pd.DataFrame([aux9], columns=names))

    return cases

def ReorderTable(cases2):
    result = pd.DataFrame(columns=['0', '1', '2'], dtype=float)

    row1 = [cases2.values[4], cases2.values[5], cases2.values[2]]
    row2 = [cases2.values[6], cases2.values[7], cases2.values[3]]
    row3 = [cases2.values[0], cases2.values[1], cases2.values[8]]

    result = result.append(pd.DataFrame([row1], columns=result.columns))
    result = result.append(pd.DataFrame([row2], columns=result.columns))
    result = result.append(pd.DataFrame([row3], columns=result.columns))
    return (result)

def CtgTest(tabla, alpha=0.05):
    tot_row = tabla.apply(np.sum, axis=1)
    tot_col = tabla.apply(np.sum, axis=0)
    total = sum(tot_col)

    tabla_rand = pd.DataFrame(columns=tabla.columns, dtype=float)

    for r in range(0, len(tot_row)):
        aux = tot_row.values[r] * tot_col.values / total

        tabla_rand = tabla_rand.append(pd.DataFrame([aux], columns=tabla.columns))

    est = sum((((tabla - tabla_rand) ** 2) / tabla_rand).sum())
    import scipy.stats as stats
    est_teo = stats.chi2.ppf(1 - alpha, (len(tot_row) - 1) * (len(tot_col) - 1))

    if abs(est) > abs(est_teo):
        print('Rechazo H0 con un ' + str((1 - alpha) * 100) + '%  de confianza"')
        print('ChiSqr = ' + str(est))

    return (tabla_rand)

def Re_ReorderTable(tabla_rnd, nombre):
    tabla_rnd = np.round(tabla_rnd, 2)
    aux = [tabla_rnd.iloc[0][2], tabla_rnd.iloc[1][2], tabla_rnd.iloc[2][0],
           tabla_rnd.iloc[2][1], tabla_rnd.iloc[0][0], tabla_rnd.iloc[1][0],
           tabla_rnd.iloc[0][1], tabla_rnd.iloc[1][1], tabla_rnd.iloc[2][2]]

    result = pd.DataFrame(columns=[nombre], dtype=float)

    result = result.append(pd.DataFrame(aux, columns=result.columns))

    return (result)

def PlotENSAM(sam, title, dataname, save=False):
    fig, ax = plt.subplots()
    im = plt.scatter(x=sam, y=-1 * ninio34_f, marker='o', s=20, edgecolor='black', color='gray')

    plt.ylim((-4, 4));
    plt.xlim((-4, 4))
    plt.axhspan(-0.5, 0.5, alpha=0.2, color='black', zorder=0)
    plt.axvspan(-0.5, 0.5, alpha=0.2, color='black', zorder=0)
    # ax.grid(True)
    fig.set_size_inches(6, 6)
    plt.xlabel('SAM')
    plt.ylabel('-Niño 3.4')

    plt.text(-3.8, 3.4, '    LN/SAM- \n Out of Phase', dict(size=10))
    plt.text(-.1, 3.4, 'LN', dict(size=10))
    plt.text(+2.6, 3.4, ' LN/SAM+ \n In Phase', dict(size=10))
    plt.text(+2.6, -.1, 'SAM+', dict(size=10))
    plt.text(+2.3, -3.4, '    EN/SAM+ \n Out of Phase', dict(size=10))
    plt.text(-.1, -3.4, 'EN', dict(size=10))
    plt.text(-3.8, -3.4, ' EN/SAM- \n In Phase', dict(size=10))
    plt.text(-3.8, -.1, 'SAM-', dict(size=10))
    plt.title(title)
    if save:
        plt.savefig(w_dir + 'ENSO-SAM' + dataname + '.jpg')
    plt.show()

def SAM_ENSO(sam):
    cases = ENSOSAMTable(sam)

    tabla = ReorderTable(cases.DJF)
    tabla_DJF = CtgTest(tabla=tabla, alpha=0.05)
    tabla_DJF = Re_ReorderTable(tabla_DJF, 'DJF')

    tabla = ReorderTable(cases.MAM)
    tabla_MAM = CtgTest(tabla=tabla, alpha=0.05)
    tabla_MAM = Re_ReorderTable(tabla_MAM, 'MAM')

    tabla = ReorderTable(cases.JJA)
    tabla_JJA = CtgTest(tabla=tabla, alpha=0.05)
    tabla_JJA = Re_ReorderTable(tabla_JJA, 'JJA')

    tabla = ReorderTable(cases.SON)
    tabla_SON = CtgTest(tabla=tabla, alpha=0.05)
    tabla_SON = Re_ReorderTable(tabla_SON, 'SON')

    tabla = ReorderTable(cases.Anual)
    tabla_Anual = CtgTest(tabla=tabla, alpha=0.05)
    tabla_Anual = Re_ReorderTable(tabla_Anual, 'Anual')

    expect = pd.concat([tabla_DJF, tabla_MAM, tabla_JJA, tabla_SON, tabla_Anual], axis=1)

    cases.index = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    table = pd.concat([cases, expect], axis=1)
    return table

########################################################################################################################
### ERA-20c 1920-2010 ###
ninio34 = sst.sel(lat=slice(4.0,-4.0), lon=slice(190, 240), time=slice('1920-01-01', '2010-12-31'))
ninio34 = ninio34.sst.mean(['lon', 'lat'], skipna=True)
ninio34_f = Preproc(ninio34.groupby('time.month')) # se usa en Sam()

psl_era20c = xr.open_dataset("/datos/luciano.andrian/ncfiles/psl.nc")
psl_era20c = psl_era20c.sel(time=slice('1920-01-01', '2010-12-01'))
psl_era20c = psl_era20c.rename({'msl':'psl'})
psl_era20c = psl_era20c.rename({'latitude':'lat'})
psl_era20c = psl_era20c.rename({'longitude':'lon'})

aux = Sam(psl_era20c)
sam = Preproc(aux.groupby('time.month'))
PlotENSAM(sam, title='ERA-20c 1920-2010', dataname='ERA20', save = True)
table_ERA20c = SAM_ENSO(sam)


### ERA5 1979-2020 ###
ninio34 = sst.sel(lat=slice(4.0,-4.0), lon=slice(190, 240), time=slice('1979-01-01', '2020-12-31'))
ninio34 = ninio34.sst.mean(['lon', 'lat'], skipna=True)
ninio34_f = Preproc(ninio34.groupby('time.month'))

psl_era5 = xr.open_dataset("/home/luciano.andrian/doc/scrips/ERA5.mon.slp.nc")
psl_era5 = psl_era5.sel(time=slice('1979-01-01', '2020-12-01'))
psl_era5 = psl_era5.rename({'msl':'psl'})
psl_era5 = psl_era5.rename({'latitude':'lat'})
psl_era5 = psl_era5.rename({'longitude':'lon'})

aux = Sam(psl_era5)
sam = Preproc(aux.groupby('time.month'))
PlotENSAM(sam, title='ERA5 1979-2020', dataname='ERA5', save = True)
table_ERA5 = SAM_ENSO(sam)


### JRA-55 1958-2013 ###
ninio34 = sst.sel(lat=slice(4.0,-4.0), lon=slice(190, 240), time=slice('1958-01-01', '2013-12-31'))
ninio34 = ninio34.sst.mean(['lon', 'lat'], skipna=True)
ninio34_f = Preproc(ninio34.groupby('time.month'))

psl_jra55 = xr.open_dataset('/pikachu/datos4/Obs/slp/slp_JRA55.nc')
psl_jra55 = psl_jra55.sel(time=slice('1958-01-01', '2013-12-31'))
psl_jra55 = psl_jra55.rename({'slp':'psl'})
sam_jra55 = Sam(psl_jra55)

aux = Sam(psl_jra55)
sam = Preproc(aux.groupby('time.month'))
PlotENSAM(sam, title='JRA-55 1958-2013', dataname='JRA-55', save = True)
table_JRA55 = SAM_ENSO(sam)


### HadSLP2 1850-2004 5x5 ###
ninio34 = sst.sel(lat=slice(4.0,-4.0), lon=slice(190, 240), time=slice('1920-01-01', '2004-12-31'))
ninio34 = ninio34.sst.mean(['lon', 'lat'], skipna=True)
ninio34_f = Preproc(ninio34.groupby('time.month'))

psl_had2 = xr.open_dataset('/home/luciano.andrian/doc/scrips/slp.mnmean.nc')
psl_had2 = psl_had2.sel(time=slice('1920-01-01', '2004-12-31'))
psl_had2 = psl_had2.rename({'slp':'psl'})

aux = Sam(psl_had2)
sam = Preproc(aux.groupby('time.month'))
PlotENSAM(sam, title='Had2 1920-2004', dataname='Had2', save = True)
table_Had2 = SAM_ENSO(sam)
