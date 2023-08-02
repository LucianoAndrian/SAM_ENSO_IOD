
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import pearsonr


os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

w_dir = '/home/luciano.andrian/doc/salidas/'


########################################################################################################################
def SamDetrend(data):
    p40 = data.sel(lat=-40, method='nearest').mean(dim='lon')
    trend = np.polyfit(range(0, len(p40.psl)), p40.psl, deg=1)
    p40 = p40.psl - trend[0] * range(0, len(p40.psl))

    p65 = data.sel(lat=-65, method='nearest').mean(dim='lon')
    trend = np.polyfit(range(0, len(p65.psl)), p65.psl, deg=1)
    p65 = p65.psl - trend[0] * range(0, len(p65.psl))

    # index
    sam = (p40 - p40.mean(dim='time')) / p40.std(dim='time') - (p65 - p65.mean(dim='time')) / p65.std(dim='time')

    return sam

def PlotScatterSam(x, y, nom_x, nom_y, corr, nombre_fig):
    fig, ax = plt.subplots()
    im = plt.scatter(x=x, y=y, color='gray', edgecolors='black')
    plt.title(nom_x + ' vs ' + nom_y + '     r = ' + str(np.round(corr,2)))
    plt.xlabel(nom_x)
    plt.ylabel(nom_y)
    plt.ylim((-6, 6))
    plt.xlim((-6, 6))
    plt.grid(True)
    fig.set_size_inches(6,6)
    plt.savefig(w_dir + nombre_fig + '.jpg')
    plt.show()
    plt.close()
########################################################################################################################
psl_era = xr.open_dataset('/datos/luciano.andrian/ncfiles/psl.nc')



#pre satelite
#periodo en comun 1958-1978

# ERA20C hasta 2010
psl_era20c = xr.open_dataset("/home/luciano.andrian/doc/scrips/ERA-20C.mon.psl.nc")
psl_era20c = psl_era20c.sel(time=slice('1958-01-01', '1978-12-01'))
psl_era20c = psl_era20c.rename({'msl':'psl'})
psl_era20c = psl_era20c.rename({'latitude':'lat'})
psl_era20c = psl_era20c.rename({'longitude':'lon'})

sam_era20c = SamDetrend(psl_era20c)

sam_era20c_seasonal = xr.DataArray(np.convolve(sam_era20c, np.ones((3,))/3, mode='same')
                                , coords=[sam_era20c.time.values], dims=['time'])


# JRA-55 1958-2013
psl_jra55 = xr.open_dataset('/pikachu/datos4/Obs/slp/slp_JRA55.nc')
psl_jra55 = psl_jra55.sel(time=slice('1958-01-01', '1978-12-01'))
psl_jra55 = psl_jra55.rename({'slp':'psl'})
sam_jra55 = SamDetrend(psl_jra55)

sam_jra55_seasonal = xr.DataArray(np.convolve(sam_jra55, np.ones((3,))/3, mode='same')
                                , coords=[sam_jra55.time.values], dims=['time'])

#HadSLP2 1850-2004 5x5º
psl_had2 = xr.open_dataset('/home/luciano.andrian/doc/scrips/slp.mnmean.nc')
psl_had2 = psl_had2.sel(time=slice('1958-01-01', '1978-12-01'))
psl_had2 = psl_had2.rename({'slp':'psl'})

sam_had2 = SamDetrend(psl_had2)

# Marshal (obs)
sam_marshall = x=pd.read_fwf('/home/luciano.andrian/doc/scrips/newsam.1957.2007.txt',skiprows=1, header=None)
aux_marshall = sam_marshall
sam_marshall_df = sam_marshall.drop(sam_marshall.columns[[0]], axis = 1)
sam_marshall_df = sam_marshall_df.drop(labels=[64], axis = 0)

sam_marshall = np.array(sam_marshall_df.iloc[:, 0: 64]).reshape(-1)
#1958-1978
sam_marshall_ps = sam_marshall[12:(22*12)]

#ERA5_preliminary 1950-1978
psl_ERA5_prel = xr.open_dataset('/home/luciano.andrian/doc/scrips/ERA5_psl_50-78_PRELIMINARY.nc')
psl_ERA5_prel = psl_ERA5_prel.rename({'latitude':'lat'})
psl_ERA5_prel = psl_ERA5_prel.rename({'longitude':'lon'})
psl_ERA5_prel = psl_ERA5_prel.rename({'msl':'psl'})
psl_ERA5_prel = psl_ERA5_prel.sel(time=slice('1958-01-01', '1978-12-01'))

sam_ERA5_prel = SamDetrend(psl_ERA5_prel)

cor_mar_era = pearsonr(sam_marshall_ps, sam_era20c)[0]
cor_mar_jra = pearsonr(sam_marshall_ps, sam_jra55)[0]
cor_mar_had = pearsonr(sam_marshall_ps, sam_had2)[0]
cor_mar_era_prel = pearsonr(sam_marshall_ps, sam_ERA5_prel)[0]



PlotScatterSam(x=sam_marshall_ps, y=sam_era20c, nom_x='Marshall', nom_y='ERA20c'
               , corr=cor_mar_era, nombre_fig='SAM_corr_Mar-ERA20_ps')

PlotScatterSam(x=sam_marshall_ps, y=sam_jra55, nom_x='Marshall', nom_y='JRA-55'
               , corr=cor_mar_jra, nombre_fig='SAM_corr_Mar-JRA55_ps')

PlotScatterSam(x=sam_marshall_ps, y=sam_had2, nom_x='Marshall', nom_y='Had2'
               , corr=cor_mar_had, nombre_fig='SAM_corr_Mar-Had_ps')

PlotScatterSam(x=sam_marshall_ps, y=sam_ERA5_prel, nom_x='Marshall', nom_y='ERA5 1958-1978 Preliminary'
               , corr=cor_mar_era_prel, nombre_fig='SAM_corr_Mar-ERA5_prel_ps')

#-------------------------------------------------------------------------------------------------#

# ERA20C hasta 2010
psl_era20c = xr.open_dataset("/home/luciano.andrian/doc/scrips/ERA-20C.mon.psl.nc")
psl_era20c = psl_era20c.sel(time=slice('1950-01-01', '1978-12-01'))
psl_era20c = psl_era20c.rename({'msl':'psl'})
psl_era20c = psl_era20c.rename({'latitude':'lat'})
psl_era20c = psl_era20c.rename({'longitude':'lon'})
sam_era20c = SamDetrend(psl_era20c)

#ERA5_preliminary 1950-1978
psl_ERA5_prel = xr.open_dataset('/home/luciano.andrian/doc/scrips/ERA5_psl_50-78_PRELIMINARY.nc')
psl_ERA5_prel = psl_ERA5_prel.rename({'latitude':'lat'})
psl_ERA5_prel = psl_ERA5_prel.rename({'longitude':'lon'})
psl_ERA5_prel = psl_ERA5_prel.rename({'msl':'psl'})
sam_ERA5_prel = SamDetrend(psl_ERA5_prel)

cor = pearsonr(sam_era20c, sam_ERA5_prel)[0]
PlotScatterSam(x=sam_era20c, y=sam_ERA5_prel, nom_x='ERA20c', nom_y='ERA5 preliminary'
               , corr=cor, nombre_fig='SAM_corr_ERA20-ERA5_prel_ps')

#-------------------------------------------------------------------------------------------------#

# post satelite
# periodo comun 1979-2004
# ERA20C hasta 2010
psl_era20c = xr.open_dataset("/home/luciano.andrian/doc/scrips/ERA-20C.mon.psl.nc")
psl_era20c = psl_era20c.sel(time=slice('1979-01-01', '2004-12-01'))
psl_era20c = psl_era20c.rename({'msl':'psl'})
psl_era20c = psl_era20c.rename({'latitude':'lat'})
psl_era20c = psl_era20c.rename({'longitude':'lon'})

sam_era20c = SamDetrend(psl_era20c)

sam_era20c_seasonal = xr.DataArray(np.convolve(sam_era20c, np.ones((3,))/3, mode='same')
                                , coords=[sam_era20c.time.values], dims=['time'])
sam_era20c_anual = xr.DataArray(np.convolve(sam_era20c, np.ones((12,))/12, mode='same')
                         , coords=[sam_era20c.time.values], dims=['time'])

# ERA5 hasta 1979-2020
psl_era5 = xr.open_dataset("/home/luciano.andrian/doc/scrips/ERA5.mon.slp.nc")
psl_era5 = psl_era5.sel(time=slice('1979-01-01', '2004-12-01'))
psl_era5 = psl_era5.rename({'msl':'psl'})
psl_era5 = psl_era5.rename({'latitude':'lat'})
psl_era5 = psl_era5.rename({'longitude':'lon'})

sam_era5 = SamDetrend(psl_era5)

sam_era5_seasonal = xr.DataArray(np.convolve(sam_era5, np.ones((3,))/3, mode='same')
                                , coords=[sam_era5.time.values], dims=['time'])
sam_era5_anual = xr.DataArray(np.convolve(sam_era5, np.ones((12,))/12, mode='same')
                         , coords=[sam_era5.time.values], dims=['time'])

# JRA-55 1958-2013
psl_jra55 = xr.open_dataset('/pikachu/datos4/Obs/slp/slp_JRA55.nc')
psl_jra55 = psl_jra55.sel(time=slice('1979-01-01', '2004-12-01'))
psl_jra55 = psl_jra55.rename({'slp':'psl'})
sam_jra55 = SamDetrend(psl_jra55)

sam_jra55_seasonal = xr.DataArray(np.convolve(sam_jra55, np.ones((3,))/3, mode='same')
                                , coords=[sam_jra55.time.values], dims=['time'])

sam_jra55_anual = xr.DataArray(np.convolve(sam_jra55, np.ones((12,))/12, mode='same')
                         , coords=[sam_jra55.time.values], dims=['time'])

#HadSLP2 1850-2004 5x5º
psl_had2 = xr.open_dataset('/home/luciano.andrian/doc/scrips/slp.mnmean.nc')
psl_had2 = psl_had2.sel(time=slice('1979-01-01', '2004-12-01'))
psl_had2 = psl_had2.rename({'slp':'psl'})

sam_had2 = SamDetrend(psl_had2)

# #HadSLP2r (igual a la anterior?) 1850-2019 5ºx5º
# psl_had2r = xr.open_dataset('/home/luciano.andrian/doc/scrips/slp.mnmean.real.nc')
# psl_had2r = psl_had2r.rename({'slp':'psl'})
# sam_had2r = SamDetrend(psl_had2r)

sam_marshall_psts = sam_marshall[264:576]

cor_mar_era = pearsonr(sam_marshall_psts, sam_era20c)[0]
cor_mar_jra = pearsonr(sam_marshall_psts, sam_jra55)[0]
cor_mar_had = pearsonr(sam_marshall_psts, sam_had2)[0]
cor_mar_era5 = pearsonr(sam_marshall_psts, sam_era5)[0]


PlotScatterSam(x=sam_marshall_psts, y=sam_era20c, nom_x='Marshall', nom_y='ERA20c'
               , corr=cor_mar_era, nombre_fig='SAM_corr_Mar-ERA20_posts')

PlotScatterSam(x=sam_marshall_psts, y=sam_jra55, nom_x='Marshall', nom_y='JRA-55'
               , corr=cor_mar_jra, nombre_fig='SAM_corr_Mar-JRA55_posts')

PlotScatterSam(x=sam_marshall_psts, y=sam_had2, nom_x='Marshall', nom_y='Had2'
               , corr=cor_mar_had, nombre_fig='SAM_corr_Mar-Had_posts')

PlotScatterSam(x=sam_marshall_psts, y=sam_era5, nom_x='Marshall', nom_y='ERA5'
               , corr=cor_mar_era5, nombre_fig='SAM_corr_Mar-ERA5_posts')

# son muy parecidas... r=0.994

########################################################################################################################
# ERA20C hasta 2010
psl_era20c = xr.open_dataset("/home/luciano.andrian/doc/scrips/ERA-20C.mon.psl.nc")
psl_era20c = psl_era20c.sel(time=slice('1920-01-01', '2010-12-01'))
psl_era20c = psl_era20c.rename({'msl':'psl'})
psl_era20c = psl_era20c.rename({'latitude':'lat'})
psl_era20c = psl_era20c.rename({'longitude':'lon'})

sam_era20c = SamDetrend(psl_era20c)

sam_era20c_seasonal = xr.DataArray(np.convolve(sam_era20c, np.ones((3,))/3, mode='same')
                                , coords=[sam_era20c.time.values], dims=['time'])
sam_era20c_anual = xr.DataArray(np.convolve(sam_era20c, np.ones((12,))/12, mode='same')
                         , coords=[sam_era20c.time.values], dims=['time'])

# ERA5 hasta 1979-2020
psl_era5 = xr.open_dataset("/home/luciano.andrian/doc/scrips/ERA5.mon.slp.nc")
psl_era5 = psl_era5.sel(time=slice('1979-01-01', '2020-12-01'))
psl_era5 = psl_era5.rename({'msl':'psl'})
psl_era5 = psl_era5.rename({'latitude':'lat'})
psl_era5 = psl_era5.rename({'longitude':'lon'})

sam_era5 = SamDetrend(psl_era5)

sam_era5_seasonal = xr.DataArray(np.convolve(sam_era5, np.ones((3,))/3, mode='same')
                                , coords=[sam_era5.time.values], dims=['time'])
sam_era5_anual = xr.DataArray(np.convolve(sam_era5, np.ones((12,))/12, mode='same')
                         , coords=[sam_era5.time.values], dims=['time'])

# JRA-55 1958-2013
psl_jra55 = xr.open_dataset('/pikachu/datos4/Obs/slp/slp_JRA55.nc')
psl_jra55 = psl_jra55.rename({'slp':'psl'})
sam_jra55 = SamDetrend(psl_jra55)

sam_jra55_seasonal = xr.DataArray(np.convolve(sam_jra55, np.ones((3,))/3, mode='same')
                                , coords=[sam_jra55.time.values], dims=['time'])

sam_jra55_anual = xr.DataArray(np.convolve(sam_jra55, np.ones((12,))/12, mode='same')
                         , coords=[sam_jra55.time.values], dims=['time'])

#HadSLP2 1850-2004 5x5º
psl_had2 = xr.open_dataset('/home/luciano.andrian/doc/scrips/slp.mnmean.nc')
psl_had2 = psl_had2.rename({'slp':'psl'})

sam_had2 = SamDetrend(psl_had2)

#HadSLP2r (igual a la anterior?) 1850-2019 5ºx5º
psl_had2r = xr.open_dataset('/home/luciano.andrian/doc/scrips/slp.mnmean.real.nc')
psl_had2r = psl_had2r.rename({'slp':'psl'})
sam_had2r = SamDetrend(psl_had2r)

# son muy parecidas... r=0.994
pearsonr(sam_had2r[:1860], sam_had2)[0]

# Marshal (obs)
sam_marshall = x=pd.read_fwf('/home/luciano.andrian/doc/scrips/newsam.1957.2007.txt',skiprows=1, header=None)
aux_marshall = sam_marshall
sam_marshall_df = sam_marshall.drop(sam_marshall.columns[[0]], axis = 1)
sam_marshall_df = sam_marshall_df.drop(labels=[64], axis = 0)

sam_marshall = np.array(sam_marshall_df.iloc[:, 0: 64]).reshape(-1)

#ERA5_preliminary 1950-1978
psl_ERA5_prel = xr.open_dataset('/home/luciano.andrian/doc/scrips/ERA5_psl_50-78_PRELIMINARY.nc')
psl_ERA5_prel = psl_ERA5_prel.rename({'latitude':'lat'})
psl_ERA5_prel = psl_ERA5_prel.rename({'longitude':'lon'})
psl_ERA5_prel = psl_ERA5_prel.rename({'msl':'psl'})
sam_ERA5_prel = SamDetrend(psl_ERA5_prel)

sam_ERA5_prel_seasonal = xr.DataArray(np.convolve(sam_ERA5_prel, np.ones((3,))/3, mode='same')
                                , coords=[sam_ERA5_prel.time.values], dims=['time'])

sam_ERA5_prel_anual = xr.DataArray(np.convolve(sam_ERA5_prel, np.ones((12,))/12, mode='same')
                         , coords=[sam_ERA5_prel.time.values], dims=['time'])

# ERA20c vs Marshall
#1957-2010
aux_era20c = sam_era20c[444:]
aux_marshall = sam_marshall[:648]
cor1 = pearsonr(aux_marshall, aux_era20c)[0]

# ERA5 vs Marshall
#1979-2020
aux_era5 = sam_era5
aux_marshall2 = sam_marshall[264:]
cor2 = pearsonr(aux_marshall2, aux_era5)[0]

#ERA20c vs ERA5
#1979-2010

aux_era52= sam_era5[:384]
aux_era20c2 = sam_era20c[708:]
cor3 = pearsonr(aux_era20c2, aux_era52)[0]

#JRA-55 vs marshall
#1958-2031
aux_jra55 = sam_jra55
aux_marshall3 = sam_marshall[12:684]
cor4 = pearsonr(aux_marshall3, aux_jra55)[0]

#ERA20c vs JRA-55
#1958-2010
aux_jra552 = sam_jra55[:636]
aux_era20c3 = sam_era20c[456:]
cor5 = pearsonr(aux_era20c3, aux_jra552)[0]

#ERA5 vs JRA-55
#1979-2013
aux_jra553 = sam_jra55[252:]
aux_era53 = sam_era5[:420]
cor6 = pearsonr(aux_era53, aux_jra553)[0]

#Had2 vs Had2r 1850-2004
aux_had2 = sam_had2
aux_had2r = sam_had2r[:-12*15]
cor7 = pearsonr(aux_had2, aux_had2r)[0]

# solo Had2 1850-2004
# Had2r tiene entr 15-20 fechas repetidas con valores distintos.
# ej. dos 1975-7-1 con valores diferentes. El dato de psl no es el mismo.

#Had2 vs Marshall 1957-2004
aux_marshall4 = sam_marshall[:-12*16]
aux_had22 = sam_had2[1284:]
cor8 = pearsonr(aux_marshall4, aux_had22)[0]

#Had2 vs ERA20c 1920-2010
aux_era20c4 = sam_era20c[:-12*6]
aux_had23 = sam_had2[840:]
cor9 = pearsonr(aux_era20c4, aux_had23)[0]

#Had2 vs ERA5 1979-2019
aux_era54 = sam_era5[:-12*16]
aux_had24 = sam_had2[1548:]
cor10 = pearsonr(aux_era54, aux_had24)[0]

#Had2 vs JRA-55 1958-2013
aux_jra554 = sam_jra55[:-9*12]
aux_had25 = sam_had2[1548-12*21:]
cor11 = pearsonr(aux_jra554, aux_had25)[0]

#Mar vs ERA5 prel 1957-1978
aux_marshall5 = sam_marshall[:-42*12]
aux_era5prel = sam_ERA5_prel[7*12:]
cor12 = pearsonr(aux_marshall5, aux_era5prel)[0]

#ERA20c vs ERA5 prel 1950-1958
aux_era20c5 = sam_era20c[30*12:-32*12]
aux_era5prel2 = sam_ERA5_prel
cor13 = pearsonr(aux_era20c5, aux_era5prel2)[0]

def PlotScatterSam(x, y, nom_x, nom_y, corr, nombre_fig):
    fig, ax = plt.subplots()
    im = plt.scatter(x=x, y=y, color='gray', edgecolors='black')
    plt.title(nom_x + ' vs ' + nom_y + '     r = ' + str(np.round(corr,2)))
    plt.xlabel(nom_x)
    plt.ylabel(nom_y)
    plt.ylim((-6, 6))
    plt.xlim((-6, 6))
    plt.grid(True)
    fig.set_size_inches(6,6)
    plt.savefig(w_dir + nombre_fig + '.jpg')
    plt.show()
    plt.close()


PlotScatterSam(x=aux_marshall, y=aux_era20c, nom_x='Marshall', nom_y='ERA20c', corr=cor1, nombre_fig='SAM_corr_Mar-ERA20')
PlotScatterSam(x=aux_marshall2, y= aux_era5, nom_x='Marshall', nom_y='ERA5', corr=cor2, nombre_fig='SAM_corr_Mar-ERA5')
PlotScatterSam(x=aux_era20c2, y=aux_era52, nom_x='ERA20c', nom_y='ERA5', corr=cor3, nombre_fig='SAM_corr_ERA20-ERA5')
PlotScatterSam(x=aux_marshall3, y=aux_jra55, nom_x='Marshall', nom_y='JRA-55', corr=cor4, nombre_fig='SAM_corr_Mar-JRA55')
PlotScatterSam(x=aux_era20c3, y=aux_jra552, nom_x='ERA20c', nom_y='JRA-55', corr=cor5, nombre_fig='SAM_corr_ERA20-JRA-55')
PlotScatterSam(x=aux_era53, y=aux_jra553, nom_x='ERA5', nom_y='JRA-55', corr=cor6, nombre_fig='SAM_corr_ERA5-JRA-55')
PlotScatterSam(x=aux_had2, y=aux_had2r, nom_x='Had2', nom_y='Had2r', corr=cor7, nombre_fig='SAM_corr_Had2-Had2')
PlotScatterSam(x=aux_marshall4, y=aux_had22, nom_x='Marshall', nom_y='Had2', corr=cor8, nombre_fig='SAM_corr_Mar-Had2')
PlotScatterSam(x=aux_era20c4, y=aux_had23, nom_x='ERA20c', nom_y='Had2', corr=cor9, nombre_fig='SAM_corr_ERA20c-Had2')
PlotScatterSam(x=aux_era54, y=aux_had24, nom_x='ERA5', nom_y='Had2', corr=cor10, nombre_fig='SAM_corr_ERA5-Had2r')
PlotScatterSam(x=aux_jra554, y=aux_had25, nom_x='JRA-55', nom_y='Had2', corr=cor11, nombre_fig='SAM_corr_JRA-55-Had2')




# Seasonal
mam_marshall = sam_marshall_df[[3,4,5]].apply(np.mean, axis=1)
jja_marshall = sam_marshall_df[[6,7,8]].apply(np.mean, axis=1)
son_marshall = sam_marshall_df[[9,10,11]].apply(np.mean, axis=1)

aux = sam_marshall_df[[1,2]].drop(labels=0, axis=0)
aux2 = sam_marshall_df[[12]].drop(labels=63, axis=0)
aux['12'] = aux2.values
djf_marshall = aux.apply(np.mean,axis=1)

aux2 = xr.DataArray(np.convolve(aux_era20c, np.ones((3,)) / 3, mode='same')
             , coords=[aux_era20c.time.values], dims=['time'])
aux3 = xr.DataArray(np.convolve(aux_era5, np.ones((3,)) / 3, mode='same')
             , coords=[aux_era5.time.values], dims=['time'])

aux4 = xr.DataArray(np.convolve(aux_jra55, np.ones((3,)) / 3, mode='same')
             , coords=[aux_jra55.time.values], dims=['time'])

aux5 = xr.DataArray(np.convolve(aux_had22, np.ones((3,)) / 3, mode='same')
             , coords=[aux_had22.time.values], dims=['time'])

aux6 = xr.DataArray(np.convolve(aux_era5prel, np.ones((3,)) / 3, mode='same')
             , coords=[aux_era5prel.time.values], dims=['time'])



def SamSeasonalBarPlot(marshall, era20c, era5, jra55, Had2r, era5prel, main_month_era = 1, titulo = 'Seasons', seasons='djf', nombre_fig = "prueba"):
    aux = marshall
    aux2 = era20c
    aux3 = era5
    # aux4 = jra55
    # aux5 = Had2r
    aux6 = era5prel

    m = main_month_era

    if seasons == 'djf':
        cont = 1
    else:
        cont = 0

    cor1 = pearsonr(aux[:-10], aux2.groupby('time.month')[m].values[cont:])[0]
    cor2 = pearsonr(aux[22:], aux3.groupby('time.month')[m].values[cont:])[0]
    cor11 = pearsonr(aux[:-42], aux6.groupby('time.month')[m].values[cont:])[0]

    cor3 = pearsonr(aux2.groupby('time.month')[m].values[22:],
                    aux3.groupby('time.month')[m].values[:-10])[0]

    cor12 = pearsonr(aux2.groupby('time.month')[m].values[:-32],
                     aux6.groupby('time.month')[m].values)[0]


    # cor4 = pearsonr(aux[1:-7], aux4.groupby('time.month')[m].values[cont:])[0]
    # cor5 = pearsonr(aux2.groupby('time.month')[m].values[1:], aux4.groupby('time.month')[m].values[:-3])[0]
    # cor6 = pearsonr(aux3.groupby('time.month')[m].values[:-7], aux4.groupby('time.month')[m].values[21:])[0]
    #
    # cor7 = pearsonr(aux[:-16], aux5.groupby('time.month')[m].values[cont:])[0]
    # cor8 = pearsonr(aux2.groupby('time.month')[m].values[:-6], aux5.groupby('time.month')[m].values)[0]
    # cor9 = pearsonr(aux3.groupby('time.month')[m].values[:-16], aux5.groupby('time.month')[m].values[22:])[0]
    # cor10 = pearsonr(aux4.groupby('time.month')[m].values[:-9], aux5.groupby('time.month')[m].values[1:])[0]


    fig, ax = plt.subplots()
    plt.bar(range(0 + 1957, len(aux) + 1957), aux, color='darkgrey', edgecolor='black',
            label='Marshall')
    plt.bar(range(0 + 1957, len(aux) + 1957 - 10), aux2.groupby('time.month')[m].values[cont:], alpha=1, fill=False,
            label='ERA20c', edgecolor='red')
    plt.bar(range(0 + 1957 + 22-cont, len(aux) + 1957), aux3.groupby('time.month')[m].values, alpha=1, fill=False, edgecolor='blue',
            label='ERA5')
    plt.bar(range(0 + 1957, len(aux) + 1957 - 42), aux6.groupby('time.month')[m].values[cont:], alpha=1, fill=False,
            edgecolor='lime', label='ERA5 1957-1978 Prel')
    # plt.bar(range(0 + 1957 + 1 - cont, len(aux) + 1950), aux4.groupby('time.month')[m].values, alpha=1, fill=False, edgecolor='lime',
    #         label='JRA-55')
    #
    # plt.bar(range(0 + 1957 - cont, len(aux) + 1957-16), aux5.groupby('time.month')[m].values, alpha=1, fill=False,
    #         edgecolor='deepskyblue',
    #         label='Had2')

    plt.ylim(-6, 6)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.ylabel('SAM')
    plt.xlabel('')
    plt.title('SAM - ' + titulo)

    plt.legend(framealpha=1, loc='lower center', ncol=5,bbox_to_anchor=(0.5, -0.15))
    fig.set_size_inches(7, 5)
    plt.text(1955, -5.5, 'Marshall - ERA20c\n' + '     r = ' + str(np.round(cor1,2)), dict(size=10), backgroundcolor='darkgray')
    plt.text(1982, -5.5, 'ERA20c - ERA5\n' + '     r = ' + str(np.round(cor3, 2)), dict(size=10), backgroundcolor='lightgrey')
    plt.text(2007, -5.5, 'Marshall - ERA5\n' + '     r =  ' + str(np.round(cor2, 2)), dict(size=10), backgroundcolor='darkgray')
    plt.text(1955, 4.8, 'Marshall - ERA5 Prel\n' + '     r =  ' + str(np.round(cor11, 2)), dict(size=10), backgroundcolor='darkgray')
    plt.text(1982, 4.8, 'ERA20c - ERA5 Prel\n' + '    r =  ' + str(np.round(cor12, 2)), dict(size=10), backgroundcolor='lightgrey')
    #plt.text(2007, 3.5, 'ERA5 - JRA-55\n' + '    r =  ' + str(np.round(cor6, 2)), dict(size=10), backgroundcolor='lightgrey')
    #plt.text(2007, 5, 'Marshall - Had2\n' + '     r =  ' + str(np.round(cor7, 2)), dict(size=10), backgroundcolor='darkgray')
    #plt.text(1955, 3.5, 'ERA20c - Had2\n' + '    r =  ' + str(np.round(cor8, 2)), dict(size=10), backgroundcolor='lightgrey')
    #plt.text(1982, 3.5, 'ERA5 - Had2\n' + '    r =  ' + str(np.round(cor9, 2)), dict(size=10), backgroundcolor='lightgrey')
    #plt.text(2006, -4, 'JRA-55 - Had2\n' + '    r =  ' + str(np.round(cor10, 2)), dict(size=10),backgroundcolor='lightgrey')

    plt.savefig(w_dir + nombre_fig + '.jpg')
    plt.grid()
    plt.show()
    plt.close()




SamSeasonalBarPlot(marshall=mam_marshall, era20c=aux2, era5=aux3, jra55=aux4, Had2r=aux5, era5prel=aux6,
                   main_month_era=4, titulo = 'MAM',
                   seasons='no importa', nombre_fig='SAM_barplot_MAM')

SamSeasonalBarPlot(marshall=jja_marshall, era20c=aux2, era5=aux3, jra55=aux4, Had2r=aux5, era5prel=aux6,
                   main_month_era=7, titulo = 'JJA',
                   seasons='no importa',nombre_fig='SAM_barplot_JJA')

SamSeasonalBarPlot(marshall=son_marshall, era20c=aux2, era5=aux3, jra55=aux4, Had2r=aux5, era5prel=aux6,
                   main_month_era=10, titulo = 'SON',
                   seasons='no importa', nombre_fig='SAM_barplot_SON')

SamSeasonalBarPlot(marshall=djf_marshall, era20c=aux2, era5=aux3, jra55=aux4, Had2r=aux5, era5prel=aux6,
                   main_month_era=1, titulo = 'DJF',
                   seasons='djf', nombre_fig='SAM_barplot_DJF')




sam_mar = np.convolve(sam_marshall, np.ones((12,))/12, mode='same')
plt.plot(sam_marshall, color='black', alpha=0.2)
plt.bar(range(0, len(sam_mar)), sam_mar, color='darkgray', label='Marshall')

plt.bar(range(0, len(sam_mar)-120)
        , np.convolve(aux2, np.ones((12,))/12, mode='same'), color='red', label='ERA20c', alpha=0.5)

plt.bar(range(0+264, len(sam_mar))
        , np.convolve(aux3, np.ones((12,))/12, mode='same'), color='blue', label='ERA5', alpha=0.5)

plt.bar(range(0+1, len(sam_mar)-95)
        , np.convolve(aux4, np.ones((12,))/12, mode='same'), color='lime', label='JRA-55', alpha=0.5)

plt.legend()
# plt.plot(np.convolve(aux2, np.ones((12,))/12), color = 'red', label='ERA20c')
# plt.plot(np.convolve(aux3, np.ones((12,))/12), color= 'blue', label='ERA5')
plt.ylim(-4, 4)
plt.axhline(y=0, color='black', linestyle='-')
plt.show()

#
#
# aux = x=pd.read_fwf('/home/luciano.andrian/doc/scrips/hadslp2.0_acts.asc',skiprows=1, header=None)
#
