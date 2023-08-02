"""
Funciones generales para ENSO_IOD
"""
from itertools import groupby
import xarray as xr
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import statsmodels.formula.api as sm
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import regionmask
import matplotlib.pyplot as plt
import matplotlib.path as mpath

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

out_dir = '~/'

# Niño3.4 & DMI ########################################################################################################
def MovingBasePeriodAnomaly(data, start=1920, end=2020):
    import xarray as xr
    # first five years
    start_num = start
    start = str(start)

    initial = data.sel(time=slice(start + '-01-01', str(start_num + 5) + '-12-31')).groupby('time.month') - \
              data.sel(time=slice(str(start_num - 14) + '-01-01', str(start_num + 5 + 10) + '-12-31')).groupby(
                  'time.month').mean('time')


    start_num = start_num + 6
    result = initial

    while (start_num != end-4) & (start_num < end-4):

        aux = data.sel(time=slice(str(start_num) + '-01-01', str(start_num + 4) + '-12-31')).groupby('time.month') - \
              data.sel(time=slice(str(start_num - 15) + '-01-01', str(start_num + 4 + 10) + '-12-31')).groupby(
                  'time.month').mean('time')

        start_num = start_num + 5

        result = xr.concat([result, aux], dim='time')

    if start_num > end - 4:
        start_num = start_num - 5

    aux = data.sel(time=slice(str(start_num) + '-01-01', str(start_num + 4) + '-12-31')).groupby('time.month') - \
          data.sel(time=slice(str(end-29) + '-01-01', str(end) + '-12-31')).groupby('time.month').mean('time')

    result = xr.concat([result, aux], dim='time')

    return (result)

def Nino34CPC(data, start=1920, end=2020):

    # Calculates the Niño3.4 index using the CPC criteria.
    # Use ERSSTv5 to obtain exactly the same values as those reported.

    #from Funciones import MovingBasePeriodAnomaly

    start_year = str(start-14)
    end_year = str(end)
    sst = data
    # N34
    ninio34 = sst.sel(lat=slice(4.0, -4.0), lon=slice(190, 240), time=slice(start_year+'-01-01', end_year + '-12-31'))
    ninio34 = ninio34.sst.mean(['lon', 'lat'], skipna=True)

    # compute monthly anomalies
    ninio34 = MovingBasePeriodAnomaly(data=ninio34, start=start, end=end)

    # compute 5-month running mean
    ninio34_filtered = np.convolve(ninio34, np.ones((3,)) / 3, mode='same')  #
    ninio34_f = xr.DataArray(ninio34_filtered, coords=[ninio34.time.values], dims=['time'])

    aux = abs(ninio34_f) > 0.5
    results = []
    for k, g in groupby(enumerate(aux.values), key=lambda x: x[1]):
        if k:
            g = list(g)
            results.append([g[0][0], len(g)])

    n34 = []
    n34_df = pd.DataFrame(columns=['N34', 'Años', 'Mes'], dtype=float)
    for m in range(0, len(results)):
        # True values
        len_true = results[m][1]

        # True values for at least 5 consecutive seasons
        if len_true >= 5:
            a = results[m][0]
            n34.append([np.arange(a, a + results[m][1]), ninio34_f[np.arange(a, a + results[m][1])].values])

            for l in range(0, len_true):
                if l < (len_true - 2):
                    main_month_num = results[m][0] + 1 + l
                    if main_month_num != 1210:
                        n34_df = n34_df.append({'N34': np.around(ninio34_f[main_month_num].values, 2),
                                            'Años': np.around(ninio34_f[main_month_num]['time.year'].values),
                                            'Mes': np.around(ninio34_f[main_month_num]['time.month'].values)},
                                           ignore_index=True)

    return ninio34_f, n34, n34_df

def DMIndex(iodw, iode, sst_anom_sd=True, xsd=0.5, opposite_signs_criteria=True):

    import numpy as np
    from itertools import groupby
    import pandas as pd

    limitsize = len(iodw) - 2

    # dipole mode index
    dmi = iodw - iode

    # criteria
    western_sign = np.sign(iodw)
    eastern_sign = np.sign(iode)
    opposite_signs = western_sign != eastern_sign



    sd = np.std(dmi) * xsd
    print(str(sd))
    sdw = np.std(iodw.values) * xsd
    sde = np.std(iode.values) * xsd

    valid_criteria = dmi.__abs__() > sd

    results = []
    if opposite_signs_criteria:
        for k, g in groupby(enumerate(opposite_signs.values), key=lambda x: x[1]):
            if k:
                g = list(g)
                results.append([g[0][0], len(g)])
    else:
        for k, g in groupby(enumerate(valid_criteria.values), key=lambda x: x[1]):
            if k:
                g = list(g)
                results.append([g[0][0], len(g)])


    iods = pd.DataFrame(columns=['DMI', 'Años', 'Mes'], dtype=float)
    dmi_raw = []
    for m in range(0, len(results)):
        # True values
        len_true = results[m][1]

        # True values for at least 3 consecutive seasons
        if len_true >= 3:

            for l in range(0, len_true):

                if l < (len_true - 2):

                    main_month_num = results[m][0] + 1 + l
                    if main_month_num != limitsize:
                        main_month_name = dmi[main_month_num]['time.month'].values  # "name" 1 2 3 4 5

                        main_season = dmi[main_month_num]
                        b_season = dmi[main_month_num - 1]
                        a_season = dmi[main_month_num + 1]

                        # abs(dmi) > sd....(0.5*sd)
                        aux = (abs(main_season.values) > sd) & \
                              (abs(b_season) > sd) & \
                              (abs(a_season) > sd)

                        if sst_anom_sd:
                            if aux:
                                sstw_main = iodw[main_month_num]
                                sstw_b = iodw[main_month_num - 1]
                                sstw_a = iodw[main_month_num + 1]
                                #
                                aux2 = (abs(sstw_main) > sdw) & \
                                       (abs(sstw_b) > sdw) & \
                                       (abs(sstw_a) > sdw)
                                #
                                sste_main = iode[main_month_num]
                                sste_b = iode[main_month_num - 1]
                                sste_a = iode[main_month_num + 1]

                                aux3 = (abs(sste_main) > sde) & \
                                       (abs(sste_b) > sde) & \
                                       (abs(sste_a) > sde)

                                if aux3 & aux2:
                                    iods = iods.append({'DMI': np.around(dmi[main_month_num].values, 2),
                                                        'Años': np.around(dmi[main_month_num]['time.year'].values),
                                                        'Mes': np.around(dmi[main_month_num]['time.month'].values)},
                                                       ignore_index=True)

                                    a = results[m][0]
                                    dmi_raw.append([np.arange(a, a + results[m][1]),
                                                    dmi[np.arange(a, a + results[m][1])].values])


                        else:
                            if aux:
                                iods = iods.append({'DMI': np.around(dmi[main_month_num].values, 2),
                                                    'Años': np.around(dmi[main_month_num]['time.year'].values),
                                                    'Mes': np.around(dmi[main_month_num]['time.month'].values)},
                                                   ignore_index=True)

    return iods, dmi_raw

def DMI(per = 0, filter_bwa = True, filter_harmonic = True,
        filter_all_harmonic=True, harmonics = [],
        start_per=1920, end_per=2020,
        sst_anom_sd=True, opposite_signs_criteria=True):

    western_io = slice(50, 70) # definicion tradicional

    start_per = str(start_per)
    end_per = str(end_per)

    if per == 2:
        movinganomaly = True
        start_year = '1906'
        end_year = '2020'
        change_baseline = False
        start_year2 = '1920'
        end_year2 = '2020_30r5'
        print('30r5')
    else:
        movinganomaly = False
        start_year = start_per
        end_year = end_per
        change_baseline = False
        start_year2 = '1920'
        end_year2 = end_per
        print('All')

    ##################################### DATA #####################################
    # ERSSTv5
    sst = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")
    dataname = 'ERSST'
    ##################################### Pre-processing #####################################
    iodw = sst.sel(lat=slice(10.0, -10.0), lon=western_io,
                       time=slice(start_year + '-01-01', end_year + '-12-31'))
    iodw = iodw.sst.mean(['lon', 'lat'], skipna=True)
    iodw2 = iodw
    if per == 2:
        iodw2 = iodw2[168:]
    # -----------------------------------------------------------------------------------#
    iode = sst.sel(lat=slice(0, -10.0), lon=slice(90, 110),
                   time=slice(start_year + '-01-01', end_year + '-12-31'))
    iode = iode.sst.mean(['lon', 'lat'], skipna=True)
    # -----------------------------------------------------------------------------------#
    bwa = sst.sel(lat=slice(20.0, -20.0), lon=slice(40, 110),
                  time=slice(start_year + '-01-01', end_year + '-12-31'))
    bwa = bwa.sst.mean(['lon', 'lat'], skipna=True)
    # ----------------------------------------------------------------------------------#

    if movinganomaly:
        iodw = MovingBasePeriodAnomaly(iodw)
        iode = MovingBasePeriodAnomaly(iode)
        bwa = MovingBasePeriodAnomaly(bwa)
    else:
        # change baseline
        if change_baseline:
            iodw = iodw.groupby('time.month') - \
                   iodw.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                       'time')

            iode = iode.groupby('time.month') - \
                   iode.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                       'time')

            bwa = bwa.groupby('time.month') - \
                  bwa.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                      'time')
            print('baseline: ' + str(start_year2) + ' - ' + str(end_year2))
        else:
            print('baseline: All period')
            iodw = iodw.groupby('time.month') - iodw.groupby('time.month').mean('time', skipna=True)
            iode = iode.groupby('time.month') - iode.groupby('time.month').mean('time', skipna=True)
            bwa = bwa.groupby('time.month') - bwa.groupby('time.month').mean('time', skipna=True)

    # ----------------------------------------------------------------------------------#
    # Detrend
    iodw_trend = np.polyfit(range(0, len(iodw)), iodw, deg=1)
    iodw = iodw - (iodw_trend[0] * range(0, len(iodw)) + iodw_trend[1])
    # ----------------------------------------------------------------------------------#
    iode_trend = np.polyfit(range(0, len(iode)), iode, deg=1)
    iode = iode - (iode_trend[0] * range(0, len(iode)) + iode_trend[1])
    # ----------------------------------------------------------------------------------#
    bwa_trend = np.polyfit(range(0, len(bwa)), bwa, deg=1)
    bwa = bwa - (bwa_trend[0] * range(0, len(bwa)) + bwa_trend[1])
    # ----------------------------------------------------------------------------------#

    # 3-Month running mean
    iodw_filtered = np.convolve(iodw, np.ones((3,)) / 3, mode='same')
    iode_filtered = np.convolve(iode, np.ones((3,)) / 3, mode='same')
    bwa_filtered = np.convolve(bwa, np.ones((3,)) / 3, mode='same')

    # Common preprocessing, for DMIs other than SY2003a
    iode_3rm = iode_filtered
    iodw_3rm = iodw_filtered

    #################################### follow SY2003a #######################################

    # power spectrum
    # aux = FFT2(iodw_3rm, maxVar=20, maxA=15).sort_values('Variance', ascending=False)
    # aux2 = FFT2(iode_3rm, maxVar=20, maxA=15).sort_values('Variance', ascending=False)

    # filtering harmonic
    if filter_harmonic:
        if filter_all_harmonic:
            for harmonic in range(15):
                iodw_filtered = WaveFilter(iodw_filtered, harmonic)
                iode_filtered = WaveFilter(iode_filtered, harmonic)
            else:
                for harmonic in harmonics:
                    iodw_filtered = WaveFilter(iodw_filtered, harmonic)
                    iode_filtered = WaveFilter(iode_filtered, harmonic)

    ## max corr. lag +3 in IODW
    ## max corr. lag +6 in IODE

    # ----------------------------------------------------------------------------------#
    # ENSO influence
    # pre processing same as before
    if filter_bwa:
        ninio3 = sst.sel(lat=slice(5.0, -5.0), lon=slice(210, 270),
                         time=slice(start_year + '-01-01', end_year + '-12-31'))
        ninio3 = ninio3.sst.mean(['lon', 'lat'], skipna=True)

        if movinganomaly:
            ninio3 = MovingBasePeriodAnomaly(ninio3)
        else:
            if change_baseline:
                ninio3 = ninio3.groupby('time.month') - \
                         ninio3.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby(
                             'time.month').mean(
                             'time')

            else:

                ninio3 = ninio3.groupby('time.month') - ninio3.groupby('time.month').mean('time', skipna=True)

            trend = np.polyfit(range(0, len(ninio3)), ninio3, deg=1)
            ninio3 = ninio3 - (trend[0] * range(0, len(ninio3)) +trend[1])

        # 3-month running mean
        ninio3_filtered = np.convolve(ninio3, np.ones((3,)) / 3, mode='same')

        # ----------------------------------------------------------------------------------#
        # removing BWA effect
        # lag de maxima corr coincide para las dos bases de datos.
        lag = 3
        x = pd.DataFrame({'bwa': bwa_filtered[lag:], 'ninio3': ninio3_filtered[:-lag]})
        result = sm.ols(formula='bwa~ninio3', data=x).fit()
        recta = result.params[1] * ninio3_filtered + result.params[0]
        iodw_f = iodw_filtered - recta

        lag = 6
        x = pd.DataFrame({'bwa': bwa_filtered[lag:], 'ninio3': ninio3_filtered[:-lag]})
        result = sm.ols(formula='bwa~ninio3', data=x).fit()
        recta = result.params[1] * ninio3_filtered + result.params[0]
        iode_f = iode_filtered - recta
        print('BWA filtrado')
    else:
        iodw_f = iodw_filtered
        iode_f = iode_filtered
        print('BWA no filtrado')
    # ----------------------------------------------------------------------------------#

    # END processing
    if movinganomaly:
        iodw_3rm = xr.DataArray(iodw_3rm, coords=[iodw.time.values], dims=['time'])
        iode_3rm = xr.DataArray(iode_3rm, coords=[iodw.time.values], dims=['time'])

        iodw_f = xr.DataArray(iodw_f, coords=[iodw.time.values], dims=['time'])
        iode_f = xr.DataArray(iode_f, coords=[iodw.time.values], dims=['time'])
        start_year = '1920'
    else:
        iodw_3rm = xr.DataArray(iodw_3rm, coords=[iodw2.time.values], dims=['time'])
        iode_3rm = xr.DataArray(iode_3rm, coords=[iodw2.time.values], dims=['time'])

        iodw_f = xr.DataArray(iodw_f, coords=[iodw2.time.values], dims=['time'])
        iode_f = xr.DataArray(iode_f, coords=[iodw2.time.values], dims=['time'])

    ####################################### compute DMI #######################################

    dmi_sy_full, dmi_raw = DMIndex(iodw_f, iode_f,
                                   sst_anom_sd=sst_anom_sd,
                                   opposite_signs_criteria=opposite_signs_criteria)

    return dmi_sy_full, dmi_raw, (iodw_f-iode_f)#, iodw_f - iode_f, iodw_f, iode_f

def DMI2(end_per=1920, start_per=2020, filter_harmonic=True, filter_bwa=False,
         sst_anom_sd=True, opposite_signs_criteria=True):

    # argumentos fijos ------------------------------------------------------------------------------------------------#
    movinganomaly = False
    change_baseline = False
    start_year2 = '6666'
    end_year2 = end_per
    #------------------------------------------------------------------------------------------------------------------#
    western_io = slice(50, 70)  # definicion tradicional
    start_per = str(start_per)
    end_per = str(end_per)

    start_year = start_per
    end_year = end_per
    ####################################################################################################################
    # DATA - ERSSTv5 --------------------------------------------------------------------------------------------------#
    sst = xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc")

    # Pre-processing --------------------------------------------------------------------------------------------------#
    iodw = sst.sel(lat=slice(10.0, -10.0), lon=western_io,
                       time=slice(start_year + '-01-01', end_year + '-12-31'))
    iodw = iodw.sst.mean(['lon', 'lat'], skipna=True)
    # -----------------------------------------------------------------------------------------------------------------#
    iode = sst.sel(lat=slice(0, -10.0), lon=slice(90, 110),
                   time=slice(start_year + '-01-01', end_year + '-12-31'))
    iode = iode.sst.mean(['lon', 'lat'], skipna=True)
    # -----------------------------------------------------------------------------------------------------------------#

    if movinganomaly:
        iodw = MovingBasePeriodAnomaly(iodw)
        iode = MovingBasePeriodAnomaly(iode)
    else:
        if change_baseline:
            iodw = iodw.groupby('time.month') - \
                   iodw.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                       'time')

            iode = iode.groupby('time.month') - \
                   iode.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')).groupby('time.month').mean(
                       'time')

            print('baseline: ' + str(start_year2) + ' - ' + str(end_year2))
        else:
            print('baseline: All period')
            iodw = iodw.groupby('time.month') - iodw.groupby('time.month').mean('time', skipna=True)
            iode = iode.groupby('time.month') - iode.groupby('time.month').mean('time', skipna=True)

    # Detrend ---------------------------------------------------------------------------------------------------------#
    iodw_trend = np.polyfit(range(0, len(iodw)), iodw, deg=1)
    iodw = iodw - (iodw_trend[0] * range(0, len(iodw)) + iodw_trend[1])
    #------------------------------------------------------------------------------------------------------------------#
    iode_trend = np.polyfit(range(0, len(iode)), iode, deg=1)
    iode = iode - (iode_trend[0] * range(0, len(iode)) + iode_trend[1])
    #------------------------------------------------------------------------------------------------------------------#
    # 3-Month running mean --------------------------------------------------------------------------------------------#
    iodw_filtered = np.convolve(iodw, np.ones((3,)) / 3, mode='same')
    iode_filtered = np.convolve(iode, np.ones((3,)) / 3, mode='same')

    # Filtering Harmonic ----------------------------------------------------------------------------------------------#
    if filter_harmonic:
        for harmonic in range(15):
            iodw_filtered = WaveFilter(iodw_filtered, harmonic)
            iode_filtered = WaveFilter(iode_filtered, harmonic)

    # Filter BWA #######################################################################################################
    if filter_bwa:
        bwa = sst.sel(lat=slice(20.0, -20.0), lon=slice(40, 110),
                      time=slice(start_year + '-01-01', end_year + '-12-31'))
        bwa = bwa.sst.mean(['lon', 'lat'], skipna=True)

        if movinganomaly:
            bwa = MovingBasePeriodAnomaly(bwa)
        else:
            bwa = bwa.groupby('time.month') - \
                  bwa.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31')). \
                      groupby('time.month').mean('time')

        # Detrend -----------------------------------------̣̣------------------------------------------------------------#
        bwa_trend = np.polyfit(range(0, len(bwa)), bwa, deg=1)
        bwa = bwa - (bwa_trend[0] * range(0, len(bwa)) + bwa_trend[1])
        bwa_filtered = np.convolve(bwa, np.ones((3,)) / 3, mode='same')

        ninio3 = sst.sel(lat=slice(5.0, -5.0), lon=slice(210, 270),
                         time=slice(start_year + '-01-01', end_year + '-12-31'))
        ninio3 = ninio3.sst.mean(['lon', 'lat'], skipna=True)

        if movinganomaly:
            ninio3 = MovingBasePeriodAnomaly(ninio3)
        else:
            if change_baseline:
                ninio3 = ninio3.groupby('time.month') - \
                         ninio3.sel(time=slice(start_year2 + '-01-01', end_year2 + '-12-31'))\
                             .groupby('time.month').mean('time')
            else:
                ninio3 = ninio3.groupby('time.month') - ninio3.groupby('time.month').mean('time', skipna=True)

            trend = np.polyfit(range(0, len(ninio3)), ninio3, deg=1)
            ninio3 = ninio3 - (trend[0] * range(0, len(ninio3)) + trend[1])

        # 3-month running mean
        ninio3_filtered = np.convolve(ninio3, np.ones((3,)) / 3, mode='same')

        # -------------------------------------------------------------------------------------------------------------#
        # removing BWA effect
        # lag de maxima corr coincide para las dos bases de datos.
        lag = 3
        x = pd.DataFrame({'bwa': bwa_filtered[lag:], 'ninio3': ninio3_filtered[:-lag]})
        result = sm.ols(formula='bwa~ninio3', data=x).fit()
        recta = result.params[1] * ninio3_filtered + result.params[0]
        iodw_f = iodw_filtered - recta

        lag = 6
        x = pd.DataFrame({'bwa': bwa_filtered[lag:], 'ninio3': ninio3_filtered[:-lag]})
        result = sm.ols(formula='bwa~ninio3', data=x).fit()
        recta = result.params[1] * ninio3_filtered + result.params[0]
        iode_f = iode_filtered - recta
        print('BWA filtrado')
    else:
        iodw_f = iodw_filtered
        iode_f = iode_filtered

    ####################################################################################################################
    # END processing --------------------------------------------------------------------------------------------------#
    iodw_f = xr.DataArray(iodw_f, coords=[iodw.time.values], dims=['time'])
    iode_f = xr.DataArray(iode_f, coords=[iodw.time.values], dims=['time'])

    # Compute DMI ######################################################################################################
    dmi_sy_full, dmi_raw = DMIndex(iodw_f, iode_f,
                                   sst_anom_sd=sst_anom_sd,
                                   opposite_signs_criteria=opposite_signs_criteria)
    return dmi_sy_full, dmi_raw, (iodw_f - iode_f)
    ####################################################################################################################

def PlotEnso_Iod(dmi, ninio, title, fig_name = 'fig_enso_iod', out_dir=out_dir, save=False):
    from numpy import ma
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = plt.scatter(x=dmi, y=ninio, marker='o', s=20, edgecolor='black', color='gray')

    plt.ylim((-4, 4));
    plt.xlim((-4, 4))
    plt.axhspan(-0.31, 0.31, alpha=0.2, color='black', zorder=0)
    plt.axvspan(-0.5, 0.5, alpha=0.2, color='black', zorder=0)
    # ax.grid(True)
    fig.set_size_inches(6, 6)
    plt.xlabel('IOD')
    plt.ylabel('Niño 3.4')

    plt.text(-3.8, 3.4, '    EN/IOD-', dict(size=10))
    plt.text(-.1, 3.4, 'EN', dict(size=10))
    plt.text(+2.6, 3.4, ' EN/IOD+', dict(size=10))
    plt.text(+2.6, -.1, 'IOD+', dict(size=10))
    plt.text(+2.3, -3.4, '    LN/IOD+', dict(size=10))
    plt.text(-.1, -3.4, 'LN', dict(size=10))
    plt.text(-3.8, -3.4, ' LN/IOD-', dict(size=10))
    plt.text(-3.8, -.1, 'IOD-', dict(size=10))
    plt.title(title)
    if save:
        plt.savefig(out_dir + 'ENSO_IOD'+ fig_name + '.jpg')
    else:
        plt.show()

def SelectYears(df, name_var, main_month=1, full_season=False):

    if full_season:
        print('Full Season JJASON')
        aux = pd.DataFrame({'Ind': df.where(df.Mes.isin([7, 8, 9, 10, 11]))[name_var],
                            'Años': df.where(df.Mes.isin([7, 8, 9, 10, 11]))['Años'],
                            'Mes': df.where(df.Mes.isin([7, 8, 9, 10, 11]))['Mes']})
        mmin, mmax = 6, 11

    else:
        aux = pd.DataFrame({'Ind': df.where(df.Mes.isin([main_month]))[name_var],
                            'Años': df.where(df.Mes.isin([main_month]))['Años'],
                            'Mes': df.where(df.Mes.isin([main_month]))['Mes']})
        mmin, mmax = main_month - 1, main_month + 1

        if main_month == 1:
            mmin, mmax = 12, 2
        elif main_month == 12:
            mmin, mmax = 11, 1

    return aux.dropna(), mmin, mmax

def ClassifierEvents(df, full_season=False):
    if full_season:
        print('full season')
        df_pos = set(df.Años.values[np.where(df['Ind'] > 0)])
        df_neg = set(df.Años.values[np.where(df['Ind'] < 0)])
    else:
        df_pos = df.Años.values[np.where(df['Ind'] > 0)]
        df_neg = df.Años.values[np.where(df['Ind'] < 0)]

    return df_pos, df_neg

def NeutralEvents(df, mmin, mmax, start=1920, end = 2020, double=False, df2=None, var_original=None):

    x = np.arange(start, end + 1, 1)

    start = str(start)
    end = str(end)

    mask = np.in1d(x, df.Años.values, invert=True)
    if mmax ==1: #NDJ
        print("NDJ Special")
        neutro = var_original.sel(time=var_original.time.dt.year.isin(x[mask]))
        neutro_1 = var_original.sel(time=var_original.time.dt.year.isin(x[mask]+1))
        if double:
            mask = np.in1d(x, df2.Años.values, invert=True)
            neutro = neutro.sel(time=neutro.time.dt.year.isin(x[mask]))
            neutro_1 = neutro_1.sel(time=neutro_1.time.dt.year.isin(x[mask]))

        neutro = neutro.sel(time=is_months(month=neutro['time.month'], mmin=11, mmax=12))
        neutro_1 = neutro_1.sel(time=neutro_1.time.dt.month.isin(1))
        neutro = xr.merge([neutro, neutro_1])
        neutro = neutro.mean(['time'], skipna=True)

    elif mmin == 12: #DJF
        print("DJF Special")
        neutro = var_original.sel(time=var_original.time.dt.year.isin(x[mask]))
        neutro_1 = var_original.sel(time=var_original.time.dt.year.isin(x[mask]-1))
        if double:
            mask = np.in1d(x, df2.Años.values, invert=True)
            neutro = neutro.sel(time=neutro.time.dt.year.isin(x[mask]))
            neutro_1 = neutro_1.sel(time=neutro_1.time.dt.year.isin(x[mask]))

        neutro = neutro.sel(time=is_months(month=neutro['time.month'], mmin=1, mmax=2))
        neutro_1 = neutro_1.sel(time=neutro_1.time.dt.month.isin(12))
        neutro = xr.merge([neutro, neutro_1])
        neutro = neutro.mean(['time'], skipna=True)

    else:
        mask = np.in1d(x, df.Años.values, invert=True)
        neutro = var_original.sel(time=var_original.time.dt.year.isin(x[mask]))
        if double:
            mask = np.in1d(x, df2.Años.values, invert=True)
            neutro = neutro.sel(time=neutro.time.dt.year.isin(x[mask]))
            neutro_years = list(set(neutro.time.dt.year.values))
        neutro = neutro.sel(time=is_months(month=neutro['time.month'], mmin=mmin, mmax=mmax))
        neutro = neutro.mean(['time'], skipna=True)

    return neutro, neutro_years
########################################################################################################################
# Varias ###############################################################################################################
def is_months(month, mmin, mmax):
    return (month >= mmin) & (month <= mmax)
def WaveFilter(serie, harmonic):

    import numpy as np

    sum = 0
    sam = 0
    N = np.size(serie)

    sum = 0
    sam = 0

    for j in range(N):
        sum = sum + serie[j] * np.sin(harmonic * 2 * np.pi * j / N)
        sam = sam + serie[j] * np.cos(harmonic * 2 * np.pi * j / N)

    A = 2*sum/N
    B = 2*sam/N

    xs = np.zeros(N)

    for j in range(N):
        xs[j] = A * np.sin(2 * np.pi * harmonic * j / N) + B * np.cos(2 * np.pi * harmonic * j / N)

    fil = serie - xs
    return(fil)

def Composite(original_data, index_pos, index_neg, mmin, mmax):
    comp_field_pos=0
    comp_field_neg=0

    if len(index_pos) != 0:
        if mmax == 1:
            print('NDJ Special')
            comp_field_pos = original_data.sel(time=original_data.time.dt.year.isin(index_pos))
            comp_field_pos_1 = original_data.sel(time=original_data.time.dt.year.isin(index_pos+1))

            comp_field_pos = comp_field_pos.sel(
                time=is_months(month=comp_field_pos['time.month'], mmin=11, mmax=12))
            comp_field_pos_1 = comp_field_pos_1.sel(time=comp_field_pos_1.time.dt.month.isin(1))

            comp_field_pos = xr.merge([comp_field_pos, comp_field_pos_1])
            if len(comp_field_pos.time) != 0:
                comp_field_pos = comp_field_pos.mean(['time'], skipna=True)
            else:
                comp_field_pos = comp_field_pos.drop_dims(['time'])

        elif mmin == 12:
            print('DJF Special')
            comp_field_pos = original_data.sel(time=original_data.time.dt.year.isin(index_pos))
            comp_field_pos_1 = original_data.sel(time=original_data.time.dt.year.isin(index_pos - 1))

            comp_field_pos = comp_field_pos.sel(
                time=is_months(month=comp_field_pos['time.month'], mmin=1, mmax=2))
            comp_field_pos_1 = comp_field_pos_1.sel(time=comp_field_pos_1.time.dt.month.isin(2))

            comp_field_pos = xr.merge([comp_field_pos, comp_field_pos_1])
            if len(comp_field_pos.time) != 0:
                comp_field_pos = comp_field_pos.mean(['time'], skipna=True)
            else:
                comp_field_pos = comp_field_pos.drop_dims(['time'])

        else:
            comp_field_pos = original_data.sel(time=original_data.time.dt.year.isin([index_pos]))
            comp_field_pos = comp_field_pos.sel(
                time=is_months(month=comp_field_pos['time.month'], mmin=mmin, mmax=mmax))
            if len(comp_field_pos.time) != 0:
                comp_field_pos = comp_field_pos.mean(['time'], skipna=True)
            else:
                comp_field_pos = comp_field_pos.drop_dims(['time'])


    if len(index_neg) != 0:
        if mmax == 1:
            print('NDJ Special')
            comp_field_neg = original_data.sel(time=original_data.time.dt.year.isin(index_neg))
            comp_field_neg_1 = original_data.sel(time=original_data.time.dt.year.isin(index_neg + 1))

            comp_field_neg = comp_field_neg.sel(
                time=is_months(month=comp_field_neg['time.month'], mmin=11, mmax=12))
            comp_field_neg_1 = comp_field_neg_1.sel(time=comp_field_neg_1.time.dt.month.isin(1))

            comp_field_neg = xr.merge([comp_field_neg, comp_field_neg_1])
            if (len(comp_field_neg.time) != 0):
                comp_field_neg = comp_field_neg.mean(['time'], skipna=True)
            else:
                comp_field_neg = comp_field_neg.drop_dmis(['time'])

        elif mmin == 12:
            print('DJF Special')
            comp_field_neg = original_data.sel(time=original_data.time.dt.year.isin(index_neg))
            comp_field_neg_1 = original_data.sel(time=original_data.time.dt.year.isin(index_neg - 1))

            comp_field_neg = comp_field_neg.sel(
                time=is_months(month=comp_field_neg['time.month'], mmin=1, mmax=2))
            comp_field_neg_1 = comp_field_neg_1.sel(time=comp_field_neg_1.time.dt.month.isin(2))

            comp_field_neg = xr.merge([comp_field_neg, comp_field_neg_1])
            if len(comp_field_neg.time) != 0:
                comp_field_neg = comp_field_neg.mean(['time'], skipna=True)
            else:
                comp_field_neg = comp_field_neg.drop_dims(['time'])

        else:
            comp_field_neg = original_data.sel(time=original_data.time.dt.year.isin([index_neg]))
            comp_field_neg = comp_field_neg.sel(time=is_months(month=comp_field_neg['time.month'],
                                                               mmin=mmin, mmax=mmax))
            if len(comp_field_neg.time) != 0:
                comp_field_neg = comp_field_neg.mean(['time'], skipna=True)
            else:
                comp_field_neg = comp_field_neg.drop_dims(['time'])

    return comp_field_pos, comp_field_neg

def MultipleComposite(var, n34, dmi, season,start = 1920, full_season=False, compute_composite=False):

    seasons = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
               'JJA','JAS', 'ASO', 'SON', 'OND', 'NDJ']


    def check(x):
        if x is None:
            x = [0]
            return x
        else:
            if len(x) == 0:
                x = [0]
                return x
        return x

    if full_season:
        main_month_name = 'JJASON'
        main_month = None
    else:
        main_month, main_month_name = len(seasons[:season]) + 1, seasons[season]

    print(main_month_name)

    N34, N34_mmin, N34_mmax = SelectYears(df=n34, name_var='N34',
                                          main_month=main_month, full_season=full_season)
    DMI, DMI_mmin, DMI_mmax = SelectYears(df=dmi, name_var='DMI',
                                          main_month=main_month, full_season=full_season)
    DMI_sim_pos = [0,0]
    DMI_sim_neg = [0,0]
    DMI_un_pos = [0,0]
    DMI_un_neg = [0,0]
    DMI_pos = [0,0]
    DMI_neg = [0,0]
    N34_sim_pos = [0,0]
    N34_sim_neg = [0,0]
    N34_un_pos = [0,0]
    N34_un_neg = [0,0]
    N34_pos = [0,0]
    N34_neg = [0,0]
    All_neutral = [0, 0]

    if (len(DMI) != 0) & (len(N34) != 0):
        # All events
        DMI_pos, DMI_neg = ClassifierEvents(DMI, full_season=full_season)
        N34_pos, N34_neg = ClassifierEvents(N34, full_season=full_season)

        # both neutral, DMI and N34
        if compute_composite:
            All_neutral = NeutralEvents(df=DMI, mmin=DMI_mmin, mmax=DMI_mmax, start=start,
                                        df2=N34, double=True, var_original=var)[0]

        else:
            All_neutral = NeutralEvents(df=DMI, mmin=DMI_mmin, mmax=DMI_mmax, start=start,
                                        df2=N34, double=True, var_original=var)[1]


        # Simultaneous events
        sim_events = np.intersect1d(N34.Años.values, DMI.Años.values)

        try:
            # Simultaneos events
            DMI_sim = DMI.where(DMI.Años.isin(sim_events)).dropna()
            N34_sim = N34.where(N34.Años.isin(sim_events)).dropna()
            DMI_sim_pos_aux, DMI_sim_neg_aux = ClassifierEvents(DMI_sim)
            N34_sim_pos_aux, N34_sim_neg_aux = ClassifierEvents(N34_sim)


            # Existen eventos simultaneos de signo opuesto?
            # cuales?
            sim_pos = np.intersect1d(DMI_sim_pos_aux, N34_sim_pos_aux)
            sim_pos2 = np.intersect1d(sim_pos, DMI_sim_pos_aux)
            DMI_sim_pos = sim_pos2

            sim_neg = np.intersect1d(DMI_sim_neg_aux, N34_sim_neg_aux)
            sim_neg2 = np.intersect1d(DMI_sim_neg_aux, sim_neg)
            DMI_sim_neg = sim_neg2


            if (len(sim_events) != (len(sim_pos) + len(sim_neg))):
                dmi_pos_n34_neg = np.intersect1d(DMI_sim_pos_aux, N34_sim_neg_aux)
                dmi_neg_n34_pos = np.intersect1d(DMI_sim_neg_aux, N34_sim_pos_aux)
            else:
                dmi_pos_n34_neg = None
                dmi_neg_n34_pos = None

            # Unique events
            DMI_un = DMI.where(-DMI.Años.isin(sim_events)).dropna()
            N34_un = N34.where(-N34.Años.isin(sim_events)).dropna()

            DMI_un_pos, DMI_un_neg = ClassifierEvents(DMI_un)
            N34_un_pos, N34_un_neg = ClassifierEvents(N34_un)

            if compute_composite:
                print('Making composites...')
                # ------------------------------------ SIMULTANEUS ---------------------------------------------#
                DMI_sim = Composite(original_data=var, index_pos=DMI_sim_pos, index_neg=DMI_sim_neg,
                                    mmin=DMI_mmin, mmax=DMI_mmax)

                # ------------------------------------ UNIQUES -------------------------------------------------#
                DMI_un = Composite(original_data=var, index_pos=DMI_un_pos, index_neg=DMI_un_neg,
                                   mmin=DMI_mmin, mmax=DMI_mmax)

                N34_un = Composite(original_data=var, index_pos=N34_un_pos, index_neg=N34_un_neg,
                                   mmin=N34_mmin, mmax=N34_mmax)
            else:
                print('Only dates, no composites')
                DMI_sim = None
                DMI_un = None
                N34_un = None

        except:
            DMI_sim = None
            DMI_un = None
            N34_un = None
            DMI_sim_pos = None
            DMI_sim_neg = None
            DMI_un_pos = None
            DMI_un_neg = None
            print('Only uniques events[3][4]')

        if compute_composite:
            # ------------------------------------ ALL ---------------------------------------------#
            dmi_comp = Composite(original_data=var, index_pos=list(DMI_pos), index_neg=list(DMI_neg),
                                 mmin=DMI_mmin, mmax=DMI_mmax)
            N34_comp = Composite(original_data=var, index_pos=list(N34_pos), index_neg=list(N34_neg),
                                 mmin=N34_mmin, mmax=N34_mmax)
        else:
            dmi_comp=None
            N34_comp=None

    DMI_sim_pos = check(DMI_sim_pos)
    DMI_sim_neg = check(DMI_sim_neg)
    DMI_un_pos = check(DMI_un_pos)
    DMI_un_neg = check(DMI_un_neg)
    DMI_pos = check(DMI_pos)
    DMI_neg = check(DMI_neg)

    N34_sim_pos = check(N34_sim_pos)
    N34_sim_neg = check(N34_sim_neg)
    N34_un_pos = check(N34_un_pos)
    N34_un_neg = check(N34_un_neg)
    N34_pos = check(N34_pos)
    N34_neg = check(N34_neg)

    DMI_pos_N34_neg = check(dmi_pos_n34_neg)
    DMI_neg_N34_pos = check(dmi_neg_n34_pos)

    All_neutral = check(All_neutral)


    if compute_composite:
        print('test')
        return DMI_sim, DMI_un, N34_un, dmi_comp, N34_comp, All_neutral, DMI_sim_pos, DMI_sim_neg, \
               DMI_un_pos, DMI_un_neg, N34_un_pos, N34_un_neg, DMI_pos, DMI_neg, N34_pos, N34_neg
    else:
        return list(All_neutral),\
               list(set(DMI_sim_pos)), list(set(DMI_sim_neg)),\
               list(set(DMI_un_pos)), list(set(DMI_un_neg)),\
               list(set(N34_un_pos)), list(set(N34_un_neg)),\
               list(DMI_pos), list(DMI_neg), \
               list(N34_pos), list(N34_neg), \
               list(DMI_pos_N34_neg), list(DMI_neg_N34_pos)

def xrFieldTimeDetrend(xrda, dim, deg=1):
    # detrend along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients[0])
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients[0])

    dt = xrda - trend
    return dt

def CompositeSimple(original_data, index, mmin, mmax):
    def is_months(month, mmin, mmax):
        return (month >= mmin) & (month <= mmax)

    if len(index) != 0:
        comp_field = original_data.sel(time=original_data.time.dt.year.isin([index]))
        comp_field = comp_field.sel(
            time=is_months(month=comp_field['time.month'], mmin=mmin, mmax=mmax))
        if len(comp_field.time) != 0:
            comp_field = comp_field.mean(['time'], skipna=True)
        else:  # si sólo hay un año
            comp_field = comp_field.drop_dims(['time'])

        return comp_field
    else:
        print(' len index = 0')


def CaseComp(data, s, mmonth, c, two_variables=False, data2=None, return_neutro_comp=False, nc_date_dir='None'):
    """
    Las fechas se toman del periodo 1920-2020 basados en el DMI y N34 con ERSSTv5
    Cuando se toman los periodos 1920-1949 y 1950_2020 las fechas que no pertencen
    se excluyen de los composites en CompositeSimple()
    """
    mmin = mmonth[0]
    mmax = mmonth[-1]

    aux = xr.open_dataset(nc_date_dir + '1920_2020' + '_' + s + '.nc')
    neutro = aux.Neutral

    try:
        case = aux[c]
        case = case.where(case >= 1940)
        aux.close()

        case_num = len(case.values[np.where(~np.isnan(case.values))])
        case_num2 = case.values[np.where(~np.isnan(case.values))]

        neutro_comp = CompositeSimple(original_data=data, index=neutro, mmin=mmin, mmax=mmax)
        data_comp = CompositeSimple(original_data=data, index=case, mmin=mmin, mmax=mmax)

        comp = data_comp - neutro_comp

        if two_variables:
            neutro_comp2 = CompositeSimple(original_data=data2, index=neutro, mmin=mmin, mmax=mmax)
            data_comp2 = CompositeSimple(original_data=data2, index=case, mmin=mmin, mmax=mmax)

            comp2 = data_comp2 - neutro_comp2
        else:
            comp2 = None
    except:
        print('Error en ' + s + c)

    if two_variables:
        if return_neutro_comp:
            return comp, case_num, comp2, neutro_comp, neutro_comp2
        else:
            return comp, case_num, comp2
    else:
        if return_neutro_comp:
            return comp, case_num, neutro_comp
        else:
            return comp, case_num


def SelectCase(original_data, index, mmin, mmax):
    def is_months(month, mmin, mmax):
        return (month >= mmin) & (month <= mmax)

    if len(index) != 0:
        comp_field = original_data.sel(
            time=original_data.time.dt.year.isin([index]))
        comp_field = comp_field.sel(
            time=is_months(
                month=comp_field['time.month'], mmin=mmin, mmax=mmax))

        return comp_field
    else:
        print('len index = 0')

def CaseSNR(data, s, mmonth, c, nc_date_dir='None'):
    mmin = mmonth[0]
    mmax = mmonth[-1]

    aux = xr.open_dataset(nc_date_dir + '1920_2020' + '_' + s + '.nc')
    neutro = aux.Neutral

    try:
        case = aux[c]
        case = case.where(case >= 1940)
        aux.close()

        case_num = len(case.values[np.where(~np.isnan(case.values))])

        neutro_comp = SelectCase(original_data=data, index=neutro,
                                     mmin=mmin, mmax=mmax)
        data_comp = SelectCase(original_data=data, index=case,
                                 mmin=mmin, mmax=mmax)

        comp = data_comp.mean(['time'], skipna=True) -\
               neutro_comp.mean(['time'], skipna=True)

        spread = data - comp
        spread = spread.std(['time'], skipna=True)

        snr = comp / spread

        return snr, case_num
    except:
        print('Error en ' + s + ' ' + c)



def ChangeLons(data, lon_name='lon'):
    data['_longitude_adjusted'] = xr.where(
        data[lon_name] < 0,
        data[lon_name] + 360,
        data[lon_name])

    data = (
        data
            .swap_dims({lon_name: '_longitude_adjusted'})
            .sel(**{'_longitude_adjusted': sorted(data._longitude_adjusted)})
            .drop(lon_name))

    data = data.rename({'_longitude_adjusted': 'lon'})

    return data

def MakeMask(DataArray, dataname='mask'):
    import regionmask
    mask=regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(DataArray)
    mask = xr.where(np.isnan(mask), mask, 1)
    mask = mask.to_dataset(name=dataname)
    return mask

def OpenDatasets(name, interp=False):
    pwd_datos = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_obs_viejo/'
    def ChangeLons(data, lon_name='lon'):
        data['_longitude_adjusted'] = xr.where(
            data[lon_name] < 0,
            data[lon_name] + 360,
            data[lon_name])

        data = (
            data
                .swap_dims({lon_name: '_longitude_adjusted'})
                .sel(**{'_longitude_adjusted': sorted(data._longitude_adjusted)})
                .drop(lon_name))

        data = data.rename({'_longitude_adjusted': 'lon'})

        return data


    def xrFieldTimeDetrend(xrda, dim, deg=1):
        # detrend along a single dimension
        aux = xrda.polyfit(dim=dim, deg=deg)
        try:
            trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients[0])
        except:
            trend = xr.polyval(xrda[dim], aux.polyfit_coefficients[0])

        dt = xrda - trend
        return dt

    aux = xr.open_dataset(pwd_datos + 'pp_20CR-V3.nc')
    pp_20cr = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))

    aux = xr.open_dataset(pwd_datos + 't_20CR-V3.nc')
    t_20cr = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))

    aux = xr.open_dataset(pwd_datos + 't_cru.nc')
    t_cru = ChangeLons(aux)

    ### Precipitation ###
    if name == 'pp_20CR-V3':
        # NOAA20CR-V3
        aux = xr.open_dataset(pwd_datos + 'pp_20CR-V3.nc')
        pp_20cr = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))
        pp_20cr = pp_20cr.rename({'prate': 'var'})
        pp_20cr = pp_20cr.__mul__(86400 * (365 / 12))  # kg/m2/s -> mm/month
        pp_20cr = pp_20cr.drop('time_bnds')
        pp_20cr = xrFieldTimeDetrend(pp_20cr, 'time')

        return pp_20cr
    elif name == 'pp_gpcc':
        # GPCC2018
        aux = xr.open_dataset(pwd_datos + 'pp_gpcc.nc')
        # interpolado igual que 20cr, los dos son 1x1 pero con distinta grilla
        pp_gpcc = aux.sel(lon=slice(270, 330), lat=slice(20, -50))
        if interp:
            pp_gpcc = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_gpcc = pp_gpcc.rename({'precip': 'var'})
        pp_gpcc = xrFieldTimeDetrend(pp_gpcc, 'time')

        return pp_gpcc
    elif name == 'pp_PREC':
        # PREC
        aux = xr.open_dataset(pwd_datos + 'pp_PREC.nc')
        pp_prec = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
        if interp:
            pp_prec = pp_prec.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_prec = pp_prec.rename({'precip': 'var'})
        pp_prec = pp_prec.__mul__(365 / 12)  # mm/day -> mm/month
        pp_prec = xrFieldTimeDetrend(pp_prec, 'time')

        return pp_prec
    elif name == 'pp_chirps':
        # CHIRPS
        aux = xr.open_dataset(pwd_datos + 'pp_chirps.nc')
        aux = ChangeLons(aux, 'longitude')
        aux = aux.rename({'precip': 'var', 'latitude': 'lat'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))
        if interp:
            aux = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_ch = aux
        pp_ch = xrFieldTimeDetrend(pp_ch, 'time')

        return pp_ch
    elif name == 'pp_CMAP':
        # CMAP
        aux = xr.open_dataset(pwd_datos + 'pp_CMAP.nc')
        aux = aux.rename({'precip': 'var'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -50))
        if interp:
            pp_cmap = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        pp_cmap = aux.__mul__(365 / 12)  # mm/day -> mm/month
        pp_cmap = xrFieldTimeDetrend(pp_cmap, 'time')

        return pp_cmap
    elif name == 'pp_gpcp':
        # GPCP2.3
        aux = xr.open_dataset(pwd_datos + 'pp_gpcp.nc')
        aux = aux.rename({'precip': 'var'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(-50, 20))
        if interp:
            pp_gpcp = aux.interp(lon=pp_20cr.lon.values, lat=pp_20cr.lat.values)
        aux = aux.drop('lat_bnds')
        aux = aux.drop('lon_bnds')
        aux = aux.drop('time_bnds')
        pp_gpcp = aux.__mul__(365 / 12)  # mm/day -> mm/month
        pp_gpcp = xrFieldTimeDetrend(pp_gpcp, 'time')

        return pp_gpcp
    elif name == 't_20CR-V3':
        # 20CR-v3
        aux = xr.open_dataset(pwd_datos + 't_20CR-V3.nc')
        t_20cr = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        t_20cr = t_20cr.rename({'air': 'var'})
        t_20cr = t_20cr - 273
        t_20cr = t_20cr.drop('time_bnds')
        t_20cr = xrFieldTimeDetrend(t_20cr, 'time')
        return t_20cr

    elif name == 't_cru':
        # CRU
        aux = xr.open_dataset(pwd_datos + 't_cru.nc')
        t_cru = ChangeLons(aux)
        t_cru = t_cru.sel(lon=slice(270, 330), lat=slice(-60, 20),
                          time=slice('1920-01-01', '2020-12-31'))
        # interpolado a 1x1
        if interp:
            t_cru = t_cru.interp(lat=t_20cr.lat.values, lon=t_20cr.lon.values)
        t_cru = t_cru.rename({'tmp': 'var'})
        t_cru = t_cru.drop('stn')
        t_cru = xrFieldTimeDetrend(t_cru, 'time')
        return t_cru
    elif name == 't_BEIC': # que mierda pasaAAA!
        # Berkeley Earth etc
        aux = xr.open_dataset(pwd_datos + 't_BEIC.nc')
        aux = aux.rename({'longitude': 'lon', 'latitude': 'lat', 'temperature': 'var'})
        aux = ChangeLons(aux)
        aux = aux.sel(lon=slice(270, 330), lat=slice(-60, 20), time=slice(1920, 2020.999))
        if interp:
            aux = aux.interp(lat=t_20cr.lat.values, lon=t_20cr.lon.values)

        t_cru = t_cru.sel(time=slice('1920-01-01', '2020-12-31'))
        aux['time'] = t_cru.time.values
        aux['month_number'] = t_cru.time.values[-12:]
        t_beic_clim_months = aux.climatology
        t_beic = aux['var']
        # reconstruyendo?¿
        t_beic = t_beic.groupby('time.month') + t_beic_clim_months.groupby('month_number.month').mean()
        t_beic = t_beic.drop('month')
        t_beic = xr.Dataset(data_vars={'var': t_beic})
        t_beic = xrFieldTimeDetrend(t_beic, 'time')
        return t_beic

    elif name == 't_ghcn_cams':
        # GHCN

        aux = xr.open_dataset(pwd_datos + 't_ghcn_cams.nc')
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_ghcn = aux.rename({'air': 'var'})
        t_ghcn = t_ghcn - 273
        t_ghcn = xrFieldTimeDetrend(t_ghcn, 'time')
        return t_ghcn

    elif name == 't_hadcrut':
        # HadCRUT
        aux = xr.open_dataset(pwd_datos + 't_hadcrut_anom.nc')
        aux = ChangeLons(aux, 'longitude')
        aux = aux.sel(lon=slice(270, 330), latitude=slice(-60, 20))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, latitude=t_20cr.lat.values)
        aux = aux.rename({'tas_mean': 'var', 'latitude': 'lat'})
        t_had = aux.sel(time=slice('1920-01-01', '2020-12-31'))

        aux = xr.open_dataset(pwd_datos + 't_hadcrut_mean.nc')
        aux = ChangeLons(aux)
        aux = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_had_clim = aux.sel(lon=slice(270, 330), lat=slice(-60, 20))
        aux = aux.rename({'tem': 'var'})
        aux['time'] = t_cru.time.values[-12:]
        # reconstruyendo?¿
        t_had = t_had.groupby('time.month') + aux.groupby('time.month').mean()
        t_had = t_had.drop('realization')
        t_had = t_had.drop('month')
        t_had = xrFieldTimeDetrend(t_had, 'time')

        return t_had

    elif name == 't_era20c':

        # ERA-20C
        aux = xr.open_dataset(pwd_datos + 't_era20c.nc')
        aux = aux.rename({'t2m': 'var', 'latitude': 'lat', 'longitude': 'lon'})
        aux = aux.sel(lon=slice(270, 330), lat=slice(20, -60))
        if interp:
            aux = aux.interp(lon=t_20cr.lon.values, lat=t_20cr.lat.values)
        t_era20 = aux - 273
        t_era20 = xrFieldTimeDetrend(t_era20, 'time')

        return t_era20
    elif name == 'pp_lieb':
        aux = xr.open_dataset(pwd_datos + 'pp_liebmann.nc')
        aux = aux.sel(time=slice('1985-01-01', '2010-12-31'))
        aux = aux.resample(time='1M', skipna=True).mean()
        aux = ChangeLons(aux, 'lon')
        pp_lieb = aux.sel(lon=slice(275, 330), lat=slice(-50, 20))
        pp_lieb = pp_lieb.__mul__(365 / 12)
        pp_lieb = pp_lieb.drop('count')
        pp_lieb = pp_lieb.rename({'precip': 'var'})
        pp_lieb = xrFieldTimeDetrend(pp_lieb, 'time')
        return pp_lieb


def fix_calendar(ds, timevar='time'):
    """
    agrega los dias a los archivos nc de NMME
    """
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'
    return ds

########################################################################################################################
# WAF ##################################################################################################################
def c_diff(arr, h, dim, cyclic=False):
    # compute derivate of array variable respect to h associated to dim
    # adapted from kuchaale script
    ndim = arr.ndim
    lst = [i for i in range(ndim)]

    lst[dim], lst[0] = lst[0], lst[dim]
    rank = lst
    arr = np.transpose(arr, tuple(rank))

    if ndim == 3:
        shp = (arr.shape[0] - 2, 1, 1)
    elif ndim == 4:
        shp = (arr.shape[0] - 2, 1, 1, 1)

    d_arr = np.copy(arr)
    if not cyclic:
        d_arr[0, ...] = (arr[1, ...] - arr[0, ...]) / (h[1] - h[0])
        d_arr[-1, ...] = (arr[-1, ...] - arr[-2, ...]) / (h[-1] - h[-2])
        d_arr[1:-1, ...] = (arr[2:, ...] - arr[0:-2, ...]) / np.reshape(h[2:] - h[0:-2], shp)

    elif cyclic:
        d_arr[0, ...] = (arr[1, ...] - arr[-1, ...]) / (h[1] - h[-1])
        d_arr[-1, ...] = (arr[0, ...] - arr[-2, ...]) / (h[0] - h[-2])
        d_arr[1:-1, ...] = (arr[2:, ...] - arr[0:-2, ...]) / np.reshape(h[2:] - h[0:-2], shp)

    d_arr = np.transpose(d_arr, tuple(rank))

    return d_arr

def WAF(psiclm, psiaa, lon, lat,reshape=True, variable='var', hpalevel=200):
    #agregar xr=True

    if reshape:
        psiclm=psiclm[variable].values.reshape(1,len(psiclm.lat),len(psiclm.lon))
        psiaa = psiaa[variable].values.reshape(1, len(psiaa.lat), len(psiaa.lon))

    lon=lon.values
    lat=lat.values

    [xxx, nlats, nlons] = psiaa.shape  # get dimensions
    a = 6400000
    coslat = np.cos(lat * np.pi / 180)

    # climatological wind at psi level
    dpsiclmdlon = c_diff(psiclm, lon, 2)
    dpsiclmdlat = c_diff(psiclm, lat, 1)

    uclm = -1 * dpsiclmdlat
    vclm = dpsiclmdlon
    magU = np.sqrt(np.add(np.power(uclm, 2), np.power(vclm, 2)))

    dpsidlon = c_diff(psiaa, lon, 2)
    ddpsidlonlon = c_diff(dpsidlon, lon, 2)
    dpsidlat = c_diff(psiaa, lat, 1)
    ddpsidlatlat = c_diff(dpsidlat, lat, 1)
    ddpsidlatlon = c_diff(dpsidlat, lon, 2)

    termxu = dpsidlon * dpsidlon - psiaa * ddpsidlonlon
    termxv = dpsidlon * dpsidlat - ddpsidlatlon * psiaa
    termyv = dpsidlat * dpsidlat - psiaa * ddpsidlatlat

    # 0.2101 is the scale of p
    if hpalevel==200:
        coef = 0.2101
    elif hpalevel==750:
        coef = 0.74

    coeff1 = np.transpose(np.tile(coslat, (nlons, 1))) * (coef) / (2 * magU)
    # x-component
    px = coeff1 / (a * a * np.transpose(np.tile(coslat, (nlons, 1)))) * (
            uclm * termxu / np.transpose(np.tile(coslat, (nlons, 1))) + (vclm * termxv))
    # y-component
    py = coeff1 / (a * a) * (uclm / np.transpose(np.tile(coslat, (nlons, 1))) * termxv + (vclm * termyv))

    return px, py

def PlotWAFCountours(comp, comp_var, title='Fig', name_fig='Fig',
                     save=False, dpi=200, levels=np.linspace(-1.5, 1.5, 13),
                     contour=False, cmap='RdBu_r', number_events='',
                     waf=False, px=None, py=None, text=True, waf_scale=None, waf_units=None,
                     two_variables = False, comp2=None, step=1,step_waf=12,
                     levels2=np.linspace(-1.5, 1.5, 13), contour0=False, color_map='#4B4B4B',
                     color_arrow='#400004'):

    from numpy import ma
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 3.5), dpi=dpi)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([0, 359, -80, 20], crs=crs_latlon)


    im = ax.contourf(comp.lon[::step], comp.lat[::step], comp_var[::step,::step],
                     levels=levels,transform=crs_latlon, cmap=cmap, extend='both')
    if contour:
        values = ax.contour(comp.lon, comp.lat, comp_var, levels=levels,
                            transform=crs_latlon, colors='k', linewidths=1)
        ax.clabel(values, inline=1, fontsize=5, fmt='%1.1f')

    if contour0:
        values = ax.contour(comp.lon, comp.lat, comp_var, levels=0,
                            transform=crs_latlon, colors='magenta', linewidths=1)
        ax.clabel(values, inline=1.5, fontsize=5, fmt='%1.1f')

    if two_variables:
        print('Plot Two Variables')
        comp_var2 = comp2['var']
        levels_contour2 = levels2.copy()
        if isinstance(levels2, np.ndarray):
            levels_contour2 = levels2[levels2 != 0]
        else:
            levels_contour2.remove(0)
        values2 = ax.contour(comp2.lon, comp2.lat, comp_var2, levels=levels_contour2,
                            transform=crs_latlon, colors='k', linewidths=0.8, alpha=0.8)
        #ax.clabel(values2, inline=1, fontsize=5, fmt='%1.1f')

    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    #ax.add_feature(cartopy.feature.COASTLINE)
    ax.coastlines(color=color_map, linestyle='-', alpha=1, linewidth=0.6)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(np.arange(0, 360, 30), crs=crs_latlon)
    ax.set_yticks(np.arange(-80, 20, 10), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)

    if waf:
        Q60 = np.percentile(np.sqrt(np.add(np.power(px, 2), np.power(py, 2))), 0)
        M = np.sqrt(np.add(np.power(px, 2), np.power(py, 2))) < Q60
        # mask array
        px_mask = ma.array(px, mask=M)
        py_mask = ma.array(py, mask=M)
        # plot vectors
        lons, lats = np.meshgrid(comp.lon.values, comp.lat.values)
        ax.quiver(lons[::step_waf, ::step_waf],
                  lats[::step_waf, ::step_waf],
                  px_mask[0, ::step_waf, ::step_waf],
                  py_mask[0, ::step_waf, ::step_waf], transform=crs_latlon,pivot='tail',
                  width=0.0020,headwidth=4.1, alpha=1, color=color_arrow, scale=waf_scale, scale_units=waf_units)
                  #, scale=1/10)#, width=1.5e-3, headwidth=3.1,  # headwidht (default3)
                  #headlength=2.2)  # (default5))

    plt.title(title, fontsize=10)
    if text:
        plt.figtext(0.5, 0.01, 'Number of events: ' + str(number_events), ha="center", fontsize=10,
                bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})
    #plt.tight_layout()

    if save:
        plt.savefig(name_fig + '.jpg')
        plt.close()
    else:
        plt.show()
########################################################################################################################
# Regression ###########################################################################################################
def LinearReg(xrda, dim, deg=1):
    # liner reg along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg, skipna=True)
    return aux

def LinearReg1_D(dmi, n34):
    import statsmodels.formula.api as smf

    df = pd.DataFrame({'dmi': dmi.values, 'n34': n34.values})

    result = smf.ols(formula='n34~dmi', data=df).fit()
    n34_pred_dmi = result.params[1] * dmi.values + result.params[0]

    result = smf.ols(formula='dmi~n34', data=df).fit()
    dmi_pred_n34 = result.params[1] * n34.values + result.params[0]

    return n34 - n34_pred_dmi, dmi - dmi_pred_n34

def RegWEffect(n34, dmi,data=None, data2=None, m=9,two_variables=False):
    var_reg_n34_2=0
    var_reg_dmi_2=1

    data['time'] = n34
     #print('Full Season')
    # try:
    #     aux = LinearReg(data.groupby('month')[m], 'time')
    # except:
    #     aux = LinearReg(data.groupby('time.month')[m], 'time')
    aux = LinearReg(data, 'time')
    # aux = xr.polyval(data.groupby('month')[m].time, aux.var_polyfit_coefficients[0]) + \
    #       aux.var_polyfit_coefficients[1]
    var_reg_n34 = aux.var_polyfit_coefficients[0]

    data['time'] = dmi
    # try:
    #     aux = LinearReg(data.groupby('month')[m], 'time')
    # except:
    #     aux = LinearReg(data.groupby('time.month')[m], 'time')
    aux = LinearReg(data, 'time')
    var_reg_dmi = aux.var_polyfit_coefficients[0]
    var_reg_dmi = aux.var_polyfit_coefficients[0]

    if two_variables:
        print('Two Variables')

        data2['time'] = n34
        #print('Full Season data2, m ignored')
        #aux = LinearReg(data2.groupby('month')[m], 'time')
        aux = LinearReg(data2, 'time')
        var_reg_n34_2 = aux.var_polyfit_coefficients[0]

        data2['time'] = dmi
        #aux = LinearReg(data2.groupby('month')[m], 'time')
        aux = LinearReg(data2, 'time')
        var_reg_dmi_2 = aux.var_polyfit_coefficients[0]

    return var_reg_n34, var_reg_dmi, var_reg_n34_2, var_reg_dmi_2

def RegWOEffect(n34, n34_wo_dmi, dmi, dmi_wo_n34, m=9, datos=None):

    datos['time'] = n34

    try:
        #aux = LinearReg(datos.groupby('month')[m], 'time')
        aux = LinearReg(datos, 'time')
        # aux = xr.polyval(datos.groupby('month')[m].time, aux.var_polyfit_coefficients[0]) +\
        #       aux.var_polyfit_coefficients[1]
        aux = xr.polyval(datos.time,
                         aux.var_polyfit_coefficients[0]) + \
              aux.var_polyfit_coefficients[1]
    except:
        #aux = LinearReg(datos.groupby('time.month')[m], 'time')
        aux = LinearReg(datos, 'time')
        # aux = xr.polyval(datos.groupby('time.month')[m].time, aux.var_polyfit_coefficients[0]) +\
        #       aux.var_polyfit_coefficients[1]
        aux = xr.polyval(datos.time, aux.var_polyfit_coefficients[0]) +\
              aux.var_polyfit_coefficients[1]
    #wo n34
    try:
        #var_regdmi_won34 = datos.groupby('month')[m]-aux
        var_regdmi_won34 = datos - aux

        #var_regdmi_won34['time'] = dmi_wo_n34.groupby('time.month')[m] #index wo influence
        var_regdmi_won34['time'] = dmi_wo_n34
        var_dmi_won34 = LinearReg(var_regdmi_won34,'time')
    except:
        #var_regdmi_won34 = datos.groupby('time.month')[m] - aux
        var_regdmi_won34 = datos - aux

        #var_regdmi_won34['time'] = dmi_wo_n34.groupby('time.month')[m]  # index wo influence
        var_regdmi_won34['time'] = dmi_wo_n34  # index wo influence
        var_dmi_won34 = LinearReg(var_regdmi_won34, 'time')

    #-----------------------------------------#

    datos['time'] = dmi
    try:
        #aux = LinearReg(datos.groupby('month')[m], 'time')
        aux = LinearReg(datos, 'time')
        # aux = xr.polyval(datos.groupby('month')[m].time, aux.var_polyfit_coefficients[0]) + \
        #   aux.var_polyfit_coefficients[1]
        aux = xr.polyval(datos.time,
                         aux.var_polyfit_coefficients[0]) + \
              aux.var_polyfit_coefficients[1]
    except:
        aux = LinearReg(datos.groupby('time.month')[m], 'time')
        aux = LinearReg(datos, 'time')
        # aux = xr.polyval(datos.groupby('time.month')[m].time, aux.var_polyfit_coefficients[0]) + \
        #   aux.var_polyfit_coefficients[1]
        aux = xr.polyval(datos.time,
                         aux.var_polyfit_coefficients[0]) + \
              aux.var_polyfit_coefficients[1]
    #wo
    try:
        # var_regn34_wodmi = datos.groupby('month')[m]-aux
        # var_regn34_wodmi['time'] = n34_wo_dmi.groupby('time.month')[m] #index wo influence
        var_regn34_wodmi = datos-aux
        var_regn34_wodmi['time'] = n34_wo_dmi #index wo influence
        var_n34_wodmi = LinearReg(var_regn34_wodmi,'time')

    except:
        # var_regn34_wodmi = datos.groupby('time.month')[m]-aux
        # var_regn34_wodmi['time'] = n34_wo_dmi.groupby('time.month')[m] #index wo influence
        var_regn34_wodmi = datos - aux
        var_regn34_wodmi['time'] = n34_wo_dmi #index wo influence
        var_n34_wodmi = LinearReg(var_regn34_wodmi,'time')

    return var_n34_wodmi.var_polyfit_coefficients[0],\
           var_dmi_won34.var_polyfit_coefficients[0],\
           var_regn34_wodmi,var_regdmi_won34

def Corr(datos, index, time_original, m=9):
    try:
        # aux_corr1 = xr.DataArray(datos.groupby('month')[m]['var'],
        #                      coords={'time': time_original.groupby('time.month')[m].values,
        #                              'lon': datos.lon.values, 'lat': datos.lat.values},
        #                      dims=['time', 'lat', 'lon'])

        aux_corr1 = xr.DataArray(datos['var'],
                             coords={'time': time_original.values,
                                     'lon': datos.lon.values, 'lat': datos.lat.values},
                             dims=['time', 'lat', 'lon'])
    except:
        # aux_corr1 = xr.DataArray(datos.groupby('time.month')[m]['var'],
        #                      coords={'time': time_original.groupby('time.month')[m].values,
        #                              'lon': datos.lon.values, 'lat': datos.lat.values},
        #                      dims=['time', 'lat', 'lon'])
        aux_corr1 = xr.DataArray(datos['var'],
                             coords={'time': time_original.values,
                                     'lon': datos.lon.values, 'lat': datos.lat.values},
                             dims=['time', 'lat', 'lon'])

    # aux_corr2 = xr.DataArray(index.groupby('time.month')[m],
    #                          coords={'time': time_original.groupby('time.month')[m]},
    #                          dims={'time'})
    aux_corr2 = xr.DataArray(index,
                             coords={'time': time_original},
                             dims={'time'})

    return xr.corr(aux_corr1, aux_corr2, 'time')

def PlotReg(data, data_cor, levels=np.linspace(-100,100,2), cmap='RdBu_r'
            , dpi=100, save=False, title='\m/', name_fig='fig_PlotReg', sig=True, out_dir=''
            ,two_variables = False, data2=None, data_cor2=None, levels2 = np.linspace(-100,100,2)
            , sig2=True, sig_point2=False, color_sig2='k'
            , color_contour2='k',step=1,SA=False, color_map='#d9d9d9'
            , color_sig='magenta', sig_point=False, r_crit=1
            , third_variable=False, data3=None, levels3=np.linspace(-1,1,11)):

    import matplotlib.pyplot as plt
    levels_contour = levels.copy()
    if isinstance(levels_contour, np.ndarray):
        levels_contour = levels_contour[levels_contour != 0]
    else:
        levels_contour.remove(0)

    crs_latlon = ccrs.PlateCarree()
    if SA:
        fig = plt.figure(figsize=(5, 6), dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent([270,330, -60,20], crs=crs_latlon)
    else:
        fig = plt.figure(figsize=(9, 3.5), dpi=dpi)
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent([0, 359, -80, 20], crs=crs_latlon)

    ax.contour(data.lon[::step], data.lat[::step], data[::step, ::step], linewidths=.5, alpha=0.5,
               levels=levels_contour, transform=crs_latlon, colors='black')

    im = ax.contourf(data.lon[::step], data.lat[::step], data[::step,::step],levels=levels,
                     transform=crs_latlon, cmap=cmap, extend='both')
    if sig:
        if sig_point:
            colors_l = [color_sig, color_sig]
            cs = ax.contourf(data_cor.lon, data_cor.lat, data_cor.where(np.abs(data_cor) > np.abs(r_crit)),
                             transform=crs_latlon, colors='none', hatches=["...", "..."],
                             extend='lower')
            for i, collection in enumerate(cs.collections):
                collection.set_edgecolor(colors_l[i % len(colors_l)])

            for collection in cs.collections:
                collection.set_linewidth(0.)
            # para hgt200 queda mejor los dos juntos
            # ax.contour(data_cor.lon[::step], data_cor.lat[::step], data_cor[::step, ::step],
            #            levels=np.linspace(-r_crit, r_crit, 2),
            #            colors=color_sig, transform=crs_latlon, linewidths=1)

        else:
            ax.contour(data_cor.lon[::step], data_cor.lat[::step], data_cor[::step, ::step],
                       levels=np.linspace(-r_crit, r_crit, 2),
                       colors=color_sig, transform=crs_latlon, linewidths=1)


    if two_variables:
        ax.contour(data2.lon, data2.lat, data2, levels=levels2,
                   colors=color_contour2, transform=crs_latlon, linewidths=1)
        if sig2:
            if sig_point2:
                colors_l = [color_sig2, color_sig2]
                cs = ax.contourf(data_cor2.lon, data_cor2.lat, data_cor2.where(np.abs(data_cor2) > np.abs(r_crit)),
                                 transform=crs_latlon, colors='none', hatches=["...", "..."],
                                 extend='lower', alpha=0.5)
                for i, collection in enumerate(cs.collections):
                    collection.set_edgecolor(colors_l[i % len(colors_l)])

                for collection in cs.collections:
                    collection.set_linewidth(0.)
                # para hgt200 queda mejor los dos juntos
                ax.contour(data_cor2.lon[::step], data_cor2.lat[::step], data_cor2[::step, ::step],
                           levels=np.linspace(-r_crit, r_crit, 2),
                           colors=color_sig2, transform=crs_latlon, linewidths=1)
            else:
                ax.contour(data_cor2.lon, data_cor2.lat, data_cor2, levels=np.linspace(-r_crit, r_crit, 2),
                       colors=color_sig2, transform=crs_latlon, linewidths=1)
                from matplotlib import colors
                cbar = colors.ListedColormap([color_sig2, 'white', color_sig2])
                cbar.set_over(color_sig2)
                cbar.set_under(color_sig2)
                cbar.set_bad(color='white')
                ax.contourf(data_cor2.lon, data_cor2.lat, data_cor2, levels=[-1,-r_crit, 0, r_crit,1],
                       cmap=cbar, transform=crs_latlon, linewidths=1, alpha=0.3)

    if third_variable:
        ax.contour(data3.lon[::2], data3.lat[::2], data3[::2,::2],levels=levels3,
                   colors=['#D300FF','#00FF5D'], transform=crs_latlon, linewidths=1.5)

    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='white', edgecolor=color_map)
    #ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
    # ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    if SA:
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5, zorder=17)
        #ax.add_feature(cartopy.feature.COASTLINE)
        ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean',
                                                    scale='50m', facecolor='white', alpha=1)
        ax.add_feature(ocean, linewidth=0.2, zorder=15)
        ax.set_xticks(np.arange(270, 330, 10), crs=crs_latlon)
        ax.set_yticks(np.arange(-60, 20, 20), crs=crs_latlon)

        ax2 = ax.twinx()
        ax2.set_yticks([])
        #ax2.set_xticks([])

    else:
        ax.set_xticks(np.arange(0, 360, 30), crs=crs_latlon)
        ax.set_yticks(np.arange(-80, 20, 10), crs=crs_latlon)
        ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
        ax.coastlines(color=color_map, linestyle='-', alpha=1)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-', zorder=20)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    #ax2.spines['left'].set_color('k')

    ax.tick_params(labelsize=7)
    plt.title(title, fontsize=10)
    plt.tight_layout()

    if save:
        print('save: ' + out_dir + name_fig + '.jpg')
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()

    else:
        plt.show()


def ComputeWithEffect(data=None, data2=None, n34=None, dmi=None,
                     two_variables=False, full_season=False,
                     time_original=None,m=9):
    print('Reg...')
    print('#-- With influence --#')
    aux_n34, aux_dmi, aux_n34_2, aux_dmi_2 = RegWEffect(data=data, data2=data2,
                                                       n34=n34.__mul__(1 / n34.std('time')),
                                                       dmi=dmi.__mul__(1 / dmi.std('time')),
                                                       m=m, two_variables=two_variables)
    if full_season:
        print('Full Season')
        n34 = n34.rolling(time=5, center=True).mean()
        dmi = dmi.rolling(time=5, center=True).mean()

    print('Corr...')
    aux_corr_n34 = Corr(datos=data, index=n34, time_original=time_original, m=m)
    aux_corr_dmi = Corr(datos=data, index=dmi, time_original=time_original, m=m)

    aux_corr_dmi_2 = 0
    aux_corr_n34_2 = 0
    if two_variables:
        print('Corr2..')
        aux_corr_n34_2 = Corr(datos=data2, index=n34, time_original=time_original, m=m)
        aux_corr_dmi_2 = Corr(datos=data2, index=dmi, time_original=time_original, m=m)

    return aux_n34, aux_corr_n34, aux_dmi, aux_corr_dmi, aux_n34_2, aux_corr_n34_2, aux_dmi_2, aux_corr_dmi_2

def ComputeWithoutEffect(data, n34, dmi, m, time_original):
    # -- Without influence --#
    print('# -- Without influence --#')
    print('Reg...')
    # dmi wo n34 influence and n34 wo dmi influence
    dmi_wo_n34, n34_wo_dmi = LinearReg1_D(n34.__mul__(1 / n34.std('time')),
                                          dmi.__mul__(1 / dmi.std('time')))

    # Reg WO
    aux_n34_wodmi, aux_dmi_won34, data_n34_wodmi, data_dmi_won34 = \
        RegWOEffect(n34=n34.__mul__(1 / n34.std('time')),
                   n34_wo_dmi=n34_wo_dmi,
                   dmi=dmi.__mul__(1 / dmi.std('time')),
                   dmi_wo_n34=dmi_wo_n34,
                   m=m, datos=data)

    print('Corr...')
    aux_corr_n34 = Corr(datos=data_n34_wodmi, index=n34_wo_dmi, time_original=time_original,m=m)
    aux_corr_dmi = Corr(datos=data_dmi_won34, index=dmi_wo_n34, time_original=time_original,m=m)

    return aux_n34_wodmi, aux_corr_n34, aux_dmi_won34, aux_corr_dmi
########################################################################################################################
# CFSv2 ################################################################################################################
def SelectNMMEFiles(model_name, variable, dir, anio='0', in_month='0', by_r=False, r='0',  All=False):
    import glob
    """
    Selecciona los archivos en funcion de del mes de entrada (in_month) o del miembro de ensamble (r)

    :param model_name: [str] nombre del modelo
    :param variable:[str] variable usada en el nombre del archivo
    :param dir:[str] directorio de los archivos a abrir
    :param anio:[str] anio de inicio del pronostico
    :param in_month:[str] mes de inicio del pronostico
    :param by_r: [bool] True para seleccionar todos los archivos de un mismo miembro de ensamble
    :param r: [str] solo si by_r = True, numero del miembro de ensamble que se quiere abrir
    :return: lista con los nombres de los archivos seleccionados
    """

    if by_r==False:

        if ((isinstance(model_name, str) == False) | (isinstance(variable, str) == False) |
                (isinstance(dir, str) == False) | (isinstance(in_month, str) == False)
                | (isinstance(anio, str) == False)):
            print('ERROR: model_name, variable, dir and in_month must be a string')
            return

        if int(in_month) < 10:
            m_in = '0'
        else:
            m_in = ''

        if in_month == '1':
            y1 = 0
            m1 = -11
            m_en = ''
        elif int(in_month) > 10:
            y1 = 1
            m1 = 1
            m_en = ''
            print('Year in chagend')
            anio = str(int(anio) - 1)
        else:
            y1 = 1
            m1 = 1
            m_en = '0'

    if by_r:
        if (isinstance(r, str) == False):
            print('ERROR: r must be a string')
            return

        files = glob.glob(dir + variable + '_Amon_' + model_name + '_*'
                          '_r'+ r +'_*' + '-*.nc')

    elif All:
        print('All=True')
        files = glob.glob(dir + variable + '_Amon_' + model_name + '_*'
                          '_r*' +'_*' + '-*.nc')

    else:
        files = glob.glob(dir + variable + '_Amon_' + model_name + '_' + anio + m_in + in_month +
                          '_r*_' + anio + m_in + in_month + '-' + str(int(anio) + y1) + m_en +
                          str(int(in_month) - m1) + '.nc')

    return files

def SelectVariables(dates, data):

    t_count=0
    t_count_aux = 0
    for t in dates.index:
        try:
            r_t = t.r.values
        except:
            r_t = dates.r[t_count_aux].values
        L_t = t.L.values
        t_t = t.values
        try: #q elegancia la de francia...
            t_t*1
            t_t = t.time.values
        except:
            pass

        if t_count == 0:
            aux = data.where(data.L == L_t).sel(r=r_t, time=t_t)
            t_count += 1
        else:
            aux = xr.concat([aux,
                             data.where(data.L == L_t).sel(r=r_t, time=t_t)],
                            dim='time')
    return aux

def SelectBins(data, min, max, sd=1):
    # sd opcional en caso de no estar escalado
    if np.abs(min) > np.abs(max):
        return (data >= min*sd) & (data < max*sd)
    elif np.abs(min) < np.abs(max):
        return (data > min*sd) & (data <= max*sd)
    elif np.abs(min) == np.abs(max):
        return (data >= min*sd) & (data <= max*sd)

def BinsByCases(v, v_name, fix_factor, s, mm, c, c_count,
                bin_limits, bins_by_cases_dmi, bins_by_cases_n34, dates_dir, cases_dir,
                snr=False, neutro_clim=False, obsdates=False):

    def Weights(data):
        weights = np.transpose(np.tile(np.cos(np.linspace(-80,20,101) * np.pi / 180),
                                       (len(data.lon), 1)))
        data_w = data * weights
        return data_w

    # 1. se abren los archivos de los índices (completos y se pesan por su SD)
    # tambien los archivos de la variable en cuestion pero para cada "case" = c

    data_dates_dmi_or = xr.open_dataset(dates_dir + 'DMI_' + s + '_Leads_r_CFSv2.nc')
    data_dates_dmi_or /=  data_dates_dmi_or.mean('r').std()

    data_dates_n34_or = xr.open_dataset(dates_dir + 'N34_' + s + '_Leads_r_CFSv2.nc')
    data_dates_n34_or /= data_dates_n34_or.mean('r').std()

    # 1.1 Climatología y case
    end_nc_file = '.nc' if v != 'tref' else '_nodetrend.nc'

    if neutro_clim:
        clim = Weights(xr.open_dataset(cases_dir + v + '_neutros' + '_' + s.upper() + end_nc_file).rename({v_name: 'var'}) * fix_factor)
    else:
        clim = Weights(xr.open_dataset(cases_dir + v + '_' + s.lower() + end_nc_file).rename({v_name: 'var'}) * fix_factor)

    case = Weights(xr.open_dataset(cases_dir + v + '_' + c + '_' + s.upper() + end_nc_file).rename({v_name: 'var'}) * fix_factor)

    # Anomalía
    for l in [0, 1, 2, 3]:
        try:
            clim_aux = clim.sel(time=clim.time.dt.month.isin(mm - l)).mean(['r', 'time'])
        except:
            clim_aux = clim.sel(time=clim.time.dt.month.isin(mm - l)).mean(['time'])

        if l==0:
            anom = case.sel(time=case.time.dt.month.isin(mm - l)) - clim_aux
        else:
            anom2 = case.sel(time=case.time.dt.month.isin(mm - l)) - clim_aux
            anom = xr.concat([anom, anom2], dim='time')

    # 1.2
    anom = anom.sortby(anom.time.dt.month)

    # 2. Vinculo fechas case -> índices DMI y N34 para poder clasificarlos
    # las fechas entre el case variable y el case indices COINCIDEN,
    # DE ESA FORMA SE ELIGIERON LOS CASES VARIABLE
    # pero diferen en orden. Para evitar complicar la selección usando r y L
    # con .sortby(..time.dt.month) en cada caso se simplifica el problema
    # y coinciden todos los eventos en fecha, r y L

    if obsdates: # por ahora no funca
        print('Fechas Observadas deshabilitado')
        return
        # aux_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
        # aux_cases = xr.open_dataset(aux_dir + v + '_' + c + '_' + s + '_CFSv2_obsDates.nc')\
        #     .rename({v: 'index'})
        # aux_cases['index'] = aux_cases.time
        # aux_cases = aux_cases.drop(['lon', 'lat'])
    else:
        cases_date_dir = '/pikachu/datos/luciano.andrian/cases_dates/'
        try:
            aux_cases = xr.open_dataset(cases_date_dir + c + '_f_' + s + '.nc') \
                .rename({'__xarray_dataarray_variable__': 'index'})
        except:
            aux_cases = xr.open_dataset(cases_date_dir + c + '_f_' + s + '.nc') \
                .rename({'sst': 'index'})

    case_sel_dmi = SelectVariables(aux_cases, data_dates_dmi_or)
    case_sel_dmi = case_sel_dmi.sortby(case_sel_dmi.time.dt.month)
    case_sel_dmi_n34 = SelectVariables(aux_cases, data_dates_n34_or)
    case_sel_dmi_n34 = case_sel_dmi_n34.sortby(case_sel_dmi_n34.time.dt.month)

    # 2.1 uniendo var, dmi y n34
    data_merged = xr.Dataset(
        data_vars=dict(
            var=(['time', 'lat', 'lon'], anom['var'].values),
            dmi=(['time'], case_sel_dmi.sst.values),
            n34=(['time'], case_sel_dmi_n34.sst.values),
        ),
        coords=dict(
            time=anom.time.values
        )
    )

    bins_aux_dmi = bins_by_cases_dmi[c_count]
    bins_aux_n34 = bins_by_cases_n34[c_count]
    # 3. Seleccion en cada bin
    anom_bin_main = list()
    num_bin_main = list()
    for ba_dmi in range(0, len(bins_aux_dmi)):  # loops en las bins para el dmi segun case
        bins_aux = data_merged.where(SelectBins(data_merged.dmi,
                                                bin_limits[bins_aux_dmi[ba_dmi]][0],
                                                bin_limits[bins_aux_dmi[ba_dmi]][1]))
        anom_bin = list()
        num_bin = list()
        for ba_n34 in range(0, len(bins_aux_n34)):  # loop en las correspondientes al n34 segun case
            bin_f = bins_aux.where(SelectBins(bins_aux.n34,
                                              bin_limits[bins_aux_n34[ba_n34]][0],
                                              bin_limits[bins_aux_n34[ba_n34]][1]))

            if snr:
                spread = bin_f - bin_f.mean(['time'])
                spread = spread.std('time')
                SNR = bin_f.mean(['time']) / spread
                anom_bin.append(SNR)
            else:
                anom_bin.append(bin_f.mean('time')['var'])

            num_bin.append(len(np.where(~np.isnan(bin_f['dmi']))[0]))

        anom_bin_main.append(anom_bin)
        num_bin_main.append(num_bin)

    return anom_bin_main, num_bin_main, clim

def DetrendClim(data, mm, v_name='prec'):
    # la diferencia es mínima en fitlrar o no tendencia para hacer una climatología,
    # pero para no perder la costumbre de complicar las cosas...

    for l in [0, 1, 2, 3]:
        season_data = data.sel(time=data.time.dt.month.isin(mm - l), L=l)
        aux = season_data.polyfit(dim='time', deg=1)
        if v_name == 'prec':
            aux_trend = xr.polyval(season_data['time'], aux.prec_polyfit_coefficients[0])  # al rededor de la media
        elif v_name == 'tref':
            aux_trend = xr.polyval(season_data['time'], aux.tref_polyfit_coefficients[0])  # al rededor de la media
        elif v_name == 'hgt':
            aux_trend = xr.polyval(season_data['time'], aux.hgt_polyfit_coefficients[0])  # al rededor de la media

        if l == 0:
            season_anom_detrend = season_data - aux_trend
        else:
            aux_detrend = season_data - aux_trend
            season_anom_detrend = xr.concat([season_anom_detrend, aux_detrend], dim='time')

    return season_anom_detrend.mean(['r', 'time'])


def ComputeFieldsByCases(v, v_name, fix_factor, snr,
                         levels_main, cbar_main, levels_clim, cbar_clim,
                         title_var, name_fig, dpi,
                         cases, bin_limits, bins_by_cases_dmi, bins_by_cases_n34,
                         cases_dir, dates_dir,
                         figsize=[16, 17], usemask=True, hcolorbar=False, save=True,
                         proj='eq', obsdates=False, out_dir='~/'):
    # no, una genialidad... -------------------------------------------------------------------------------------------#
    sec_plot = [13, 14, 10, 11,
                7, 2, 22, 17,
                8, 3, 9, 4,
                20, 15, 21, 16,
                5, 0, 6, 1,
                23, 18, 24, 19]
    row_titles = ['Strong El Niño', None, None, None, None,
                  'Moderate El Niño', None, None, None, None,
                  'Neutro ENSO', None, None, None, None,
                  'Moderate La Niña', None, None, None, None,
                  'Strong La Niña', None, None, None, None]
    col_titles = ['Strong IOD - ', 'Moderate IOD - ', 'Neutro IOD', 'Moderate IOD + ', 'Strong IOD + ']
    num_neutros = [483, 585, 676, 673]
    porcentaje = 0.1
    # ------------------------------------------------------------------------------------------------------------------#
    print('Only SON')
    print('No climatology')
    mm = 10
    for s in ['SON']:
        n_check = []
        sec_count = 0
        # esto no tiene sentido
        # comp_case_clim = DetrendClim(data, mm, v_name=v_name)

        crs_latlon = ccrs.PlateCarree()
        if proj == 'eq':
            fig, axs = plt.subplots(nrows=5, ncols=5,
                                    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
                                    figsize=(figsize[0], figsize[1]))
        else:
            fig, axs = plt.subplots(nrows=5, ncols=5,
                                    subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=200)},
                                    figsize=(figsize[0], figsize[1]))
        axs = axs.flatten()
        # Loop en los cases -{neutro} ---------------------------------------------------------------------------------#
        for c_count in [0, 1, 2, 3, 4, 5, 6, 7]:  # , 8]:
            cases_bin, num_bin, aux = BinsByCases(v=v, v_name=v_name, fix_factor=fix_factor,
                                                  s=s, mm=mm, c=cases[c_count], c_count=c_count,
                                                  bin_limits=bin_limits, bins_by_cases_dmi=bins_by_cases_dmi,
                                                  bins_by_cases_n34=bins_by_cases_n34, snr=snr,
                                                  cases_dir=cases_dir, dates_dir=dates_dir, obsdates=obsdates)

            bins_aux_dmi = bins_by_cases_dmi[c_count]
            bins_aux_n34 = bins_by_cases_n34[c_count]
            for b_dmi in range(0, len(bins_aux_dmi)):
                for b_n34 in range(0, len(bins_aux_n34)):
                    n = sec_plot[sec_count]
                    if proj != 'eq':
                        axs[n].set_extent([0, 360, -80, 20],
                                          ccrs.PlateCarree(central_longitude=200))
                    comp_case = cases_bin[b_dmi][b_n34]


                    # if v == 'prec' and s == 'JJA':
                    #
                    #     mask2 = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(comp_case_clim)
                    #     mask2 = xr.where(np.isnan(mask2), mask2, 1)
                    #     mask2 = mask2.to_dataset(name='prec')
                    #
                    #     dry_season_mask = comp_case_clim.where(comp_case_clim.prec>30)
                    #     dry_season_mask = xr.where(np.isnan(dry_season_mask), dry_season_mask, 1)
                    #     dry_season_mask *= mask2
                    #
                    #     if snr:
                    #         comp_case['var'] *= dry_season_mask.prec
                    #     else:
                    #         comp_case *= dry_season_mask.prec.values

                    if usemask:
                        mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(aux)
                        mask = xr.where(np.isnan(mask), mask, 1)
                        comp_case *= mask
                    if snr:
                        comp_case = comp_case['var']

                    if num_bin[b_dmi][b_n34] > num_neutros[mm - 7] * porcentaje:
                        im = axs[n].contourf(aux.lon, aux.lat, comp_case,
                                             levels=levels_main, transform=crs_latlon,
                                             cmap=cbar_main, extend='both')

                        axs[n].add_feature(cartopy.feature.LAND, facecolor='lightgrey')
                        axs[n].add_feature(cartopy.feature.COASTLINE)
                        if proj == 'eq':
                            axs[n].gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-', color='gray')
                            axs[n].set_xticks([])
                            axs[n].set_yticks([])
                            # axs[n].set_xticks(x_lon, crs=crs_latlon)
                            # axs[n].set_yticks(x_lat, crs=crs_latlon)
                            # lon_formatter = LongitudeFormatter(zero_direction_label=True)
                            # lat_formatter = LatitudeFormatter()
                            # axs[n].xaxis.set_major_formatter(lon_formatter)
                            # axs[n].yaxis.set_major_formatter(lat_formatter)
                        else:
                            # polar
                            gls = axs[n].gridlines(draw_labels=True, crs=crs_latlon, lw=0.3, color="gray",
                                                   y_inline=True, xlocs=range(-180, 180, 30),
                                                   ylocs=np.arange(-80, 20, 20))
                            r_extent = 1.2e7
                            axs[n].set_xlim(-r_extent, r_extent)
                            axs[n].set_ylim(-r_extent, r_extent)
                            circle_path = mpath.Path.unit_circle()
                            circle_path = mpath.Path(circle_path.vertices.copy() * r_extent,
                                                     circle_path.codes.copy())
                            axs[n].set_boundary(circle_path)
                            axs[n].set_frame_on(False)
                            plt.draw()
                            for ea in gls._labels:
                                pos = ea[2].get_position()
                                if (pos[0] == 150):
                                    ea[2].set_position([0, pos[1]])

                        axs[n].tick_params(labelsize=0)

                        if n == 0 or n == 1 or n == 2 or n == 3 or n == 4:
                            axs[n].set_title(col_titles[n], fontsize=15)

                        if n == 0 or n == 5 or n == 10 or n == 15 or n == 20:
                            axs[n].set_ylabel(row_titles[n], fontsize=15)

                        axs[n].xaxis.set_label_position('top')
                        axs[n].set_xlabel('N=' + str(num_bin[b_dmi][b_n34]), fontsize=12, loc='left', fontweight="bold")

                    else:
                        n_check.append(n)
                        axs[n].axis('off')

                    sec_count += 1

        # subtitulos columnas de no ploteados -------------------------------------------------------------------------#
        for n_aux in [0, 1, 2, 3, 4]:
            if n_aux in n_check:
                if n_aux + 5 in n_check:
                    axs[n_aux + 10].set_title(col_titles[n_aux], fontsize=15)
                else:
                    axs[n_aux + 5].set_title(col_titles[n_aux], fontsize=15)

        for n_aux in [0, 5, 10, 15, 20]:
            if n_aux in n_check:
                if n_aux + 1 in n_check:
                    axs[n_aux + 2].set_ylabel(row_titles[n_aux], fontsize=15)
                else:
                    axs[n_aux + 1].set_ylabel(row_titles[n_aux], fontsize=15)

        # Climatologia = NADA en HGT ----------------------------------------------------------------------------------#
        # en el lugar del neutro -> climatología de la variable (data)

        # if usemask:
        #     comp_case_clim = comp_case_clim[v_name] * fix_factor * mask
        # else:
        #     comp_case_clim = comp_case_clim[v_name] * fix_factor

        # if v_name=='hgt':
        #     comp_case_clim = 0

        aux0 = aux.sel(r=1, time='1982-10-01').drop(['r', 'L', 'time'])
        im2 = axs[12].contourf(aux.lon, aux.lat, aux0['var'][0, :, :],
                               levels=levels_clim, transform=crs_latlon, cmap=cbar_clim, extend='max')
        # --------------------------------------------------------------------------------------------------------------#
        axs[12].add_feature(cartopy.feature.LAND, facecolor='grey')
        axs[12].add_feature(cartopy.feature.COASTLINE)

        if v_name != 'hgt':
            cb = plt.colorbar(im2, fraction=0.042, pad=0.032, shrink=1, ax=axs[12], aspect=20)
            cb.ax.tick_params(labelsize=5)

        if proj == 'eq':
            axs[12].gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-', color='gray')
            axs[12].set_xticks([])
            axs[12].set_yticks([])
        else:
            # polar
            gls = axs[12].gridlines(draw_labels=True, crs=crs_latlon, lw=0.3, color="gray",
                                    y_inline=True, xlocs=range(-180, 180, 30), ylocs=np.arange(-80, 0, 20))
            r_extent = 1.2e7
            axs[12].set_xlim(-r_extent, r_extent)
            axs[12].set_ylim(-r_extent, r_extent)
            circle_path = mpath.Path.unit_circle()
            circle_path = mpath.Path(circle_path.vertices.copy() * r_extent,
                                     circle_path.codes.copy())
            axs[12].set_boundary(circle_path)
            axs[12].set_frame_on(False)
            axs[12].tick_params(labelsize=0)
            plt.draw()
            for ea in gls._labels:
                pos = ea[2].get_position()
                if (pos[0] == 150):
                    ea[2].set_position([0, pos[1]])

        if hcolorbar:
            pos = fig.add_axes([0.2, 0.05, 0.6, 0.01])
            cbar = fig.colorbar(im, cax=pos, pad=0.1, orientation='horizontal')
        else:
            pos = fig.add_axes([0.90, 0.2, 0.012, 0.6])
            cbar = fig.colorbar(im, cax=pos, pad=0.1)

        cbar.ax.tick_params(labelsize=18)
        if snr:
            fig.suptitle('SNR:' + title_var + ' - ' + s, fontsize=20)
        else:
            fig.suptitle(title_var + ' - ' + s, fontsize=20)
        # fig.tight_layout() #BUG matplotlib 3.5.0 #Solucionado definitivamnete en 3.6 ?
        if hcolorbar:
            fig.subplots_adjust(top=0.93, bottom=0.07)
        else:
            fig.subplots_adjust(top=0.93)
        if save:
            if snr:
                plt.savefig(out_dir + 'SNR_' + name_fig + '_' + s + '.jpg', bbox_inches='tight', dpi=dpi)
            else:
                plt.savefig(out_dir + name_fig + '_' + s + '.jpg', bbox_inches='tight', dpi=dpi)

            plt.close('all')
        else:
            plt.show()
        mm += 1

# Plots ################################################################################################################
def PlotComp(comp, comp_var, title='Fig', fase=None, name_fig='Fig',
             save=False, dpi=200, levels=np.linspace(-1.5, 1.5, 13),
             contour=False, cmap='RdBu_r', number_events='', season = '',
             waf=False, px=None, py=None, text=True, SA=False,
             two_variables = False, comp2=None, step = 1,
             levels2=np.linspace(-1.5, 1.5, 13), contour0 = False):

    from numpy import ma
    import matplotlib.pyplot as plt


    if SA:
        fig = plt.figure(figsize=(5, 6), dpi=dpi)
    else:
        fig = plt.figure(figsize=(7, 3), dpi=dpi)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    if SA:
        ax.set_extent([270,330, -60,20],crs_latlon)
    else:
        ax.set_extent([0, 359, -90, 10], crs=crs_latlon)


    im = ax.contourf(comp.lon[::step], comp.lat[::step], comp_var[::step,::step],
                     levels=levels,transform=crs_latlon, cmap=cmap, extend='both')
    if contour:
        values = ax.contour(comp.lon, comp.lat, comp_var, levels=levels,
                            transform=crs_latlon, colors='darkgray', linewidths=1)
        ax.clabel(values, inline=1, fontsize=5, fmt='%1.1f')

    if contour0:
        values = ax.contour(comp.lon, comp.lat, comp_var, levels=0,
                            transform=crs_latlon, colors='magenta', linewidths=1)
        ax.clabel(values, inline=1.5, fontsize=5, fmt='%1.1f')

    if two_variables:
        print('Plot Two Variables')
        comp_var2 = comp2['var'] ######## CORREGIR en caso de generalizar #############
        values2 = ax.contour(comp2.lon, comp2.lat, comp_var2, levels=levels2,
                            transform=crs_latlon, colors='k', linewidths=0.5, alpha=0.6)
        #ax.clabel(values2, inline=1, fontsize=5, fmt='%1.1f')


    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='#d9d9d9')
    ax.add_feature(cartopy.feature.COASTLINE)
    # ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    if SA:
        ax.set_xticks(np.arange(270, 330, 10), crs=crs_latlon)
        ax.set_yticks(np.arange(-60, 40, 20), crs=crs_latlon)
    else:
        ax.set_xticks(np.arange(30, 330, 60), crs=crs_latlon)
        ax.set_yticks(np.arange(-90, 10, 10), crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)

    if waf:
        Q60 = np.percentile(np.sqrt(np.add(np.power(px, 2), np.power(py, 2))), 0)
        M = np.sqrt(np.add(np.power(px, 2), np.power(py, 2))) < Q60
        # mask array
        px_mask = ma.array(px, mask=M)
        py_mask = ma.array(py, mask=M)
        # plot vectors
        lons, lats = np.meshgrid(comp.lon.values, comp.lat.values)
        ax.quiver(lons[::17, ::17], lats[::17, ::17], px_mask[0, ::17, ::17],
                  py_mask[0, ::17, ::17], transform=crs_latlon,pivot='tail',
                  width=0.0014,headwidth=4.1, alpha=0.8, color='k')
                  #, scale=1/10)#, width=1.5e-3, headwidth=3.1,  # headwidht (default3)
                  #headlength=2.2)  # (default5))

    plt.title(title, fontsize=10)
    if text:
        plt.figtext(0.5, 0.01, number_events, ha="center", fontsize=10,
                bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5})
    plt.tight_layout()

    if save:
        plt.savefig(name_fig + str(season) + '_' + str(fase.split(' ', 1)[1]) + '.jpg')
        plt.close()
    else:
        plt.show()

def Plots(data, variable='var', neutral=None, DMI_pos=None, DMI_neg=None,
          N34_pos=None, N34_neg=None, neutral_name='', cmap='RdBu_r',
          dpi=200, mode='', levels=np.linspace(-1.5, 1.5, 13),
          name_fig='', save=False, contour=False, title="", waf=False,
          two_variables=False, data2=None, neutral2=None, levels2=None,
          season=None, text=True, SA=False, contour0=False, step=1,
          px=None, py=None):

    if two_variables == False:
        if data is None:
            if data2 is None:
                print('data None!')
            else:
                data = data2
                print('Data is None!')
                print('Using data2 instead data')
                levels = levels2
                neutral = neutral2

    def Title(DMI_phase, N34_phase, title=title):
        DMI_phase = set(DMI_phase)
        N34_phase = set(N34_phase)
        if mode.split(' ', 1)[0] != 'Simultaneus':
            if mode.split(' ', 1)[1] == 'IODs':
                title = title + mode + ': ' + str(len(DMI_phase)) + '\n' + 'against ' + clim
                number_events = str(DMI_phase)
            else:
                title = title + mode + ': ' + str(len(N34_phase)) + '\n' + 'against ' + clim
                number_events = str(N34_phase)

        elif mode.split(' ', 1)[0] == 'Simultaneus':
            title = title +mode + '\n' + 'IODs: ' + str(len(DMI_phase)) + \
                    ' - ENSOs: ' + str(len(N34_phase)) + '\n' + 'against ' + clim
            number_events = str(N34_phase)
        return title, number_events




    if data[0] != 0:
        comp = data[0] - neutral
        clim = neutral_name
        try:
            comp2 = data2[0] - neutral2
        except:
            comp2 = None
            print('One Variable')

        PlotComp(comp=comp, comp_var=comp[variable],
                 title=Title(DMI_phase=DMI_pos, N34_phase=N34_pos)[0],
                 fase=' - Positive', name_fig=name_fig,
                 save=save, dpi=dpi, levels=levels,
                 contour=contour, cmap=cmap,
                 number_events=Title(DMI_phase=DMI_pos, N34_phase=N34_pos)[1],
                 season=season,
                 waf=waf, px=px, py=py, text=text, SA=SA,
                 two_variables=two_variables,
                 comp2=comp2, step=step,
                 levels2=levels2, contour0=contour0)

    if data[1] != 0:
        comp = data[1] - neutral
        clim = neutral_name
        try:
            comp2 = data2[1] - neutral2
        except:
            comp2 = None
            print('One Variable')

        PlotComp(comp=comp, comp_var=comp[variable],
                 title=Title(DMI_phase=DMI_neg, N34_phase=N34_neg)[0],
                 fase=' - Negative', name_fig=name_fig,
                 save=save, dpi=dpi, levels=levels,
                 contour=contour, cmap=cmap,
                 number_events=Title(DMI_phase=DMI_neg, N34_phase=N34_neg)[1],
                 season=season,
                 waf=waf, px=px, py=py, text=text, SA=SA,
                 two_variables=two_variables,
                 comp2=comp2, step=step,
                 levels2=levels2, contour0=contour0)


def PlotComposite_wWAF(comp, levels, cmap, step1, contour1=True,
                       two_variables=False, comp2=None, levels2=np.linspace(-1, 1, 13), step2=4,
                       mapa='sa', title='title', name_fig='name_fig', dpi=100, save=False,
                       comp_sig=None, color_sig='k', significance=True, linewidht2=.5, color_map='#d9d9d9',
                       out_dir='RUTA', proj='eq', borders=False,
                       third_variable=False, comp3=None, levels_contour3=np.linspace(-1, 1, 13),
                       waf=False, data_waf=None, px=None, py=False, waf_scale=1 / 1000, step_waf=10, hatches='..'):
    import matplotlib.pyplot as plt
    import cartopy.feature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs

    if mapa.lower() == 'sa':
        fig_size = (5, 6)
        extent = [270, 330, -60, 20]
        xticks = np.arange(270, 330, 10)
        yticks = np.arange(-60, 40, 20)

    elif mapa.lower() == 'tropical':
        fig_size = (7, 2)
        extent = [40, 280, -20, 20]
        xticks = np.arange(40, 280, 60)
        yticks = np.arange(-20, 20, 20)

    elif mapa.lower() == 'hs':
        fig_size = (9, 3.5)
        extent = [0, 359, -80, 20]
        xticks = np.arange(0, 360, 30)
        yticks = np.arange(-80, 20, 10)
        if proj != 'eq':
            fig_size = (5, 5)
    else:
        fig_size = (8, 3)
        extent = [30, 330, -80, 20]
        xticks = np.arange(30, 330, 30)
        yticks = np.arange(-80, 20, 10)
        if proj != 'eq':
            fig_size = (5, 5)

    levels_contour = levels.copy()
    comp_var = comp['var']
    if isinstance(levels, np.ndarray):
        levels_contour = levels[levels != 0]
    else:
        levels_contour.remove(0)

    if two_variables:
        levels_contour2 = levels2.copy()
        comp_var2 = comp2['var']
        if isinstance(levels2, np.ndarray):
            levels_contour2 = levels2[levels2 != 0]
        else:
            levels_contour2.remove(0)

    crs_latlon = ccrs.PlateCarree()
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    if proj == 'eq':
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent(extent, crs=crs_latlon)
    else:
        ax = plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=200))
        ax.set_extent([30, 340, -90, 0],
                      ccrs.PlateCarree(central_longitude=200))

    if two_variables:
        ax.contour(comp2.lon[::step2], comp2.lat[::step2], comp_var2[::step2, ::step2],
                   linewidths=linewidht2, levels=levels_contour2, transform=crs_latlon, colors='k')
    else:
        if contour1:
            ax.contour(comp.lon[::step1], comp.lat[::step1], comp_var[::step1, ::step1],
                       linewidths=.8, levels=levels_contour, transform=crs_latlon, colors='black')

    im = ax.contourf(comp.lon[::step1], comp.lat[::step1], comp_var[::step1, ::step1],
                     levels=levels, transform=crs_latlon, cmap=cmap, extend='both')

    if third_variable:
        comp_var3 = comp3['var']
        tv = ax.contour(comp3.lon[::2], comp3.lat[::2], comp_var3[::2, ::2], levels=levels_contour3,
                        colors=['#D300FF', '#00FF5D'], transform=crs_latlon, linewidths=1.5)
        # tv.monochrome = True
        # for col, ls in zip(tv.collections, tv._process_linestyles()):
        #     col.set_linestyle(ls)

    if significance:
        colors_l = [color_sig, color_sig]
        comp_sig_var = comp_sig['var']
        cs = ax.contourf(comp_sig.lon, comp_sig.lat, comp_sig_var,
                         transform=crs_latlon, colors='none',
                         hatches=[hatches, hatches], extend='lower')
        for i, collection in enumerate(cs.collections):
            collection.set_edgecolor(colors_l[i % len(colors_l)])

        for collection in cs.collections:
            collection.set_linewidth(0.)

    if waf:
        from numpy import ma
        Q60 = np.nanpercentile(np.sqrt(np.add(np.power(px, 2), np.power(py, 2))), 60)
        M = np.sqrt(np.add(np.power(px, 2), np.power(py, 2))) < Q60
        # mask array
        px_mask = ma.array(px, mask=M)
        py_mask = ma.array(py, mask=M)

        Q99 = np.nanpercentile(np.sqrt(np.add(np.power(px, 2), np.power(py, 2))), 99)
        M = np.sqrt(np.add(np.power(px, 2), np.power(py, 2))) > Q99
        # mask array
        px_mask = ma.array(px_mask, mask=M)
        py_mask = ma.array(py_mask, mask=M)

        # plot vectors
        lons, lats = np.meshgrid(data_waf.lon.values, data_waf.lat.values)
        ax.quiver(lons[::step_waf, ::step_waf], lats[::step_waf, ::step_waf],
                  px_mask[0, ::step_waf, ::step_waf], py_mask[0, ::step_waf, ::step_waf],
                  transform=crs_latlon, pivot='tail', width=1.5e-3, headwidth=3, alpha=1,
                  headlength=2.5, color='k', scale=waf_scale)

    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='white', edgecolor=color_map)
    if borders:
        ax.add_feature(cartopy.feature.BORDERS, facecolor='white',
                       edgecolor=color_map)
    # ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
    ax.coastlines(color=color_map, linestyle='-', alpha=1)

    if proj == 'eq':
        ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
        ax.set_xticks(xticks, crs=crs_latlon)
        ax.set_yticks(yticks, crs=crs_latlon)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
    else:
        gls = ax.gridlines(draw_labels=True, crs=crs_latlon, lw=0.3, color="gray",
                           y_inline=True, xlocs=range(-180, 180, 30), ylocs=np.arange(-80, 0, 20))

        r_extent = 1.2e7
        ax.set_xlim(-r_extent, r_extent)
        ax.set_ylim(-r_extent, r_extent)
        circle_path = mpath.Path.unit_circle()
        circle_path = mpath.Path(circle_path.vertices.copy() * r_extent,
                                 circle_path.codes.copy())
        ax.set_boundary(circle_path)
        ax.set_frame_on(False)
        plt.draw()
        for ea in gls._labels:
            pos = ea[2].get_position()
            if (pos[0] == 150):
                ea[2].set_position([0, pos[1]])

    ax.tick_params(labelsize=7)
    plt.title(title, fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()
########################################################################################################################

