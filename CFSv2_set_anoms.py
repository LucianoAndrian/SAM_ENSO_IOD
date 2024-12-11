"""
Seteo variables CFSv2
"""
# ---------------------------------------------------------------------------- #
save = True

dir_hc = '/pikachu/datos/luciano.andrian/hindcast/'
dir_rt = '/pikachu/datos/luciano.andrian/real_time/'
out_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from ENSO_IOD_Funciones import SelectNMMEFiles
# ---------------------------------------------------------------------------- #
def fix_calendar(ds, timevar='time'):
    """
    agrega los dias a los archivos nc de NMME
    """
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'
    return ds

def TwoClim_Anom_Seasons(data_1982_1998, data_1999_2011, main_month_season):

    for l in [0,1,2,3]:
        season_1982_1998 = data_1982_1998.sel(
            time=data_1982_1998.time.dt.month.isin(main_month_season-l), L=l)
        season_1999_2011 = data_1999_2011.sel(
            time=data_1999_2011.time.dt.month.isin(main_month_season-l), L=l)

        if l==0:
            season_clim_1982_1998 = season_1982_1998.mean(['r', 'time'])
            season_clim_1999_2011 = season_1999_2011.mean(['r', 'time'])

            season_anom_1982_1998 = season_1982_1998 - season_clim_1982_1998
            season_anom_1999_2011 = season_1999_2011 - season_clim_1999_2011
        else:
            season_clim_1982_1998 = xr.concat(
                [season_clim_1982_1998,
                 season_1982_1998.mean(['r', 'time'])], dim='L')
            season_clim_1999_2011 = xr.concat(
                [season_clim_1999_2011,
                 season_1999_2011.mean(['r', 'time'])], dim='L')

            aux_1982_1998 = (season_1982_1998 -
                             season_1982_1998.mean(['r', 'time']))
            aux_1999_2011 = (season_1999_2011 -
                             season_1999_2011.mean(['r', 'time']))

            season_anom_1982_1998 = xr.concat(
                [season_anom_1982_1998, aux_1982_1998], dim='time')
            season_anom_1999_2011 = xr.concat(
                [season_anom_1999_2011, aux_1999_2011], dim='time')

    return (season_clim_1982_1998, season_clim_1999_2011,
            season_anom_1982_1998, season_anom_1999_2011)


def Detrend_Seasons(season_anom_1982_1998, season_anom_1999_2011,
                    main_month_season):

    for l in [0,1,2,3]:
        # 1982-1998 ---------------- #
        aux_season_anom_1982_1998 \
            = season_anom_1982_1998.sel(
            time=season_anom_1982_1998.time.dt.month.isin(main_month_season-l))

        aux = aux_season_anom_1982_1998.mean('r').polyfit(dim='time', deg=1)
        v_name = list(aux.data_vars)[0].split('_')[0]
        aux_trend = xr.polyval(
            aux_season_anom_1982_1998['time'],
            aux[f'{v_name}_polyfit_coefficients'])
        if l == 0:
            season_anom_1982_1998_detrened = (
                    aux_season_anom_1982_1998 - aux_trend)
        else:
            aux_detrend = aux_season_anom_1982_1998 - aux_trend
            season_anom_1982_1998_detrened = xr.concat(
                [season_anom_1982_1998_detrened, aux_detrend], dim='time')

        # 1999-2011 ----------------
        aux_season_anom_1999_2011 = season_anom_1999_2011.sel(
            time=season_anom_1999_2011.time.dt.month.isin(
                main_month_season - l))

        aux = aux_season_anom_1999_2011.mean('r').polyfit(dim='time', deg=1)
        v_name = list(aux.data_vars)[0].split('_')[0]
        aux_trend = xr.polyval(aux_season_anom_1999_2011['time'],
            aux[f'{v_name}_polyfit_coefficients'])

        if l==0:
            season_anom_1999_2011_detrend = (
                    aux_season_anom_1999_2011 - aux_trend)
        else:
            aux_detrend = aux_season_anom_1999_2011 - aux_trend
            season_anom_1999_2011_detrend = xr.concat(
                [season_anom_1999_2011_detrend, aux_detrend], dim='time')

    return season_anom_1982_1998_detrened, season_anom_1999_2011_detrend


def Anom_Detrend_SeasonRealTime(data_realtime, season_clim_1999_2011,
                                main_month_season):

    for l in [0,1,2,3]:
        season_data = data_realtime.sel(
            time=data_realtime.time.dt.month.isin(main_month_season-l), L=l)
        aux_season_clim_1999_2011 = season_clim_1999_2011.sel(L=l)

        #Anomalia
        season_anom = season_data - aux_season_clim_1999_2011

        #Detrend
        aux = season_anom.mean('r').polyfit(dim='time', deg=1)
        v_name = list(aux.data_vars)[0].split('_')[0]
        aux_trend = xr.polyval(season_anom['time'],
                               aux[f'{v_name}_polyfit_coefficients'])

        if l==0:
            season_anom_detrend = season_anom - aux_trend
        else:
            aux_detrend = season_anom - aux_trend
            season_anom_detrend = xr.concat(
                [season_anom_detrend, aux_detrend], dim='time')

    return season_anom_detrend


def OpenVariablesCFSv2(variable,  dir=dir_hc, lat=[-80,20], hgt_P=200):

    files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=variable,
                            dir=dir, All=True)
    files = sorted(files, key=lambda x: x.split()[0])

    # abriendo todos los archivos
    data = xr.open_mfdataset(files, decode_times=False)
    data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
    data = data.sel(L=[0.5, 1.5, 2.5, 3.5])  # Solo leads 0 1 2 3
    data['L'] = [0, 1, 2, 3]
    data = xr.decode_cf(fix_calendar(data))  # corrigiendo fechas
    if variable=='hgt':
        try:
            data = data.sel(lat=slice(min(lat), max(lat)), P=hgt_P)
            data = data.drop('P')
            try:
                data = data.drop('Z')
            except:
                pass
        except:
            data = data.sel(lat=slice(min(lat), max(lat)))

    elif variable.lower()=='u50':
        data = data.sel(lat=-60, P=50)
        data = data.drop('P')
        data = data.drop('lat')

    else:
        data = data.sel(lat=slice(min(lat), max(lat)))

    return data


def TwoHindcast(hindcast):

    data_1982_1998 = hindcast.sel(
        time=hindcast.time.dt.year.isin(np.linspace(1982, 1998, 17)))
    data_1999_2011 = hindcast.sel(
        time=hindcast.time.dt.year.isin(np.linspace(1999, 2011, 13)))

    return data_1982_1998, data_1999_2011


def SetAnoms(hincast_1982_1998, hindcast_1999_2011, realtime, mm):

    # Hindcast
    (season_clim_82_98, season_clim_99_11,
     season_anom_82_98, season_anom_99_11) = TwoClim_Anom_Seasons(
        hincast_1982_1998, hindcast_1999_2011, mm)

    season_anom_82_98_detrend, season_anom_99_11_detrend = \
        Detrend_Seasons(season_anom_82_98, season_anom_99_11, mm)

    season_hindcast_detrend = xr.concat([season_anom_82_98_detrend,
                                         season_anom_99_11_detrend],
                                        dim='time')

    season_clim_99_11 = season_clim_99_11.load()

    # Realtime
    season_realtime_detrend = Anom_Detrend_SeasonRealTime(realtime,
                                                          season_clim_99_11, mm)

    # Full
    season_total = xr.concat([season_hindcast_detrend,
                              season_realtime_detrend], dim='time')

    return season_total
# ---------------------------------------------------------------------------- #
# DMI, N34
hindcast = OpenVariablesCFSv2(variable='sst', dir=dir_hc, lat=[-80,20])
realtime = OpenVariablesCFSv2(variable='sst', dir=dir_rt, lat=[-80,20])

hindcast = hindcast.rolling(time=3, center=True).mean()
realtime = realtime.rolling(time=3, center=True).mean()

hincast_1982_1998, hindcast_1999_2011 = TwoHindcast(hindcast)

seasons_name = ['MJJ', 'JJA', 'JAS', 'ASO', 'OND']
seasons_mm = [6,7,8,9,11]

for mm, s_name in zip(seasons_mm, seasons_name):

    season_set = SetAnoms(hincast_1982_1998, hindcast_1999_2011, realtime, mm)
    n34_season = season_set.sel(lat=slice(-5, 5),
                                lon=slice(190, 240)).mean(['lon', 'lat'])

    iodw = season_set.sel(lon=slice(50, 70),
                          lat=slice(-10, 10)).mean(['lon', 'lat'])
    iode = season_set.sel(lon=slice(90, 110),
                          lat=slice(-10, 0)).mean(['lon', 'lat'])
    dmi_season = iodw - iode

    # Save files
    if save:
        n34_season.to_netcdf(f'{out_dir}N34_{s_name}_Leads_r_CFSv2.nc')
        dmi_season.to_netcdf(f'{out_dir}DMI_{s_name}_Leads_r_CFSv2.nc')

    del n34_season
    del dmi_season
    del iodw
    del iode
    del season_set
# ---------------------------------------------------------------------------- #
# U50
hindcast = OpenVariablesCFSv2(variable='U50', dir=dir_hc)
realtime = OpenVariablesCFSv2(variable='U50', dir=dir_rt)

hindcast = hindcast.rolling(time=3, center=True).mean()
realtime = realtime.rolling(time=3, center=True).mean()

seasons_name = ['MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND']
seasons_mm = [6, 7, 8, 9, 10, 11]

seasons_name = ['SON', 'OND']
seasons_mm = [10, 11]

hincast_1982_1998, hindcast_1999_2011 = TwoHindcast(hindcast)

for mm, s_name in zip(seasons_mm, seasons_name):

    season_set = SetAnoms(hincast_1982_1998, hindcast_1999_2011, realtime, mm)
    u50 = season_set.mean('lon')

    # Save files
    if save:
        u50.to_netcdf(f'{out_dir}U50_{s_name}_Leads_r_CFSv2.nc')

    del season_set
    del u50

# ---------------------------------------------------------------------------- #
# SAM ya creado, solo seleccionar meses y seasons
# A partir de hgt_mon_anom_d.nc creado por CFSv2_HGT_PreSelect.py con anomalias
# por mes
# Luego calculo del indice SAM --> CFSv2_SAMIndex.py

# path = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/eof/'
# sam_index = xr.open_dataset(path + 'sam_rmon_r_z200.nc')
#
# sam_index = sam_index.rename({'pcs': 'sam'})
# mm_seasons = [[10, 9, 8, 7], [9, 8, 7, 6], [8, 7, 6, 5]]
# for s in mm_seasons:
#     for l, mm in zip([0, 1, 2, 3], [10, 9, 8, 7]):
#         aux = sam_index.sel(time2=sam_index.time2.dt.month.isin(mm), L=l)
#         aux = aux.assign_coords({'L': l})
#
#         if l == 0:
#             sam_season = aux
#         else:
#             sam_season = xr.concat([sam_season, aux], dim='time2')
#
#         sam_season = sam_season.drop(['mode', 'month'])
#         sam_season = sam_season.rename({'time2': 'time'})
#         # sam_season.to_netcdf(out_dir2 + 'SAM_SON_Leads_r_CFSv2.nc')


# ---------------------------------------------------------------------------- #
