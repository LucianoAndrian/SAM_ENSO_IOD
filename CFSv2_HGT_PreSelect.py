"""
IDEM ENSO-IOD: En este caso solo es necesario para calcular el indice SAM
Para los composites y demás USAR LOS ARCHIVOS DE ENSO-IOD

Pre-procesamiento HGT200
Anomalías respecto a la climatologia del hindcast y detrend de las anomalias
(similar a ENSO_IOD_fixCFSv2_DMI_N34.py)
"""
################################################################################
import xarray as xr
import numpy as np
from ENSO_IOD_Funciones import SelectNMMEFiles
################################################################################
dir_hc = '/pikachu/datos/luciano.andrian/hindcast/'
dir_rt = '/pikachu/datos/luciano.andrian/real_time/'
out_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
v = 'hgt'
# Funciones ####################################################################
def fix_calendar(ds, timevar='time'):
    """
    agrega los dias a los archivos nc de NMME
    """
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'
    return ds

def TwoClim_MonthlyAnom(data_1982_1998, data_1999_2011):
    for l in [0,1,2,3]: # leads
        months_1982_1998 = data_1982_1998.sel(L=l).groupby('time.month')
        months_1999_2011 = data_1999_2011.sel(L=l).groupby('time.month')

        if l==0:
            # Climatologia ----------------------------------------------------#
            months_clim_1982_1998 = months_1982_1998.mean(['r', 'time'])
            months_clim_1999_2011 = months_1999_2011.mean(['r', 'time'])

            # Anomalia---------------------------------------------------------#
            months_anom_1982_1998 = months_1982_1998 - months_clim_1982_1998
            months_anom_1999_2011 = months_1999_2011 - months_clim_1999_2011
        else:
            # Climatologia ----------------------------------------------------#
            aux_months_clim_1982_1998 = months_1982_1998.mean(['r', 'time'])
            aux_months_clim_1999_2011 = months_1999_2011.mean(['r', 'time'])
            months_clim_1982_1998 = xr.concat(
                [months_clim_1982_1998, aux_months_clim_1982_1998], dim='L')
            months_clim_1999_2011 = xr.concat(
                [months_clim_1999_2011, aux_months_clim_1999_2011], dim='L')

            # Anomalia---------------------------------------------------------#
            aux_1982_1998 = months_1982_1998 - aux_months_clim_1982_1998
            aux_1999_2011 = months_1999_2011 - aux_months_clim_1999_2011

            months_anom_1982_1998 = xr.concat(
                [months_anom_1982_1998, aux_1982_1998], dim='time')
            months_anom_1999_2011 = xr.concat(
                [months_anom_1999_2011, aux_1999_2011], dim='time')


    return months_clim_1982_1998, months_clim_1999_2011, \
           months_anom_1982_1998, months_anom_1999_2011


def Detrend_Seasons(anom_82_98, anom_99_11):
    for l in [0, 1, 2, 3]:
        # 1982-1998 -----------------------------------------------------------#
        aux_anom_1982_1998 = anom_82_98.sel(time=anom_82_98['L'] == l)

        aux = aux_anom_1982_1998.mean('r').polyfit(dim='time', deg=1)
        aux_trend = xr.polyval(aux_anom_1982_1998['time'],
                               aux.hgt_polyfit_coefficients)
        if l == 0:
            anom_1982_1998_detrened = aux_anom_1982_1998 - aux_trend
        else:
            aux_detrend = aux_anom_1982_1998 - aux_trend
            anom_1982_1998_detrened = xr.concat(
                [anom_1982_1998_detrened, aux_detrend], dim='time')

        # 1999-2011 -----------------------------------------------------------#
        aux_anom_1999_2011 = anom_99_11.sel(time=anom_99_11['L'] == l)

        aux = aux_anom_1999_2011.mean('r').polyfit(dim='time', deg=1)
        aux_trend = xr.polyval(aux_anom_1999_2011['time'],
                               aux.hgt_polyfit_coefficients)
        if l == 0:
            anom_1999_2011_detrend = aux_anom_1999_2011 - aux_trend
        else:
            aux_detrend = aux_anom_1999_2011 - aux_trend
            anom_1999_2011_detrend = \
                xr.concat([anom_1999_2011_detrend, aux_detrend], dim='time')


    return anom_1982_1998_detrened, anom_1999_2011_detrend


def MonthlyAnom_Detrend_RealTime(data_realtime, clim_1999_2011):

    for l in [0,1,2,3]:
        aux = data_realtime.sel(L=l).groupby('time.month')
        aux_clim_1999_2011 = clim_1999_2011.sel(L=l)

        #Anomalia -------------------------------------------------------------#
        anom = aux - aux_clim_1999_2011

        #Detrend --------------------------------------------------------------#
        aux_dt = anom.mean('r').polyfit(dim='time', deg=1)
        aux_trend = xr.polyval(anom['time'], aux_dt.hgt_polyfit_coefficients)

        if l==0:
            anom_detrend = anom - aux_trend
        else:
            aux_detrend = anom - aux_trend
            anom_detrend = xr.concat([anom_detrend, aux_detrend],  dim='time')

    return anom_detrend

################################################################################
# usando SelectNMMEFiles con All=True,
# abre TODOS los archivos .nc de la ruta en dir
# HINDCAST --------------------------------------------------------------------#
files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                        dir=dir_hc, All=True)
files = sorted(files, key=lambda x: x.split()[0])

#abriendo todos los archivos
#xr no entiende la codificacion de Leads, r y las fechas
data = xr.open_mfdataset(files, decode_times=False)
data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
data = data.sel(L=[0.5, 1.5, 2.5, 3.5]) # Solo leads 0 1 2 3
data['L'] = [0,1,2,3]
data = xr.decode_cf(fix_calendar(data)) # corrigiendo fechas
data = data.sel(lat=slice(-90, -20), P=200)
data = data.drop('P')
#data = data.drop('Z')

# ESTO AHORA NO, SAM mensual luego la media mensual.
#media movil de 3 meses para separar en estaciones
#data = data.rolling(time=3, center=True).mean()

# 1982-1998, 1999-2011
data_1982_1998 = data.sel(
    time=data.time.dt.year.isin(np.linspace(1982,1998,17)))
data_1999_2011 = data.sel(
    time=data.time.dt.year.isin(np.linspace(1999,2011,13)))

# Climatologias y anomalias detrend -------------------------------------------#
#------------------------------------------------------------------------------#
clim_82_98, clim_99_11, anom_82_98, anom_99_11 = \
    TwoClim_MonthlyAnom(data_1982_1998, data_1999_2011)

anom_82_98, anom_99_11 = Detrend_Seasons(anom_82_98, anom_99_11)

hindcast_detrend = xr.concat([anom_82_98, anom_99_11], dim='time')

clim_99_11 = clim_99_11.load()
#------------------------------------------------------------------------------#
# Real-time -------------------------------------------------------------------#
files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                        dir=dir_rt, All=True)
files = sorted(files, key=lambda x: x.split()[0])

#abriendo todos los archivos
#xr no entiende la codificacion de Leads, r y las fechas
data = xr.open_mfdataset(files, decode_times=False)
data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
data = data.sel(L=[0.5, 1.5, 2.5, 3.5]) # Solo leads 0 1 2 3
data['L'] = [0,1,2,3]
data = xr.decode_cf(fix_calendar(data)) # corrigiendo fechas
data = data.sel(lat=slice(-90, -20), P=200)
data = data.drop('P')
data = data.drop('Z')

#media movil de 3 meses para separar en estaciones
#data = data.rolling(time=3, center=True).mean()

#- Anomalias detrend por seasons --------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
realtime_detrend = MonthlyAnom_Detrend_RealTime(data, clim_99_11)


hgt_f = xr.concat([hindcast_detrend, realtime_detrend], dim='time')

# save ----------------------------------------------------------------------------------------------------------------#
# jja_f.to_netcdf(out_dir + 'hgt_jja.nc')
# jas_f.to_netcdf(out_dir + 'hgt_jas.nc')
# aso_f.to_netcdf(out_dir + 'hgt_aso.nc')
hgt_f.to_netcdf(out_dir + 'hgt_mon_anom_d.nc')
########################################################################################################################