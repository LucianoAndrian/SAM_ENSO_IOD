"""
Calculo de Niño3.4 y DMI para el CFSv2 con leads 0 1 2 3
Climatologías 1982-1998, 1999-2011 (última también para real-time)
y detrend en cada período
"""
########################################################################################################################
import xarray as xr
import numpy as np
from ENSO_IOD_Funciones import SelectNMMEFiles
# Funciones ############################################################################################################
def fix_calendar(ds, timevar='time'):
    """
    agrega los dias a los archivos nc de NMME
    """
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'
    return ds


def SelectSeason(data, main_month_season):
    first = True
    for l in [0,1,2,3]:
        season_data = data.sel(
            time=data.time.dt.month.isin(main_month_season-l), L=l)
        if first:
            first = False
            season = season_data
        else:
            season = xr.concat([season, season_data], dim='time')
    return season

########################################################################################################################
dir_hc = '/pikachu/datos/luciano.andrian/hindcast/'
dir_rt = '/pikachu/datos/luciano.andrian/real_time/'
out_dir = '/datos/luciano.andrian/ncfiles/NMME_CFSv2/DMI_N34_Leads_r/'
v = 'hgt'
########################################################################################################################
### HINDCAST ###
files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                        dir=dir_hc, All=True)
files = sorted(files, key=lambda x: x.split()[0])

#abriendo todos los archivos
data = xr.open_mfdataset(files, decode_times=False) #xr no entiende la codificacion de Leads, r y las fechas
data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
data = data.sel(L=[0.5, 1.5, 2.5, 3.5]) # Solo leads 0 1 2 3
data['L'] = [0,1,2,3]
data = xr.decode_cf(fix_calendar(data)) # corrigiendo fechas
data = data.sel(lat=slice(-80, 20), P=200)
data = data.drop('P')
#media movil de 3 meses para separar en estaciones
data = data.rolling(time=3, center=True).mean()

son = SelectSeason(data, 10)

### REAL_TIME ###
# en este caso se usa la climatologia 1999-2011 calculada antes
files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                        dir=dir_rt, All=True)
files = sorted(files, key=lambda x: x.split()[0])

#abriendo todos los archivos
data = xr.open_mfdataset(files, decode_times=False) #xr no entiende la codificacion de Leads, r y las fechas
data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
data = data.sel(L=[0.5, 1.5, 2.5, 3.5]) # Solo leads 0 1 2 3
data['L'] = [0,1,2,3]
data = xr.decode_cf(fix_calendar(data)) # corrigiendo fechas
data = data.sel(lat=slice(-80, 20), P=200)
data = data.drop('P')
data = data.drop('Z')

#media movil de 3 meses para separar en estaciones
data = data.rolling(time=3, center=True).mean()
son_rt = SelectSeason(data, 10)




son_f = xr.concat([son, son_rt], dim='time')
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cases_fields/'
son_f.to_netcdf(out_dir + 'hgt_no_anom_SON.nc')

cases_date_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/' \
                 'events_dates/'

from ENSO_IOD_Funciones import SelectVariables

def SelectEventsHGT(c):
    s = 'SON'
    try:
        aux_cases = xr.open_dataset(
            cases_date_dir + c + '_f_' + s + '.nc') \
            .rename({'__xarray_dataarray_variable__': 'index'})
    except:
        aux_cases = xr.open_dataset(
            cases_date_dir + c + '_f_' + s + '.nc') \
            .rename({'sst': 'index'})

    data_hgt_s = son_f

    case_events = SelectVariables(aux_cases, data_hgt_s)
    return case_events

neutros_no_anom_values = SelectEventsHGT('neutros')

out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cases_fields/'
neutros_no_anom_values.to_netcdf(out_dir + 'hgt_neutro_no_anoms_SON.nc')

########################################################################################################################