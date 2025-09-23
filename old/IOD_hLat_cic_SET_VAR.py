"""
SET VARIABLES: IOD - SAM/U50
"""
# ---------------------------------------------------------------------------- #
hgt_vs_p = False
use_pp = False
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import xarray as xr
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

from ENSO_IOD_Funciones import (DMI2, DMI2_singlemonth, DMI2_twomonths,
                                ChangeLons, SameDateAs)
from cen_funciones import Detrend, Weights, OpenObsDataSet
# ---------------------------------------------------------------------------- #
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
hgt_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
dir_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_no_detrend/'
# ---------------------------------------------------------------------------- #
def convertdates(dataarray, dimension, rename=None):
    fechas = pd.to_datetime(dataarray[dimension].values.astype(str),
                            format='%Y%m%d')
    dataarray[dimension] = fechas
    if rename is not None:
        dataarray = dataarray.rename({dimension: rename})
    return dataarray

def ddn(serie, serie_ref):
    """
    SameDateAs, Detrned, Norm.

    :param serie: xr.dataset or xr.datarray o lista de ellos
    :param serie_ref: xr.dataset or xr.datarray
    :return: xr.dataarray
    """
    output = None
    if isinstance(serie, list):
        output = []
        for s in serie:
            serie_sd = SameDateAs(s, serie_ref)
            serie_sd_d = Detrend(serie_sd, 'time')
            # Da medio igual la resta de la media xq son todas anomalias
            serie_sd_d_n = ((serie_sd_d - serie_sd_d.mean('time')) /
                            serie_sd_d.std('time'))
            output.append(serie_sd_d_n)
        output = tuple(output)
    else:
        serie = SameDateAs(serie, serie_ref)
        serie = Detrend(serie, 'time')
        # Da medio igual la resta de la media xq son todas anomalias
        serie = (serie - serie.mean('time')) / serie.std('time')
        output = serie

    return output

def SelectMonths(data, months_to_select, years_to_remove=None):
    """
    Selecciona meses
    :param series: xr.dataset o xr.dataarray o lista de ellos
    :param months_to_select: int o lista de meses
    :param years_to_remove: int o lista de años a quitar, default None
    :return: data con meses seleccionados
    """
    output = None
    if isinstance(data, list):
        output = []
        for d in data:
            d_aux = d.sel(time=d.time.dt.month.isin(months_to_select))
            if years_to_remove is not None:
                d_aux = d_aux.sel(time=~d_aux.time.dt.year.isin(years_to_remove))
            output.append(d_aux)
        output = tuple(output)
    else:
        output = data.sel(time=data.time.dt.month.isin(months_to_select))
        output = output.sel(time=~output.time.dt.year.isin(years_to_remove))

    return output

# ---------------------------------------------------------------------------- #
def compute():
    years_to_remove = [2002, 2019]
    # HGT -------------------------------------------------------------------- #
    hgt = xr.open_dataset(hgt_dir + 'ERA5_HGT200_40-20.nc')
    hgt = hgt.rename({'longitude': 'lon', 'latitude': 'lat', 'z': 'var'})
    hgt = hgt.interp(lon=np.arange(0, 360, 2), lat=np.arange(-90, 90, 2))
    hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
    hgt200_anom_or = (hgt.groupby('time.month') -
                      hgt_clim.groupby('time.month').mean('time'))
    weights = np.sqrt(np.abs(np.cos(np.radians(hgt200_anom_or.lat))))
    hgt200_anom_or_1rm = hgt200_anom_or * weights
    hgt200_anom_or_1rm = hgt200_anom_or_1rm.sel(
        time=~hgt200_anom_or_1rm.time.dt.year.isin(years_to_remove))
    hgt200_anom_or_1rm = hgt200_anom_or_1rm.sel(lat=slice(-80,20))
    hgt200_anom_or_2rm = hgt200_anom_or_1rm.rolling(time=2, center=True).mean()
    hgt200_anom_or_3rm = hgt200_anom_or_1rm.rolling(time=3, center=True).mean()


    weights = np.sqrt(np.abs(np.cos(np.radians(hgt.lat))))
    hgt_1rm = hgt * weights
    hgt_1rm = hgt_1rm.sel(
        time=~hgt_1rm.time.dt.year.isin(years_to_remove))
    hgt_1rm = hgt_1rm.sel(lat=slice(-80,20))
    hgt_2rm = hgt_1rm.rolling(time=2, center=True).mean()
    hgt_3rm = hgt_1rm.rolling(time=3, center=True).mean()

    #
    # if hgt_vs_p:
    #     hgt_lvls = xr.open_dataset('/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/'
    #                                'ERA5_HGT500-10_79-20.mon.nc')
    #     hgt_lvls = convertdates(hgt_lvls, 'date', 'time')
    #     hgt_lvls = ChangeLons(hgt_lvls, 'longitude')
    #     hgt_lvls = hgt_lvls.rename({'latitude': 'lat', 'z': 'var'})
    #     hgt_lvls = hgt_lvls.drop('expver')
    #     hgt_lvls = hgt_lvls.drop('number')
    #
    #     # Esto tarda mucho y es mucho peor cuando se selecciona antes una region
    #     # mas chica de longitud.
    #     # Va por niveles xq ocupa menos ram
    #     print('interp...')
    #     first = True
    #     for l in hgt_lvls.pressure_level.values:
    #         print(l)
    #         aux = hgt_lvls.sel(pressure_level=l)
    #
    #         aux = aux.interp(lon=np.arange(hgt_lvls.lon.values[0],
    #                                        hgt_lvls.lon.values[-1] + 1, 1),
    #                          lat=np.arange(hgt_lvls.lat.values[-1],
    #                                        hgt_lvls.lat.values[0] + 1, 1)[::-1])
    #         if first:
    #             first = False
    #             hgt_lvls_interp = aux
    #         else:
    #             hgt_lvls_interp = xr.concat([hgt_lvls_interp, aux],
    #                                         dim='pressure_level')
    #
    # # PP ----------------------------------------------------------------- #
    # if use_pp:
    #     pp_or = OpenObsDataSet(name='pp_pgcc_v2020_1891-2023_1', sa=True,
    #                            dir=dir_pp)
    #     pp_or = pp_or.rename({'precip': 'var'})
    #     pp_or = pp_or.sel(time=slice('1959-01-16', '2020-12-16'))
    #
    #     pp_or = Weights(pp_or)
    #     pp_or = pp_or.sel(lat=slice(20, -60), lon=slice(270, 330))  # SA

    # ------------------------------------------------------------------------ #
    # Indices
    # ------------------------------------------------------------------------ #
    dmi_or_1rm = \
    DMI2_singlemonth(filter_bwa=False, start_per='1959', end_per='2020',
                     sst_anom_sd=False, opposite_signs_criteria=False)[2]
    dmi_or_2rm = \
    DMI2_twomonths(filter_bwa=False, start_per='1959', end_per='2020',
                   sst_anom_sd=False, opposite_signs_criteria=False)[2]
    dmi_or_3rm = DMI2(filter_bwa=False, start_per='1959', end_per='2020',
                      sst_anom_sd=False, opposite_signs_criteria=False)[2]

    dmi_or_1rm = dmi_or_1rm.sel(
        time=~dmi_or_1rm.time.dt.year.isin(years_to_remove))
    dmi_or_2rm = dmi_or_2rm.sel(
        time=~dmi_or_2rm.time.dt.year.isin(years_to_remove))
    dmi_or_3rm = dmi_or_3rm.sel(
        time=~dmi_or_3rm.time.dt.year.isin(years_to_remove))

    sam_or_1rm = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
    sam_or_2rm = sam_or_1rm.rolling(time=2, center=True).mean()
    sam_or_3rm = sam_or_1rm.rolling(time=3, center=True).mean()
    sam_or_3rm[-1] = 0

    sam_or_1rm = sam_or_1rm.sel(
        time=~sam_or_1rm.time.dt.year.isin(years_to_remove))
    sam_or_2rm = sam_or_2rm.sel(
        time=~sam_or_2rm.time.dt.year.isin(years_to_remove))
    sam_or_3rm = sam_or_3rm.sel(
        time=~sam_or_3rm.time.dt.year.isin(years_to_remove))


    u50_or = xr.open_dataset('/pikachu/datos/luciano.andrian/observado/'
                             'ncfiles/ERA5/downloaded/ERA5_U50hpa_40-20.mon.nc')
    u50_or = u50_or.rename({'u': 'var'})
    u50_or = u50_or.rename({'longitude': 'lon'})
    u50_or = u50_or.rename({'latitude': 'lat'})
    u50_or = Weights(u50_or)
    u50_or = u50_or.sel(lat=-60)
    u50_or = u50_or.sel(expver=1).drop('expver')

    u50_or = (u50_or.groupby('time.month') -
              u50_or.groupby('time.month').mean('time'))

    u50_or = u50_or.sel(
        time=~u50_or.time.dt.year.isin(years_to_remove))

    u50_or_1rm = u50_or.mean('lon')
    u50_or_2rm = u50_or.rolling(time=2, center=True).mean()
    u50_or_2rm = u50_or_2rm.mean('lon')
    u50_or_3rm = u50_or.rolling(time=3, center=True).mean()
    u50_or_3rm = u50_or_3rm.mean('lon')

    # ------------------------------------------------------------------------ #

    (dmi_or_1rm, dmi_or_2rm, dmi_or_3rm, sam_or_1rm, sam_or_2rm, sam_or_3rm,
     u50_or_1rm, u50_or_2rm, u50_or_3rm, hgt200_anom_or_1rm, hgt200_anom_or_2rm,
     hgt200_anom_or_3rm) = ddn([dmi_or_1rm, dmi_or_2rm, dmi_or_3rm, sam_or_1rm,
                                sam_or_2rm, sam_or_3rm, u50_or_1rm, u50_or_2rm,
                                u50_or_3rm, hgt200_anom_or_1rm,
                                hgt200_anom_or_2rm,
                                hgt200_anom_or_3rm], dmi_or_1rm)

    return (dmi_or_1rm, dmi_or_2rm, dmi_or_3rm, sam_or_1rm, sam_or_2rm,
            sam_or_3rm,u50_or_1rm, u50_or_2rm, u50_or_3rm, hgt200_anom_or_1rm,
            hgt200_anom_or_2rm,hgt200_anom_or_3rm, hgt_1rm, hgt_2rm, hgt_3rm)

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    (dmi_or_1rm, dmi_or_2rm, dmi_or_3rm, sam_or_1rm, sam_or_2rm, sam_or_3rm,
     u50_or_1rm, u50_or_2rm, u50_or_3rm, hgt200_anom_or_1rm, hgt200_anom_or_2rm,
     hgt200_anom_or_3rm, hgt_1rm, hgt_2rm, hgt_3rm) = compute()
# ---------------------------------------------------------------------------- #