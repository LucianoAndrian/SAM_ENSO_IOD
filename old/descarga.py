import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels-monthly-means',
    {
        'format': 'netcdf',
        'variable': 'u_component_of_wind',
        'pressure_level': '50',
        'year': [
            '1940', '1941', '1942',
            '1943', '1944', '1945',
            '1946', '1947', '1948',
            '1949', '1950', '1951',
            '1952', '1953', '1954',
            '1955', '1956', '1957',
            '1958', '1959', '1960',
            '1961', '1962', '1963',
            '1964', '1965', '1966',
            '1967', '1968', '1969',
            '1970', '1971', '1972',
            '1973', '1974', '1975',
            '1976', '1977', '1978',
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021', '2022', '2023',
            '2024',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'product_type': 'monthly_averaged_reanalysis',
        'time': '00:00',
        "grid": "1.0/1.0",
    },
    '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
    'ERA5_U50hpa_40-20.mon.nc')
###

import xarray as xr
import numpy as np
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
###############################################################################
#dir_files = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/merged/'
#out_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/mer_d_w/'
dir_files = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
out_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
# Funciones ###################################################################
def Detrend(xrda, dim):
    aux = xrda.polyfit(dim=dim, deg=1)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients)
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients)
    dt = xrda - trend
    return dt

def Weights(data):
    weights = np.transpose(np.tile(np.cos(data.lat * np.pi / 180),
                                   (len(data.lon), 1)))
    data_w = data * weights
    return data_w

###############################################################################
# U V va por RWS_makerterms.py

data = xr.open_dataset(dir_files + 'ERA5_U50hpa_40-20.mon.nc')
data = data.rename({u: 'var'})
data = data.rename({'longitude': 'lon'})
data = data.rename({'latitude': 'lat'})

data = Weights(data)
data = data.sel(lat=slice(0, -90))
#data = data.rolling(time=3, center=True).mean()
#for mm, s_name in zip([7], ['JJA']):
#for mm, s_name in zip([10], ['SON']): # main month seasons
for mm, s_name in zip([7, 10], ['JJA', 'SON']):
    aux = data.sel(time=data.time.dt.month.isin(mm))
    aux = Detrend(aux, 'time')

    print('to_netcdf...')
    if v == 'UV200':
        aux.to_netcdf(out_dir + n_v + '_' + s_name + '_mer_d_w.nc')
    else:
        aux.to_netcdf(out_dir + v + '_' + s_name + '_mer_d_w.nc')
