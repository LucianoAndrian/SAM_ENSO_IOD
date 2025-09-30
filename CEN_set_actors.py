"""
Seteos de indices para CEN
"""
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import xarray as xr
import pandas as pd
pd.options.mode.chained_assignment = None
from funciones.indices import Nino34CPC, DMI2
from cen.cen_funciones import Detrend, set_data_to_cen

# ---------------------------------------------------------------------------- #
# set data
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
dir_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_no_detrend/'

# indices -------------------------------------------------------------------- #
# DMI
dmi_or = DMI2(filter_bwa=False, start_per='1959', end_per='2020',
              sst_anom_sd=False, opposite_signs_criteria=False)[2]

# N34
sst_aux = xr.open_dataset(
    '/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc')
sst_aux = sst_aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34_or = Nino34CPC(sst_aux, start=1920, end=2020)[0]

# U50
u50_dir_file = '/pikachu/datos/luciano.andrian/observado/' \
          'ncfiles/ERA5/downloaded/ERA5_U50hpa_40-20.mon.nc'
u50_or = set_data_to_cen(u50_dir_file, interp_2x2=False, select_lat=[-60],
                         rolling=True, rl_win=3, purge_extra_dims=True)
u50_or = Detrend(u50_or, 'time')
u50_or = u50_or.mean('lon')
u50_or = xr.DataArray(u50_or['var'].drop('lat'))
# ---------------------------------------------------------------------------- #