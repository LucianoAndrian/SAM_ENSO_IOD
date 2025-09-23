"""
Campos e indices clasificados a partir de los eventos seleccionados para
cada indice
"""
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/events_variables/'

import xarray as xr
from funciones.SelectVariables_utils import parallel_SelectVariables
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# ---------------------------------------------------------------------------- #
events_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/events/'
data_dir_indices = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'

files = os.listdir(events_dir)
files = [f for f in files if f.endswith('.nc')]
div_files = len(files) // 2 # por memoria del servidor

# Indices -------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
for i in ['DMI', 'N34', 'SAM']:
    variable_file = f'{i}_SON_Leads_r_CFSv2.nc'
    parallel_SelectVariables(files, variable_file, div_files,
                             data_dir=data_dir_indices,
                             cases_dir=events_dir,
                             out_dir=out_dir)


