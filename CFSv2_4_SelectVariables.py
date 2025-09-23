"""
Campos e indices clasificados a partir de los eventos seleccionados para
cada indice
"""
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/events_variables/'

from funciones.SelectVariables_utils import parallel_SelectVariables
import os
import glob
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

# Variables ------------------------------------------------------------------ #
data_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
for variable_file in ['hgt_son.nc', 'hgt750_son_detrend.nc',
                      'prec_son.nc', 'tref_son.nc']:

    var_prefix = variable_file.split('_')[0]
    existing_files = glob.glob(os.path.join(out_dir, f"{var_prefix}_*"))

    if existing_files:
        print(f"[SKIP] Ya existen archivos {out_dir} que empiezan con "
              f"'{var_prefix}_'.")
        continue
    else:
        print(f"[RUN] Procesando {variable_file}...")
        parallel_SelectVariables(files, variable_file, div_files,
                                 data_dir=data_dir,
                                 cases_dir=events_dir,
                                 out_dir=out_dir)

# ---------------------------------------------------------------------------- #