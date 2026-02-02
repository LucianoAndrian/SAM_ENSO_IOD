"""
Campos e indices clasificados a partir de los eventos seleccionados para
cada indice
"""
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/events_variables/'

import os
import glob
from funciones.SelectVariables_utils import parallel_SelectVariables
from funciones.utils import init_logger
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# ---------------------------------------------------------------------------- #
events_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/events/'
data_dir_indices = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'

# ---------------------------------------------------------------------------- #
logger = init_logger('CFSv2_4_SelectVariables.log')

# ---------------------------------------------------------------------------- #
files = os.listdir(events_dir)
files = [f for f in files if f.endswith('.nc')]
div_files = len(files) // 2 # por memoria del servidor

# Indices -------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
logger.info('Indices')
for i in ['DMI', 'N34', 'SAM']:
    logger.debug(f'Indice: {i}')
    variable_file = f'{i}_SON_Leads_r_CFSv2.nc'

    var_prefix = variable_file.split('_')[0]
    existing_files = glob.glob(os.path.join(out_dir, f"{var_prefix}_*"))

    if existing_files:
        logger.warning(f"[SKIP] Ya existen archivos {out_dir} que empiezan con "
              f"'{var_prefix}_'.")
        continue
    else:
        logger.info(f"[RUN] Procesando {variable_file}...")
        parallel_SelectVariables(files, variable_file, div_files,
                                 data_dir=data_dir_indices,
                                 cases_dir=events_dir,
                                 out_dir=out_dir)


# Variables ------------------------------------------------------------------ #
logger.info('Variables')
data_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
for variable_file in ['hgt_son.nc', 'hgt750_son_detrend.nc',
                      'prec_son.nc', 'tref_son.nc']:
    logger.debug(f'Variable: {i}')

    var_prefix = variable_file.split('_')[0]
    existing_files = glob.glob(os.path.join(out_dir, f"{var_prefix}_*"))

    if existing_files:
        logger.warning(f"[SKIP] Ya existen archivos {out_dir} que empiezan con "
              f"'{var_prefix}_'.")
        continue
    else:
        logger.info(f"[RUN] Procesando {variable_file}...")
        parallel_SelectVariables(files, variable_file, div_files,
                                 data_dir=data_dir,
                                 cases_dir=events_dir,
                                 out_dir=out_dir)

# ---------------------------------------------------------------------------- #