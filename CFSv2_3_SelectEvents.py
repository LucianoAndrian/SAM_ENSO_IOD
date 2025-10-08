"""
Select Events DMI, N34 y SAM
"""
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/events/'
save = True

# ---------------------------------------------------------------------------- #
import xarray as xr
from funciones.SelectEvents_utils import main_SelectEvents

# ---------------------------------------------------------------------------- #
data_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'

# Funciones ------------------------------------------------------------------ #
def OpenAndSetIndice(dates_dir, name_file):
    indice = xr.open_dataset(dates_dir + name_file)
    indice_std = indice.std(['r', 'time'])
    indice = indice/indice_std
    return indice

# ---------------------------------------------------------------------------- #
dmi = OpenAndSetIndice(data_dir, 'DMI_SON_Leads_r_CFSv2.nc')
n34 = OpenAndSetIndice(data_dir, 'N34_SON_Leads_r_CFSv2.nc')
sam = OpenAndSetIndice(data_dir, 'SAM_SON_Leads_r_CFSv2.nc')


variables = ['dmi', 'n34', 'sam']
main_SelectEvents(variables, ds1=dmi, ds2=n34, ds3=sam, thr=0.5,
                  save=save, out_dir=out_dir,
                  season_name='SON',  prefix_file='CFSv2')

# ---------------------------------------------------------------------------- #