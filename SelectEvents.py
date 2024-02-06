"""
Similar al SelectEvents de ENSO_IOD
"""
# ---------------------------------------------------------------------------- #
save = False
# ---------------------------------------------------------------------------- #

path = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/eof/'
path_aux = '/pikachu/datos/luciano.andrian/cases_fields/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/events_dates/'

if save:
    dpi=300
else:
    dpi=100

# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
# ---------------------------------------------------------------------------- #

sam_index = xr.open_dataset(path + 'sam_rmon_r_z200.nc').\
    rename({'time2':'time'})
hgt = xr.open_dataset(path_aux + 'hgt_mon_anom_d.nc')

# ---------------------------------------------------------------------------- #

def SelectEvents(indice, operador, umbral='0'):
    runs = np.arange(1, 25)
    dates = indice.time.values
    leads = [0, 1, 2, 3]

    first = True
    for l in leads:
        for r in runs:
            for d in dates:
                aux = sam_index.sel(L=l, r=r, time=d)
                expresion = f"aux.pcs.values {operador} {umbral}"
                if eval(expresion):
                    aux = aux.assign_coords(L=l)
                    if first:
                        first = False
                        sam_selected = aux
                    else:
                        sam_selected = xr.concat([sam_selected, aux],
                                                 dim='time')

    sam_selected = sam_selected.rename({'pcs':'sam'})
    sam_selected = sam_selected.drop('mode')

    return sam_selected

# ---------------------------------------------------------------------------- #
sam_pos = SelectEvents(sam_index, '>', '0')
sam_neg = SelectEvents(sam_index, '<', '0')
sam_p1 = SelectEvents(sam_index, '>', '1')
sam_n1 = SelectEvents(sam_index, '<', '-1')
# ---------------------------------------------------------------------------- #
#if save:

# ---------------------------------------------------------------------------- #