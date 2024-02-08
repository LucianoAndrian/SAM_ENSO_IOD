"""
Similar al SelectEvents de ENSO_IOD
"""
# ---------------------------------------------------------------------------- #
save = False
# ---------------------------------------------------------------------------- #
path = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/eof/'
path_aux = '/pikachu/datos/luciano.andrian/cases_fields/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/events_dates/'
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import asyncio # **
# **
# La selección de los campos en hgt tarda mucho debido a la cantidad >10mil
# en general para cada categoria.
# Al hacerlo de manera asincronica en paralelo es 7 veces mas rapido.
# No se con cuantos hilos opera, en htop no se ve. Ni como configurarlos7
# consumo de RAM, bajo
# ---------------------------------------------------------------------------- #
sam_index = xr.open_dataset(path + 'sam_rmon_r_z200.nc').\
    rename({'time2':'time'})
hgt = xr.open_dataset(path_aux + 'hgt_mon_anom_d.nc')
# ---------------------------------------------------------------------------- #
def SelectEvents(indice, operador, umbral='0', abs_val=False):
    runs = np.arange(1, 25)
    dates = indice.time.values
    leads = [0, 1, 2, 3]

    first = True
    for l in leads:
        for r in runs:
            for d in dates:
                aux = sam_index.sel(L=l, r=r, time=d)
                expresion = f"aux.pcs.values {operador} {umbral}"
                if abs_val:
                    expresion = f"np.abs(aux.pcs.values) {operador} {umbral}"
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

async def SelectFields(event_time):
    data_field = hgt

    d = event_time[0]
    l = event_time[1]
    r = event_time[2]

    aux = data_field.sel(time=data_field['L']==l)
    aux = aux.sel(time=d, r=r)

    return aux

def SetOrder(dataset):
    t = dataset['time'].values
    L = dataset['L'].values
    r = dataset['r'].values

    values = [[t[i], L[i], r[i]] for i in range(len(t))]

    return values

async def main(index_selected):
    datos = SetOrder(index_selected)
    tareas = [SelectFields(event_time) for event_time in datos]

    resultados = await asyncio.gather(*tareas)

    return resultados
# ---------------------------------------------------------------------------- #
# Determinar categorias <, >, "neutro" y umbrales.
sam_neutro = SelectEvents(sam_index, '<', '0.5', True)
resultados = asyncio.run(main(sam_neutro))
aux_xr = xr.concat(resultados, dim='time')
if save:
    aux_xr.to_netcdf(out_dir + 'sam_neutro_hgt200.nc')
    sam_neutro.to_netcdf(out_dir + 'sam_neutro.nc')

sam_pos = SelectEvents(sam_index, '>', '0.5')
resultados = asyncio.run(main(sam_pos))
aux_xr = xr.concat(resultados, dim='time')
if save:
    aux_xr.to_netcdf(out_dir + 'sam_pos_hgt200.nc')
    sam_pos.to_netcdf(out_dir + 'sam_pos.nc')

sam_neg = SelectEvents(sam_index, '<', '-0.5')
resultados = asyncio.run(main(sam_neg))
aux_xr = xr.concat(resultados, dim='time')
if save:
    aux_xr.to_netcdf(out_dir + 'sam_neg_hgt200.nc')
    sam_neg.to_netcdf(out_dir + 'sam_neg.nc')

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #