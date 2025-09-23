"""
Funciones para Seleccionar/Clasificar eventos a partir de tres indices
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np

## Funciones ----------------------------------------------------------------- #
def InitEvent_dict(variables):
    def empty(): return xr.Dataset(coords={'time': []})

    index1 = variables[0]
    index2 = variables[1]
    index3 = variables[2]

    signs = ['pos', 'neg']
    dobles = [f'{index1}_{index2}', f'{index1}_{index3}', f'{index2}_{index3}']

    dobles_op = [f'{index1}_pos_{index2}_neg', f'{index1}_neg_{index2}_pos',
                 f'{index1}_pos_{index3}_neg', f'{index1}_neg_{index3}_pos',
                 f'{index2}_pos_{index3}_neg', f'{index2}_neg_{index3}_pos']

    triples = [f'{index1}_{index2}_{index3}']

    triples_op = [f'{index1}_pos_{index2}_pos_{index3}_neg',
                  f'{index1}_pos_{index2}_neg_{index3}_neg',
                  f'{index1}_pos_{index2}_neg_{index3}_pos',
                  f'{index1}_neg_{index2}_pos_{index3}_pos',
                  f'{index1}_neg_{index2}_pos_{index3}_neg',
                  f'{index1}_neg_{index2}_neg_{index3}_pos']

    data = {
        'todo': {i: {s: empty() for s in signs} for i in variables},
        'puros': {i: {s: empty() for s in signs} for i in variables},
        'simultaneos': {
            'dobles': {i: {s: empty() for s in signs} for i in dobles},
            'dobles_op': {i: empty() for i in dobles_op},
            'triples': {i: {s: empty() for s in signs} for i in triples},
            'triples_opuestos': {k: empty() for k in triples_op}
        },
        'neutros': empty()
    }
    return data

def merge_event_dicts(dict1, dict2):
    merged = {}

    def merge_ds(ds1, ds2):
        if ds1 is None:
            return ds2
        if ds2 is None:
            return ds1
        if not hasattr(ds1, 'time') or not hasattr(ds2, 'time'):
            raise TypeError(
                f"Uno de los elementos no tiene atributo `.time`: {type(ds1)}, {type(ds2)}")
        if ds1.time.size == 0:
            return ds2
        if ds2.time.size == 0:
            return ds1
        return xr.concat([ds1, ds2], dim='time')

    for key in dict1:
        if isinstance(dict1[key], dict):
            merged[key] = {}
            for subkey in dict1[key]:
                if isinstance(dict1[key][subkey], dict):
                    merged[key][subkey] = {}
                    for subsubkey in dict1[key][subkey]:
                        val1 = dict1[key][subkey][subsubkey]
                        val2 = dict2[key][subkey][subsubkey]
                        if isinstance(val1, (xr.Dataset, xr.DataArray)):
                            merged[key][subkey][subsubkey] = merge_ds(val1, val2)
                        else:
                            # Nivel más profundo
                            merged[key][subkey][subsubkey] = {}
                            for subsubsubkey in val1:
                                merged[key][subkey][subsubkey][subsubsubkey] = merge_ds(
                                    val1[subsubsubkey], dict2[key][subkey][subsubkey][subsubsubkey]
                                )
                else:
                    merged[key][subkey] = merge_ds(dict1[key][subkey], dict2[key][subkey])
        else:
            merged[key] = merge_ds(dict1[key], dict2[key])
    return merged

def save_event_dict_to_netcdf(event_dict, out_dir, season='', prefix=''):
    """
    Guarda cada elemento del dict anidado de eventos como archivo NetCDF,
    usando la clave como nombre de archivo.

    Args:
        event_dict (dict): Diccionario de eventos anidado.
        out_dir (str): Carpeta de salida.
        season (str): Opcional, nombre de la estación, e.g. 'SON'.
        prefix (str): Prefijo opcional para los nombres de archivo.
    """
    import os

    def recursive_save(d, name_parts):
        for k, v in d.items():
            new_name_parts = name_parts + [k]
            if isinstance(v, (xr.Dataset, xr.DataArray)):
                # Verificar si tiene dimensión 'time' y si está vacío
                if 'time' in v.dims and v.sizes.get('time', 0) == 0:
                    print(f"Skipping save for {'_'.join(new_name_parts)}: Dataset is empty.")
                else:
                    # Generar nombre de archivo
                    filename = '_'.join([prefix] + new_name_parts + [season]).strip('_') + '.nc'
                    filepath = os.path.join(out_dir, filename)
                    print(f"Saving: {filepath}")
                    v.to_netcdf(filepath)
            elif isinstance(v, dict):
                recursive_save(v, new_name_parts)
            else:
                print(f"Skipping unknown type at: {'_'.join(new_name_parts)}")

    recursive_save(event_dict, [])

def CheckNaN_leads(indice):
    arr_L = indice.L.values
    nan_indices = np.where(np.isnan(arr_L))[0]
    for i in nan_indices:
        da_month = indice.time[i].dt.month.values
        lead = 10 - da_month
        indice.L.values[i] = lead

    return indice

def ClasificarEventos(r, ds1, ds2, ds3, variables, thr=0.5):
    """
    Clasifica eventos para un miembro r, con claves compatibles con InitEvent_dict.

    Args:
        r (int or float): miembro a seleccionar en la dimensión 'r'.
        ds1, ds2, ds3 (xr.Dataset): indices con dimensiones (time, r).
        variables (list of str): lista con los nombres de los índices [i1, i2, i3].
        thr (float): Umbral absoluto.

    Returns:
        dict: Diccionario anidado de eventos con fechas por categoría.
    """
    index1, index2, index3 = variables

    i1 = ds1.sel(r=r)[list(ds1.data_vars)[0]]
    i2 = ds2.sel(r=r)[list(ds2.data_vars)[0]]
    i3 = ds3.sel(r=r)[list(ds3.data_vars)[0]]

    # Máscaras absolutas
    i1_abs, i2_abs, i3_abs = np.abs(i1) > thr, np.abs(i2) > thr, np.abs(i3) > thr

    # Máscaras por signo
    i1_pos, i1_neg = i1 > thr, i1 < -thr
    i2_pos, i2_neg = i2 > thr, i2 < -thr
    i3_pos, i3_neg = i3 > thr, i3 < -thr

    i1_pos = CheckNaN_leads(i1_pos)
    i1_neg = CheckNaN_leads(i1_neg)
    i2_pos = CheckNaN_leads(i2_pos)
    i2_neg = CheckNaN_leads(i2_neg)
    i3_pos = CheckNaN_leads(i3_pos)
    i3_neg = CheckNaN_leads(i3_neg)

    # Diccionario de salida
    eventos = InitEvent_dict(variables)

    # Neutros
    eventos['neutros'] = ds1.time.where(~(i1_abs | i2_abs | i3_abs)).dropna('time')

    # Puros
    eventos['puros'][index1]['pos'] = \
        ds1.time.where(i1_pos & ~i2_abs & ~i3_abs).dropna('time')
    eventos['puros'][index1]['neg'] = \
        ds1.time.where(i1_neg & ~i2_abs & ~i3_abs).dropna('time')
    eventos['puros'][index2]['pos'] = \
        ds1.time.where(i2_pos & ~i1_abs & ~i3_abs).dropna('time')
    eventos['puros'][index2]['neg'] = \
        ds1.time.where(i2_neg & ~i1_abs & ~i3_abs).dropna('time')
    eventos['puros'][index3]['pos'] = \
        ds1.time.where(i3_pos & ~i1_abs & ~i2_abs).dropna('time')
    eventos['puros'][index3]['neg'] = \
        ds1.time.where(i3_neg & ~i1_abs & ~i2_abs).dropna('time')

    # Dobles simultáneos (mismo signo)
    eventos['simultaneos']['dobles'][f'{index1}_{index2}']['pos'] = \
        ds1.time.where(i1_pos & i2_pos & ~i3_abs).dropna('time')
    eventos['simultaneos']['dobles'][f'{index1}_{index2}']['neg'] = \
        ds1.time.where(i1_neg & i2_neg & ~i3_abs).dropna('time')
    eventos['simultaneos']['dobles'][f'{index1}_{index3}']['pos'] = \
        ds1.time.where(i1_pos & i3_pos & ~i2_abs).dropna('time')
    eventos['simultaneos']['dobles'][f'{index1}_{index3}']['neg'] = \
        ds1.time.where(i1_neg & i3_neg & ~i2_abs).dropna('time')
    eventos['simultaneos']['dobles'][f'{index2}_{index3}']['pos'] = \
        ds1.time.where(i2_pos & i3_pos & ~i1_abs).dropna('time')
    eventos['simultaneos']['dobles'][f'{index2}_{index3}']['neg'] = \
        ds1.time.where(i2_neg & i3_neg & ~i1_abs).dropna('time')

    # Dobles de signo opuesto
    eventos['simultaneos']['dobles_op'][f'{index1}_pos_{index2}_neg'] = \
        ds1.time.where(i1_pos & i2_neg & ~i3_abs).dropna('time')
    eventos['simultaneos']['dobles_op'][f'{index1}_neg_{index2}_pos'] = \
        ds1.time.where(i1_neg & i2_pos & ~i3_abs).dropna('time')
    eventos['simultaneos']['dobles_op'][f'{index1}_pos_{index3}_neg'] = \
        ds1.time.where(i1_pos & i3_neg & ~i2_abs).dropna('time')
    eventos['simultaneos']['dobles_op'][f'{index1}_neg_{index3}_pos'] = \
        ds1.time.where(i1_neg & i3_pos & ~i2_abs).dropna('time')
    eventos['simultaneos']['dobles_op'][f'{index2}_pos_{index3}_neg'] = \
        ds1.time.where(i2_pos & i3_neg & ~i1_abs).dropna('time')
    eventos['simultaneos']['dobles_op'][f'{index2}_neg_{index3}_pos'] = \
        ds1.time.where(i2_neg & i3_pos & ~i1_abs).dropna('time')

    # Triples mismo signo
    eventos['simultaneos']['triples'][f'{index1}_{index2}_{index3}']['pos'] = \
        ds1.time.where(i1_pos & i2_pos & i3_pos).dropna('time')
    eventos['simultaneos']['triples'][f'{index1}_{index2}_{index3}']['neg'] = \
        ds1.time.where(i1_neg & i2_neg & i3_neg).dropna('time')

    # Triples signo opuesto
    eventos['simultaneos']['triples_opuestos']\
        [f'{index1}_pos_{index2}_pos_{index3}_neg'] = \
        ds1.time.where(i1_pos & i2_pos & i3_neg).dropna('time')
    eventos['simultaneos']['triples_opuestos']\
        [f'{index1}_pos_{index2}_neg_{index3}_neg'] = \
        ds1.time.where(i1_pos & i2_neg & i3_neg).dropna('time')
    eventos['simultaneos']['triples_opuestos']\
        [f'{index1}_pos_{index2}_neg_{index3}_pos'] = \
        ds1.time.where(i1_pos & i2_neg & i3_pos).dropna('time')
    eventos['simultaneos']['triples_opuestos']\
        [f'{index1}_neg_{index2}_pos_{index3}_pos'] = \
        ds1.time.where(i1_neg & i2_pos & i3_pos).dropna('time')
    eventos['simultaneos']['triples_opuestos']\
        [f'{index1}_neg_{index2}_pos_{index3}_neg'] = \
        ds1.time.where(i1_neg & i2_pos & i3_neg).dropna('time')
    eventos['simultaneos']['triples_opuestos']\
        [f'{index1}_neg_{index2}_neg_{index3}_pos'] = \
        ds1.time.where(i1_neg & i2_neg & i3_pos).dropna('time')

    return eventos

def main_SelectEvents(variables, ds1, ds2, ds3, thr=0.5,
                      save=False, out_dir='/',
                      season_name='SON',  prefix_file='CFSv2'):
    """
    Aplica la clasificación de eventos a todos los miembros `r` y concatena los resultados.

    Args:
        ds1, ds2, ds3: xarray.Dataset con dimensiones (time, r).
        thr: Umbral para eventos.

    Returns:
        dict anidado con fechas combinadas para todos los miembros.
    """
    miembros = ds1.r.values
    eventos_total = InitEvent_dict(variables)

    for r in miembros:
        eventos_r = ClasificarEventos(r, ds1, ds2, ds3, variables, thr=thr)
        eventos_total = merge_event_dicts(eventos_total, eventos_r)

    if save is True:
        save_event_dict_to_netcdf(eventos_total, out_dir,
                                  season=season_name,
                                  prefix=prefix_file)
    else:
        return eventos_total
# ---------------------------------------------------------------------------- #