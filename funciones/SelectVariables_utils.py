"""
Funciones para seleccionar las fechas correspondiendes a cada evento
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
from multiprocessing import Process

# ---------------------------------------------------------------------------- #
def SelectVariables(dates, data):
    data_select = []

    for t in dates.index:

        try:
            r_t = t.r.values
        except:
            r_t = dates.r[0].values
        L_t = t.L.values
        t_t = t.values

        try:
            t_t * 1
            t_t = t.time.values
        except:
            pass

        mask = ((data.L == L_t) &
                (data.r == r_t) &
                (data.time == t_t))

        selected = data.where(mask, drop=True)
        data_select.append(selected.isel(r=0))

    try:
        out_put = xr.concat(data_select, dim='time')
    except:
        out_put = None

    return out_put

def Aux_SelectVariables(f, var_file, cases_dir, data_dir, out_dir,
                     replace_name):
    """
    Auxiliar para poder correr SelectVariables en paralelo
    """
    aux_cases = xr.open_dataset(f'{cases_dir}{f}')
    aux_cases = aux_cases.rename({list(aux_cases.data_vars)[0]:'index'})

    data_var = xr.open_dataset(f'{data_dir}{var_file}')
    if 'tref' in var_file:
        data_var = data_var.sel(lon=slice(275, 330), lat=slice(-60, 15))

    case_events = SelectVariables(aux_cases, data_var)

    f_name = f.replace(replace_name, "")
    var_name = var_file.split('_')[0]
    case_events.to_netcdf(f'{out_dir}{var_name}_{f_name}')

def parallel_SelectVariables(files, var_file, div, cases_dir=None,
                             data_dir=None, out_dir=None,
                             replace_name='CFSv2_'):
    """
    Corre SelectVariables en paralelo
    usar div para dividir los files en grupos manejables segun recursos
    """
    for i in range(0, len(files), div):
        batch = files[i:i + div]
        processes = [Process(target=Aux_SelectVariables,
                             args=(f, var_file, cases_dir, data_dir, out_dir,
                                   replace_name))
                     for f in batch]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

# ---------------------------------------------------------------------------- #