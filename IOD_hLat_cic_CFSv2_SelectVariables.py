"""

"""
# ---------------------------------------------------------------------------- #
events_dir = ('/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
              'IOD_hLat_cic/cases_events/')
data_dir = ('/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
              'IOD_hLat_cic/cases_fields/')
out_dir = ('/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
              'IOD_hLat_cic/cases_fields_selected/')

# Setos varios --------------------------------------------------------------- #
seasons = ['MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND']

# ---------------------------------------------------------------------------- #
import xarray as xr
from multiprocessing.pool import ThreadPool
from multiprocessing import Process
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
# ---------------------------------------------------------------------------- #
def set_cases(indice1, indice2, todos=False):
    if todos:
        cases = [# puros
            f'{indice1}_puros_pos', f'{indice1}_puros_neg',
            f'{indice2}_puros_pos', f'{indice2}_puros_neg',
            # sim misma fase
            f'sim_pos_{indice1}-{indice2}',
            f'sim_neg_{indice1}-{indice2}',
            # fases opuestas
            f'{indice1}_pos_{indice2}_neg',
            f'{indice1}_neg_{indice2}_pos',
            f'neutros_{indice1}-{indice2}',  # neutros
            # restantes
            f'{indice1}_pos', f'{indice1}_neg',
            f'{indice2}_pos', f'{indice2}_neg']

    else:
        cases = [  # puros
            f'{indice1}_puros_pos', f'{indice1}_puros_neg',
            f'{indice2}_puros_pos', f'{indice2}_puros_neg',
            # sim misma fase
            f'sim_pos_{indice1}-{indice2}',
            f'sim_neg_{indice1}-{indice2}',
            # fases opuestas
            f'{indice1}_pos_{indice2}_neg',
            f'{indice1}_neg_{indice2}_pos',
            f'neutros_{indice1}-{indice2}']

    return cases

def SelectVariables(dates, data):

    t_count=0
    t_count_aux = 0
    for t in dates.index:
        try:
            r_t = t.r.values
        except:
            r_t = dates.r[t_count_aux].values
        L_t = t.L.values
        t_t = t.values
        try: #q elegancia la de francia...
            t_t*1
            t_t = t.time.values
        except:
            pass

        if t_count == 0:
            aux = data.where(data.L == L_t).sel(r=r_t, time=t_t)
            t_count += 1
        else:
            aux = xr.concat([aux,
                             data.where(data.L == L_t).sel(r=r_t, time=t_t)],
                            dim='time')
    return aux
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

cases = set_cases('DMI', 'U50')

if len(seasons)>1:
    def SelectEventsHGT(c):
        for s in seasons:
            try:
                aux_cases = \
                    xr.open_dataset(events_dir + c + '_f_' + s + '_05.nc')\
                        .rename({'__xarray_dataarray_variable__': 'index'})
            except:
                aux_cases = \
                    xr.open_dataset(events_dir + c + '_f_' + s + '_05.nc')\
                        .rename({'sst': 'index'})

            data_hgt_s = xr.open_dataset(f'{data_dir}hgt_{s.upper()}_Leads_r_CFSv2.nc')
            case_events = SelectVariables(aux_cases, data_hgt_s)

            case_events.to_netcdf(out_dir + 'hgt_' + c + '_' + s + '_05.nc')

    pool = ThreadPool(len(seasons))
    pool.map_async(SelectEventsHGT, [c for c in cases])

else:
    for s in seasons:
        def SelectEventsHGT(c):
            try:
                aux_cases = \
                    xr.open_dataset(events_dir + c + '_f_' + s + '_05.nc')\
                        .rename({'__xarray_dataarray_variable__': 'index'})
            except:
                aux_cases = \
                    xr.open_dataset(events_dir + c + '_f_' + s + '_05.nc')\
                        .rename({'sst': 'index'})

            data_hgt_s = xr.open_dataset(f'{data_dir}hgt_{s.upper()}_Leads_r_CFSv2.nc')
            case_events = SelectVariables(aux_cases, data_hgt_s)

            case_events.to_netcdf(out_dir + 'hgt_' + c + '_' + s + '_05.nc')


        processes = [Process(target=SelectEventsHGT, args=(c,)) for c in cases]
        for process in processes:
            process.start()
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #