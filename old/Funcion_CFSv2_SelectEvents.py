"""
Funcion
Similar al SelectEvents de ENSO_IOD
INTENTO de algo más practico y generico
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
# ---------------------------------------------------------------------------- #
def xrClassifierEvents(index, r, var_name, by_r=True):
    if by_r:
        index_r = index.sel(r=r)
        aux_index_r = index_r.time[np.where(~np.isnan(index_r[var_name]))]
        index_r_f = index_r.sel(time=index_r.time.isin(aux_index_r))

        index_pos = index_r_f[var_name].time[index_r_f[var_name] > 0]
        index_neg = index_r_f[var_name].time[index_r_f[var_name] < 0]

        return index_pos, index_neg, index_r_f
    else:
        index_pos = index[var_name].time[index[var_name] > 0]
        index_neg = index[var_name].time[index[var_name] < 0]

        return index_pos, index_neg

def ConcatEvent(xr_original, xr_to_concat, dim='time'):
    if (len(xr_to_concat.time) != 0) and (len(xr_original.time) != 0):
        xr_concat = xr.concat([xr_original, xr_to_concat], dim=dim)
    elif (len(xr_to_concat.time) == 0) and (len(xr_original.time) != 0):
        xr_concat = xr_original
    elif (len(xr_to_concat.time) != 0) and (len(xr_original.time) == 0):
        xr_concat = xr_to_concat
    elif (len(xr_to_concat.time) == 0) and (len(xr_original.time) == 0):
        return xr_original

    return xr_concat

def UniqueValues(target, extract):
    target = set(target)
    extract = set(extract)

    target_uniques = target - extract
    return np.array(list(target_uniques))

def SameDatesAs(data, dates):
    return data.sel(time=data.time.isin(dates))

def SetDates(*args):
    if len(args) == 2:
        dates = np.intersect1d(args[0], args[1])
    elif len(args) == 3:
        dates = np.intersect1d(args[0], np.intersect1d(args[1], args[2]))
    else:
        raise ValueError("Error: número de argumentos")

    return SameDatesAs(args[0], dates)

def SelectData(dict, dates):
    out_put = {}
    for k, v in dict.items():
        out_put[k] = v.sel(time=~v.time.isin(dates))
    return out_put

# ---------------------------------------------------------------------------- #
def Compute(indices_names, seasons, dates_dir, out_dir):

    for s in seasons:
        indices_season = {}
        indices_or = {}

        for i in indices_names:
             aux = xr.open_dataset(f"{dates_dir}{i}_{s}_Leads_r_CFSv2.nc")
             indices_or[i] = aux
             aux = aux.where(np.abs(aux)>0.5*aux.mean('r').std())
             indices_season[i] = aux

        check_ind1_pos_ind2_neg = 666
        check_ind1_neg_ind2_pos = 666
        for r in range(1, 25):
            # Clasificados en positivos y negativos
            aux = indices_season[indices_names[0]]
            ind1_pos, ind1_neg, ind1 = xrClassifierEvents(
                aux, r, var_name=list(aux.data_vars)[0])

            aux = indices_season[indices_names[1]]
            ind2_pos, ind2_neg, ind2 = xrClassifierEvents(
                aux, r, var_name=list(aux.data_vars)[0])

            # Eventos simultaneos
            sim_events = np.intersect1d(ind2.time, ind1.time)

            # Identificando que eventos IOD y ENSO fueron simultaneos
            ind1_sim = ind1.sel(time=ind1.time.isin(sim_events))
            ind2_sim = ind2.sel(time=ind2.time.isin(sim_events))

            # Clasificando los simultaneos
            ind1_sim_pos, ind1_sim_neg = (
                xrClassifierEvents(ind1_sim, r=666,
                                   var_name=list(ind1_sim.data_vars)[0],
                                   by_r=False))
            ind2_sim_pos, ind2_sim_neg = (
                xrClassifierEvents(ind2_sim, r=666,
                                   var_name=list(ind2_sim.data_vars)[0],
                                   by_r=False))

            sim_pos = np.intersect1d(ind1_sim_pos, ind2_sim_pos)
            sim_pos = ind1_sim_pos.sel(time=ind1_sim_pos.time.isin(sim_pos))

            sim_neg = np.intersect1d(ind1_sim_neg, ind2_sim_neg)
            sim_neg = ind1_sim_neg.sel(time=ind1_sim_neg.time.isin(sim_neg))

            # Existen eventos simultaneos de signo opuesto?
            # cuales?
            if (len(sim_events) != (len(sim_pos) + len(sim_neg))):
                ind1_pos_ind2_neg = np.intersect1d(ind1_sim_pos, ind2_sim_neg)
                ind1_pos_ind2_neg = ind1_sim.sel(
                    time=ind1_sim.time.isin(ind1_pos_ind2_neg))

                ind1_neg_ind2_pos = np.intersect1d(ind1_sim_neg, ind2_sim_pos)
                ind1_neg_ind2_pos = ind1_sim.sel(
                    time=ind1_sim.time.isin(ind1_neg_ind2_pos))
            else:
                ind1_pos_ind2_neg = []
                ind1_neg_ind2_pos = []

            # Eventos puros --> los eventos que No ocurrieron en simultaneo
            ind1_puros = ind1.sel(time=~ind1.time.isin(sim_events))
            ind2_puros = ind2.sel(time=~ind2.time.isin(sim_events))

            # Clasificacion de eventos puros negativos y positivos
            ind1_puros_pos, ind1_puros_neg = (
                xrClassifierEvents(ind1_puros, r=666,
                                   var_name=list(ind1_puros.data_vars)[0],
                                   by_r=False))

            ind2_puros_pos, ind2_puros_neg = (
                xrClassifierEvents(ind2_puros, r=666,
                                   var_name=list(ind2_puros.data_vars)[0],
                                   by_r=False))

            # Años neutros. Sin ningun evento.
            """
            Un paso mas acá para elimiar las fechas q son nan debido a dato 
            faltante del CFSv2. En todo el resto del código no importan xq 
            fueron descartados todos los nan luego de tomar criterios para cada
            índice.
            """
            aux_ind1_season = indices_or[indices_names[0]].sel(r=r)
            dates_ref = aux_ind1_season.time[
                np.where(~np.isnan(
                    aux_ind1_season[list(aux_ind1_season.data_vars)[0]]))]
            mask = np.in1d(dates_ref, ind1.time,
                           invert=True)  # cuales de esas fechas no fueron ind1
            neutros = indices_or[indices_names[0]].sel(
                time=indices_or[indices_names[0]].time.isin(
                    dates_ref.time[mask]), r=r)

            mask = np.in1d(neutros.time, ind2.time,
                           invert=True)  # cuales de esas fechas no fueron ind2
            neutros = neutros.time[mask]

            if r == 1:
                ind1_puros_pos_f = ind1_puros_pos
                ind1_puros_neg_f = ind1_puros_neg

                ind2_puros_pos_f = ind2_puros_pos
                ind2_puros_neg_f = ind2_puros_neg

                sim_neg_f = sim_neg
                sim_pos_f = sim_pos

                ind1_pos_f = ind1_pos
                ind1_neg_f = ind1_neg
                ind2_pos_f = ind2_pos
                ind2_neg_f = ind2_neg

                neutros_f = neutros

            else:

                ind1_puros_pos_f = ConcatEvent(ind1_puros_pos_f, ind1_puros_pos)
                ind1_puros_neg_f = ConcatEvent(ind1_puros_neg_f, ind1_puros_neg)

                ind2_puros_pos_f = ConcatEvent(ind2_puros_pos_f, ind2_puros_pos)
                ind2_puros_neg_f = ConcatEvent(ind2_puros_neg_f, ind2_puros_neg)

                sim_neg_f = ConcatEvent(sim_neg_f, sim_neg)
                sim_pos_f = ConcatEvent(sim_pos_f, sim_pos)

                ind1_pos_f = ConcatEvent(ind1_pos_f, ind1_pos)
                ind1_neg_f = ConcatEvent(ind1_neg_f, ind1_neg)
                ind2_pos_f = ConcatEvent(ind2_pos_f, ind2_pos)
                ind2_neg_f = ConcatEvent(ind2_neg_f, ind2_neg)

                neutros_f = ConcatEvent(neutros_f, neutros)

            # Signos opuestos
            if (check_ind1_neg_ind2_pos == 666) and (
                    len(ind1_neg_ind2_pos) != 0):
                ind1_neg_ind2_pos_f = ind1_neg_ind2_pos
                check_ind1_neg_ind2_pos = 616
            elif (len(ind1_neg_ind2_pos) != 0):
                ind1_neg_ind2_pos_f = ConcatEvent(ind1_neg_ind2_pos_f,
                                                  ind1_neg_ind2_pos)

            if (check_ind1_pos_ind2_neg == 666) and (
                    len(ind1_pos_ind2_neg) != 0):
                ind1_pos_ind2_neg_f = ind1_pos_ind2_neg
                check_ind1_pos_ind2_neg = 616
            elif (len(ind1_pos_ind2_neg) != 0):
                ind1_pos_ind2_neg_f = ConcatEvent(ind1_pos_ind2_neg_f,
                                                  ind1_pos_ind2_neg)

        print('Saving...')

        ind1_puros_pos_f.to_netcdf(
            f'{out_dir}{indices_names[0]}_puros_pos_f_{s}_05.nc')
        ind1_puros_neg_f.to_netcdf(
            f'{out_dir}{indices_names[0]}_puros_neg_f_{s}_05.nc')
        ind2_puros_pos_f.to_netcdf(
            f'{out_dir}{indices_names[1]}_puros_pos_f_{s}_05.nc')
        ind2_puros_neg_f.to_netcdf(
            f'{out_dir}{indices_names[1]}_puros_neg_f_{s}_05.nc')

        sim_neg_f.to_netcdf(f'{out_dir}sim_neg_{indices_names[0]}-'
                            f'{indices_names[1]}_f_{s}_05.nc')
        sim_pos_f.to_netcdf(f'{out_dir}sim_pos_{indices_names[0]}-'
                            f'{indices_names[1]}_f_{s}_05.nc')
        neutros_f.to_netcdf(f'{out_dir}neutros_{indices_names[0]}-'
                            f'{indices_names[1]}_f_{s}_05.nc')

        ind1_pos_f.to_netcdf(f'{out_dir}{indices_names[0]}_pos_f_{s}_05.nc')
        ind1_neg_f.to_netcdf(f'{out_dir}{indices_names[0]}_neg_f_{s}_05.nc')
        ind2_pos_f.to_netcdf(f'{out_dir}{indices_names[1]}_pos_f_{s}_05.nc')
        ind2_neg_f.to_netcdf(f'{out_dir}{indices_names[1]}_neg_f_{s}_05.nc')

        if len(ind1_neg_ind2_pos_f) != 0:
            ind1_neg_ind2_pos_f.to_netcdf(
                f'{out_dir}{indices_names[0]}_neg_{indices_names[1]}'
                f'_pos_f_{s}_05.nc')

        if len(ind1_pos_ind2_neg_f) != 0:
            ind1_pos_ind2_neg_f.to_netcdf(
                f'{out_dir}{indices_names[0]}_pos_{indices_names[1]}'
                f'_neg_f_{s}_05.nc')

    print('Done')
    print(f'Results in {out_dir}')
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #