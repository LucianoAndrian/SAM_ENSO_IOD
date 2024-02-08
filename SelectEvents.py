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
def xrClassifierEvents(index, r, var_name, by_r=True):
    if by_r:
        index_r = index.sel(r=r)
        aux_index_r = index_r.time[np.where(~np.isnan(index_r[var_name]))]
        index_r_f = index_r.sel(time=index_r.time.isin(aux_index_r))

        index_pos = index_r_f[var_name].time[index_r_f[var_name] > 0]
        index_neg = index_r_f[var_name].time[index_r_f[var_name] < 0]

        return index_pos, index_neg, index_r_f
    else:
        index_pos = index.sst.time[index.sst > 0]
        index_neg = index.sst.time[index.sst < 0]
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
# ---------------------------------------------------------------------------- #

sam_index = xr.open_dataset(path + 'sam_rmon_r_z200.nc')\
    .rolling(time2=3, center=True).mean()
sam_index = sam_index.rename({'pcs':'sam'})
for l, mm in zip([0,1,2,3],[10,9,8,7]):
    aux = sam_index.sel(
        time2=sam_index.time2.dt.month.isin(mm), L=l)
    aux = aux.assign_coords({'L':l})
    if l==0:
        sam_season = aux
    else:
        sam_season = xr.concat([sam_season, aux], dim='time2')
sam_season = sam_season.drop(['mode','month'])
sam_season = sam_season.rename({'time2':'time'})

hgt = xr.open_dataset(path_aux + 'hgt_mon_anom_d.nc')
# ---------------------------------------------------------------------------- #
dates_dir = '/datos/luciano.andrian/ncfiles/NMME_CFSv2/DMI_N34_Leads_r/'

out_dir = '/pikachu/datos/luciano.andrian/cases_dates/'
########################################################################################################################
seasons = ['JJA', 'JAS', 'ASO', 'SON']
main_month_season = [7, 8, 9, 10]

ms =main_month_season[3]
s = seasons[ms - 7]
print(s)
n34_season = xr.open_dataset(dates_dir + 'N34_' + s + '_Leads_r_CFSv2.nc')
dmi_season = xr.open_dataset(dates_dir + 'DMI_' + s + '_Leads_r_CFSv2.nc')

# Criterio Niño3.4
data_n34 = n34_season.where(np.abs(n34_season) >
                            0.5*n34_season.mean('r').std())

# Criterio DMI
data_dmi = dmi_season.where(np.abs(dmi_season) >
                            0.5*dmi_season.mean('r').std())

# Criterio SAM (?????)
data_sam = sam_season.where(np.abs(sam_season) >
                            0.5*sam_season.mean('r').std())

"""
Clasificacion de eventos para cada r en funcion de los criterios de arriba
el r, la seasons y la fecha funcionan de labels para identificar los campos
correspondientes en las variables
"""
check_dmi_pos_n34_neg = 666
check_dmi_neg_n34_pos = 666
for r in range(1, 25):
    # Clasificados en positivos y negativos
    dmi_pos, dmi_neg, dmi = xrClassifierEvents(data_dmi, r, 'sst')
    n34_pos, n34_neg, n34 = xrClassifierEvents(data_n34, r, 'sst')
    sam_pos, sam_neg, sam = xrClassifierEvents(data_sam, r, 'sam')
# continuar...


    # # Si se usa n34 restringido para los dmi puros
    # # n34_pos_r, n34_neg_r, n34_r = xrClassifierEvents(data_n34_restricted, r)
    #
    # # Eventos simultaneos
    # sim_events = np.intersect1d(n34.time, dmi.time)
    #
    #
    #
    # # Identificando que eventos IOD y ENSO fueron simultaneos
    # dmi_sim = dmi.sel(time=dmi.time.isin(sim_events))
    # n34_sim = n34.sel(time=n34.time.isin(sim_events))
    #
    # # Clasificando los simultaneos
    # dmi_sim_pos, dmi_sim_neg = xrClassifierEvents(dmi_sim, r=666,
    #                                               by_r=False)
    # n34_sim_pos, n34_sim_neg = xrClassifierEvents(n34_sim, r=666,
    #                                               by_r=False)
    #
    # sim_pos = np.intersect1d(dmi_sim_pos, n34_sim_pos)
    # sim_pos = dmi_sim_pos.sel(time=dmi_sim_pos.time.isin(sim_pos))
    #
    # sim_neg = np.intersect1d(dmi_sim_neg, n34_sim_neg)
    # sim_neg = dmi_sim_neg.sel(time=dmi_sim_neg.time.isin(sim_neg))
    #
    # # Existen eventos simultaneos de signo opuesto?
    # # cuales?
    # if (len(sim_events) != (len(sim_pos) + len(sim_neg))):
    #     dmi_pos_n34_neg = np.intersect1d(dmi_sim_pos, n34_sim_neg)
    #     dmi_pos_n34_neg = dmi_sim.sel(
    #         time=dmi_sim.time.isin(dmi_pos_n34_neg))
    #
    #     dmi_neg_n34_pos = np.intersect1d(dmi_sim_neg, n34_sim_pos)
    #     dmi_neg_n34_pos = dmi_sim.sel(
    #         time=dmi_sim.time.isin(dmi_neg_n34_pos))
    # else:
    #     dmi_pos_n34_neg = []
    #     dmi_neg_n34_pos = []
    #
    # # Eventos puros --> los eventos que No ocurrieron en simultaneo
    # dmi_puros = dmi.sel(time=~dmi.time.isin(sim_events))
    # n34_puros = n34.sel(time=~n34.time.isin(sim_events))
    #
    # # Valor de n34 restringido para dmi puros
    # # dmi_puros = dmi_puros.sel(time=dmi_puros.time.isin(n34_rest.time))
    #
    # # Clasificacion de eventos puros negativos y positivos
    # dmi_puros_pos, dmi_puros_neg = xrClassifierEvents(dmi_puros, r=666,
    #                                                   by_r=False)
    # n34_puros_pos, n34_puros_neg = xrClassifierEvents(n34_puros, r=666,
    #                                                   by_r=False)
    #
    # # Años neutros. Sin ningun evento.
    # """
    # Un paso mas acá para elimiar las fechas q son nan debido a dato faltante del CFSv2
    # En tdo el resto del código no importan xq fueron descartados todos los nan luego de
    # tomar criterios para cada índice.
    # """
    # aux_dmi_season = dmi_season.sel(r=r)
    # dates_ref = aux_dmi_season.time[np.where(~np.isnan(aux_dmi_season.sst))]
    # mask = np.in1d(dates_ref, dmi.time,
    #                invert=True)  # cuales de esas fechas no fueron dmi
    # neutros = dmi_season.sel(
    #     time=dmi_season.time.isin(dates_ref.time[mask]), r=r)
    #
    # mask = np.in1d(neutros.time, n34.time,
    #                invert=True)  # cuales de esas fechas no fueron n34
    # neutros = neutros.time[mask]
    #
    # if r == 1:
    #     dmi_puros_pos_f = dmi_puros_pos
    #     dmi_puros_neg_f = dmi_puros_neg
    #
    #     n34_puros_pos_f = n34_puros_pos
    #     n34_puros_neg_f = n34_puros_neg
    #
    #     sim_neg_f = sim_neg
    #     sim_pos_f = sim_pos
    #
    #     dmi_pos_f = dmi_pos
    #     dmi_neg_f = dmi_neg
    #     n34_pos_f = n34_pos
    #     n34_neg_f = n34_neg
    #
    #     neutros_f = neutros
    # else:
    #
    #     dmi_puros_pos_f = ConcatEvent(dmi_puros_pos_f, dmi_puros_pos)
    #     dmi_puros_neg_f = ConcatEvent(dmi_puros_neg_f, dmi_puros_neg)
    #
    #     n34_puros_pos_f = ConcatEvent(n34_puros_pos_f, n34_puros_pos)
    #     n34_puros_neg_f = ConcatEvent(n34_puros_neg_f, n34_puros_neg)
    #
    #     sim_neg_f = ConcatEvent(sim_neg_f, sim_neg)
    #     sim_pos_f = ConcatEvent(sim_pos_f, sim_pos)
    #
    #     dmi_pos_f = ConcatEvent(dmi_pos_f, dmi_pos)
    #     dmi_neg_f = ConcatEvent(dmi_neg_f, dmi_neg)
    #     n34_pos_f = ConcatEvent(n34_pos_f, n34_pos)
    #     n34_neg_f = ConcatEvent(n34_neg_f, n34_neg)
    #
    #     neutros_f = ConcatEvent(neutros_f, neutros)
    #
    # # Signos opuestos
    # # Son mas raros, necesitan mas condiciones
    #
    # if (check_dmi_neg_n34_pos == 666) and (len(dmi_neg_n34_pos) != 0):
    #     dmi_neg_n34_pos_f = dmi_neg_n34_pos
    #     check_dmi_neg_n34_pos = 616
    # elif (len(dmi_neg_n34_pos) != 0):
    #     dmi_neg_n34_pos_f = ConcatEvent(dmi_neg_n34_pos_f, dmi_neg_n34_pos)
    # # ----#
    # if (check_dmi_pos_n34_neg == 666) and (len(dmi_pos_n34_neg) != 0):
    #     dmi_pos_n34_neg_f = dmi_pos_n34_neg
    #     check_dmi_pos_n34_neg = 616
    # elif (len(dmi_pos_n34_neg) != 0):
    #     dmi_pos_n34_neg_f = ConcatEvent(dmi_pos_n34_neg_f, dmi_pos_n34_neg)

# ---------------------------------------------------------------------------- #
# def SelectEvents(indice, operador, umbral='0', abs_val=False):
#     runs = np.arange(1, 25)
#     dates = indice.time.values
#     leads = [0, 1, 2, 3]
#
#     first = True
#     for l in leads:
#         for r in runs:
#             for d in dates:
#                 aux = sam_index.sel(L=l, r=r, time=d)
#                 expresion = f"aux.pcs.values {operador} {umbral}"
#                 if abs_val:
#                     expresion = f"np.abs(aux.pcs.values) {operador} {umbral}"
#                 if eval(expresion):
#                     aux = aux.assign_coords(L=l)
#                     if first:
#                         first = False
#                         sam_selected = aux
#                     else:
#                         sam_selected = xr.concat([sam_selected, aux],
#                                                  dim='time')
#
#     sam_selected = sam_selected.rename({'pcs':'sam'})
#     sam_selected = sam_selected.drop('mode')
#
#     return sam_selected
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #