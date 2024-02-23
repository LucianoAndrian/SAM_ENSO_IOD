"""
Similar al SelectEvents de ENSO_IOD
"""
# ---------------------------------------------------------------------------- #
save = False
# ---------------------------------------------------------------------------- #
path = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/eof/'
path_aux = '/pikachu/datos/luciano.andrian/cases_fields/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/events_dates/'
out_dir2 = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cases_index/'
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
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


# ---------------------------------------------------------------------------- #

try:
    sam_season = xr.open_dataset(out_dir2 + 'SAM_SON_Leads_r_CFSv2.nc')
except:
    print('SAM_SON_Leads_r_CFSv2.nc not found, computing...')
    sam_index = xr.open_dataset(path + 'sam_rmon_r_z200.nc') \
        .rolling(time2=3, center=True).mean()
    sam_index = sam_index.rename({'pcs': 'sam'})
    for l, mm in zip([0, 1, 2, 3], [10, 9, 8, 7]):
        aux = sam_index.sel(
            time2=sam_index.time2.dt.month.isin(mm), L=l)
        aux = aux.assign_coords({'L': l})
        if l == 0:
            sam_season = aux
        else:
            sam_season = xr.concat([sam_season, aux], dim='time2')
    sam_season = sam_season.drop(['mode', 'month'])
    sam_season = sam_season.rename({'time2': 'time'})
    sam_season.to_netcdf(out_dir2 + 'SAM_SON_Leads_r_CFSv2.nc')


file = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/' \
       'SAM_SON_Leads_r_CFSv2.nc'
if ~os.path.exists(file):
    sam_season.to_netcdf(file)


hgt = xr.open_dataset(path_aux + 'hgt_mon_anom_d.nc')
# ---------------------------------------------------------------------------- #
dates_dir = '/datos/luciano.andrian/ncfiles/NMME_CFSv2/DMI_N34_Leads_r/'

########################################################################################################################
seasons = ['JJA', 'JAS', 'ASO', 'SON']
main_month_season = [7, 8, 9, 10]

# for ms in main_month_season:
ms = main_month_season[3]
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

# Criterio SAM
data_sam = sam_season.where(np.abs(sam_season) >
                            0.5*sam_season.mean('r').std())

"""
Clasificacion de eventos para cada r en funcion de los criterios de arriba
el r, la seasons y la fecha funcionan de labels para identificar los campos
correspondientes en las variables
"""
for r in range(1, 25):
    # Clasificados en positivos y negativos y seleccionados para cada r
    dmi_pos, dmi_neg, dmi = xrClassifierEvents(data_dmi, r, 'sst')
    n34_pos, n34_neg, n34 = xrClassifierEvents(data_n34, r, 'sst')
    sam_pos, sam_neg, sam = xrClassifierEvents(data_sam, r, 'sam')
    # Estos serian 28-33

    # ------------------------------------------------------------------------ #
    # Eventos simultaneos ---------------------------------------------------- #
    # Fechas
    # x2
    dmi_n34_sim = np.intersect1d(n34.time, dmi.time)
    dmi_sam_sim = np.intersect1d(dmi.time, sam.time)
    n34_sam_sim = np.intersect1d(n34.time, sam.time)

    # x3
    dmi_n34_sam_sim = np.intersect1d(dmi_n34_sim, sam.time)

    # simultaneos "wo"
    dmi_n34_wo_sam = UniqueValues(dmi_n34_sim, dmi_n34_sam_sim)
    dmi_sam_wo_n34 = UniqueValues(dmi_sam_sim, dmi_n34_sam_sim)
    n34_sam_wo_dmi = UniqueValues(n34_sam_sim, dmi_n34_sam_sim)
    # ------------------------------------------------------------------------ #

    # Simultaneos triples
    dmi_sim_n34_sam = dmi.sel(time=dmi.time.isin(dmi_n34_sam_sim))
    n34_sim_dmi_sam = n34.sel(time=n34.time.isin(dmi_n34_sam_sim))
    sam_sim_dmi_n34 = sam.sel(time=sam.time.isin(dmi_n34_sam_sim))

    # Simultaneos dobles
    dmi_sim_n34_wo_sam = dmi.sel(time=dmi.time.isin(dmi_n34_wo_sam))
    dmi_sim_sam_wo_n34 = dmi.sel(time=dmi.time.isin(dmi_sam_wo_n34))
    n34_sim_dmi_wo_sam = n34.sel(time=n34.time.isin(dmi_n34_wo_sam))
    n34_sim_sam_wo_dmi = n34.sel(time=n34.time.isin(n34_sam_wo_dmi))
    sam_sim_n34_wo_dmi = sam.sel(time=sam.time.isin(n34_sam_wo_dmi))
    sam_sim_dmi_wo_n34 = sam.sel(time=sam.time.isin(dmi_sam_wo_n34))
    # ------------------------------------------------------------------------ #
    # Clasificando los simultaneos

    # Simultaneos triples
    # Puede estar demás ahcerlo para todos...
    dmi_pos_sim_n34_sam, dmi_neg_sim_n34_sam = xrClassifierEvents(
        dmi_sim_n34_sam, r=666, var_name='sst', by_r=False)
    n34_pos_sim_dmi_sam, n34_neg_sim_dmi_sam = xrClassifierEvents(
        n34_sim_dmi_sam, r=666, var_name='sst', by_r=False)
    sam_pos_sim_dmi_n34, sam_neg_sim_dmi_n34 = xrClassifierEvents(
        sam_sim_dmi_n34, r=666, var_name='sam', by_r=False)

    # Simultaneos dobles
    dmi_pos_sim_n34_wo_sam, dmi_neg_sim_n34_wo_sam = xrClassifierEvents(
        dmi_sim_n34_wo_sam, r=666, var_name='sst', by_r=False)
    dmi_pos_sim_sam_wo_n34, dmi_neg_sim_sam_wo_n34 = xrClassifierEvents(
        dmi_sim_sam_wo_n34, r=666, var_name='sst', by_r=False)
    n34_pos_sim_dmi_wo_sam, n34_neg_sim_dmi_wo_sam = xrClassifierEvents(
        n34_sim_dmi_wo_sam, r=666, var_name='sst', by_r=False)
    n34_pos_sim_sam_wo_dmi, n34_neg_sim_sam_wo_dmi = xrClassifierEvents(
        n34_sim_sam_wo_dmi, r=666, var_name='sst', by_r=False)
    sam_pos_sim_n34_wo_dmi, sam_neg_sim_n34_wo_dmi = xrClassifierEvents(
        sam_sim_n34_wo_dmi, r=666, var_name='sam', by_r=False)
    sam_pos_sim_dmi_wo_n34, sam_neg_sim_dmi_wo_n34 = xrClassifierEvents(
        sam_sim_dmi_wo_n34, r=666, var_name='sam', by_r=False)

    # 2 triples full
    sim_pos = SetDates(dmi_pos_sim_n34_sam, n34_pos_sim_dmi_sam,
                       sam_pos_sim_dmi_n34)
    sim_neg = SetDates(dmi_neg_sim_n34_sam, n34_neg_sim_dmi_sam,
                       sam_neg_sim_dmi_n34)


    # 12 dobles "wo"
    dmi_sim_n34_pos_wo_sam = SetDates(dmi_pos_sim_n34_wo_sam,
                                      n34_pos_sim_dmi_wo_sam)
    dmi_sim_sam_pos_wo_n34 = SetDates(dmi_pos_sim_sam_wo_n34,
                                      sam_pos_sim_dmi_wo_n34)
    n34_sim_sam_pos_wo_dmi = SetDates(n34_pos_sim_sam_wo_dmi,
                                      sam_pos_sim_n34_wo_dmi)
    n34_sim_dmi_pos_wo_sam = dmi_sim_n34_pos_wo_sam.copy()
    sam_sim_n34_pos_wo_dmi = n34_sim_sam_pos_wo_dmi.copy()
    sam_sim_dmi_pos_wo_n34 = dmi_sim_sam_pos_wo_n34.copy()

    dmi_sim_n34_neg_wo_sam = SetDates(dmi_neg_sim_n34_wo_sam,
                                      n34_neg_sim_dmi_wo_sam)
    dmi_sim_sam_neg_wo_n34 = SetDates(dmi_neg_sim_sam_wo_n34,
                                      sam_neg_sim_dmi_wo_n34)
    n34_sim_sam_neg_wo_dmi = SetDates(n34_neg_sim_sam_wo_dmi,
                                      sam_neg_sim_n34_wo_dmi)
    n34_sim_dmi_neg_wo_sam = dmi_sim_n34_neg_wo_sam.copy()
    sam_sim_n34_neg_wo_dmi = n34_sim_sam_neg_wo_dmi.copy()
    sam_sim_dmi_neg_wo_n34 = dmi_sim_sam_neg_wo_n34.copy()

    dmi_pos_n34_neg_wo_sam = SetDates(dmi_pos_sim_n34_wo_sam,
                                      n34_neg_sim_dmi_wo_sam)
    dmi_pos_sam_neg_wo_n34 = SetDates(dmi_pos_sim_sam_wo_n34,
                                      sam_neg_sim_dmi_wo_n34)
    n34_pos_sam_neg_wo_dmi = SetDates(n34_pos_sim_sam_wo_dmi,
                                      sam_neg_sim_n34_wo_dmi)
    n34_pos_dmi_neg_wo_sam = dmi_pos_n34_neg_wo_sam.copy()
    sam_pos_dmi_neg_wo_n34 = dmi_pos_sam_neg_wo_n34.copy()
    sam_pos_n34_neg_wo_dmi = n34_pos_sam_neg_wo_dmi.copy()

    dmi_neg_n34_pos_wo_sam = SetDates(dmi_neg_sim_n34_wo_sam,
                                      n34_pos_sim_dmi_wo_sam)
    dmi_neg_sam_pos_wo_n34 = SetDates(dmi_neg_sim_sam_wo_n34,
                                      sam_pos_sim_dmi_wo_n34)
    n34_neg_sam_pos_wo_dmi = SetDates(n34_neg_sim_sam_wo_dmi,
                                      sam_pos_sim_n34_wo_dmi)
    n34_neg_dmi_pos_wo_sam = dmi_neg_n34_pos_wo_sam.copy()
    sam_neg_dmi_pos_wo_n34 = dmi_neg_sam_pos_wo_n34.copy()
    sam_neg_n34_pos_wo_dmi = n34_neg_sam_pos_wo_dmi.copy()

    # 6 tiples 'op'
    dmi_pos_n34_pos_sam_neg = SetDates(dmi_pos_sim_n34_sam, n34_pos_sim_dmi_sam,
                                       sam_neg_sim_dmi_n34)
    dmi_pos_n34_neg_sam_pos = SetDates(dmi_pos_sim_n34_sam, n34_neg_sim_dmi_sam,
                                       sam_pos_sim_dmi_n34)
    dmi_pos_n34_neg_sam_neg = SetDates(dmi_pos_sim_n34_sam, n34_neg_sim_dmi_sam,
                                       sam_neg_sim_dmi_n34)
    dmi_neg_n34_neg_sam_pos = SetDates(dmi_neg_sim_n34_sam, n34_neg_sim_dmi_sam,
                                       sam_pos_sim_dmi_n34)
    dmi_neg_n34_pos_sam_pos = SetDates(dmi_neg_sim_n34_sam, n34_pos_sim_dmi_sam,
                                       sam_pos_sim_dmi_n34)
    dmi_neg_n34_pos_sam_neg = SetDates(dmi_neg_sim_n34_sam, n34_pos_sim_dmi_sam,
                                       sam_neg_sim_dmi_n34)
    # ------------------------------------------------------------------------ #
    # Eventos puros
    dmi_puros = dmi.sel(time=~dmi.time.isin(dmi_n34_sam_sim))
    dmi_puros = dmi_puros.sel(time=~dmi_puros.time.isin(dmi_n34_wo_sam))
    dmi_puros = dmi_puros.sel(time=~dmi_puros.time.isin(dmi_sam_wo_n34))

    n34_puros = n34.sel(time=~n34.time.isin(dmi_n34_sam_sim))
    n34_puros = n34_puros.sel(time=~n34_puros.time.isin(dmi_n34_wo_sam))
    n34_puros = n34_puros.sel(time=~n34_puros.time.isin(n34_sam_wo_dmi))

    sam_puros = sam.sel(time=~sam.time.isin(dmi_n34_sam_sim))
    sam_puros = sam_puros.sel(time=~sam_puros.time.isin(dmi_sam_wo_n34))
    sam_puros = sam_puros.sel(time=~sam_puros.time.isin(n34_sam_wo_dmi))

    # 6 puros
    dmi_puros_pos, dmi_puros_neg = xrClassifierEvents(dmi_puros,
                                                      r=666, var_name='sst',
                                                      by_r=False)
    n34_puros_pos, n34_puros_neg = xrClassifierEvents(n34_puros,
                                                      r=666, var_name='sst',
                                                      by_r=False)
    sam_puros_pos, sam_puros_neg = xrClassifierEvents(sam_puros,
                                                      r=666, var_name='sam',
                                                      by_r=False)

    # Puede haber datos faltantes en algunas inicializaciónes (especialmente SST)
    # por eso primero se revisa que solo se consideren los valores no nan para los
    # neutros.
    # 1 neutro
    aux_dmi_season = dmi_season.sel(r=r)
    dates_ref = aux_dmi_season.time[np.where(~np.isnan(aux_dmi_season.sst))]
    aux_sam_season = sam_season.sel(r=r)
    dates_ref_sam = aux_sam_season.time[np.where(~np.isnan(aux_sam_season.sam))]

    dates_ref = np.intersect1d(dates_ref, dates_ref_sam)

    mask = np.in1d(dates_ref, dmi.time,
                   invert=True)  # cuales de esas fechas no fueron dmi
    neutros = dmi_season.sel(
        time=dmi_season.time.isin(dates_ref[mask]), r=r)

    mask = np.in1d(neutros.time, n34.time,
                   invert=True)  # cuales de esas fechas no fueron n34
    neutros = neutros.time[mask]

    mask = np.in1d(neutros.time, sam.time,
                   invert=True)  # cuales de esas fechas no fueron sam
    neutros = neutros.time[mask]

    if r == 1:
        # 6 puros
        dmi_puros_pos_f = dmi_puros_pos
        dmi_puros_neg_f = dmi_puros_neg
        n34_puros_pos_f = n34_puros_pos
        n34_puros_neg_f = n34_puros_neg
        sam_puros_pos_f = sam_puros_pos
        sam_puros_neg_f = sam_puros_neg

        # 6 extras
        dmi_pos_f = dmi_pos
        dmi_neg_f = dmi_neg
        n34_pos_f = n34_pos
        n34_neg_f = n34_neg
        sam_pos_f = sam_pos
        sam_neg_f = sam_neg

        # 1 neutro
        neutros_f = neutros

        # 2 triples full
        sim_neg_f = sim_neg
        sim_pos_f = sim_pos

        # 6 triples 'op'
        dmi_pos_n34_pos_sam_neg_f = dmi_pos_n34_pos_sam_neg
        dmi_pos_n34_neg_sam_pos_f = dmi_pos_n34_neg_sam_pos

        dmi_pos_n34_neg_sam_neg_f = dmi_pos_n34_neg_sam_neg
        dmi_neg_n34_neg_sam_pos_f = dmi_neg_n34_neg_sam_pos

        dmi_neg_n34_pos_sam_pos_f = dmi_neg_n34_pos_sam_pos
        dmi_neg_n34_pos_sam_neg_f = dmi_neg_n34_pos_sam_neg

        # 12 dobles "wo"
        dmi_sim_n34_pos_wo_sam_f = dmi_sim_n34_pos_wo_sam
        dmi_sim_sam_pos_wo_n34_f = dmi_sim_sam_pos_wo_n34
        n34_sim_sam_pos_wo_dmi_f = n34_sim_sam_pos_wo_dmi
        n34_sim_dmi_pos_wo_sam_f = n34_sim_dmi_pos_wo_sam
        sam_sim_n34_pos_wo_dmi_f = sam_sim_n34_pos_wo_dmi
        sam_sim_dmi_pos_wo_n34_f = sam_sim_dmi_pos_wo_n34
        dmi_sim_n34_neg_wo_sam_f = dmi_sim_n34_neg_wo_sam
        n34_sim_sam_neg_wo_dmi_f = n34_sim_sam_neg_wo_dmi
        n34_sim_dmi_neg_wo_sam_f = n34_sim_dmi_neg_wo_sam
        sam_sim_n34_neg_wo_dmi_f = sam_sim_n34_neg_wo_dmi
        sam_sim_dmi_neg_wo_n34_f = sam_sim_dmi_neg_wo_n34
        dmi_sim_sam_neg_wo_n34_f = dmi_sim_sam_neg_wo_n34

        dmi_pos_n34_neg_wo_sam_f = dmi_pos_n34_neg_wo_sam
        dmi_pos_sam_neg_wo_n34_f = dmi_pos_sam_neg_wo_n34
        n34_pos_sam_neg_wo_dmi_f = n34_pos_sam_neg_wo_dmi
        n34_pos_dmi_neg_wo_sam_f = n34_pos_dmi_neg_wo_sam
        sam_pos_dmi_neg_wo_n34_f = sam_pos_dmi_neg_wo_n34
        sam_pos_n34_neg_wo_dmi_f = sam_pos_n34_neg_wo_dmi

        dmi_neg_n34_pos_wo_sam_f = dmi_neg_n34_pos_wo_sam
        dmi_neg_sam_pos_wo_n34_f = dmi_neg_sam_pos_wo_n34
        n34_neg_sam_pos_wo_dmi_f = n34_neg_sam_pos_wo_dmi
        n34_neg_dmi_pos_wo_sam_f = n34_neg_dmi_pos_wo_sam
        sam_neg_dmi_pos_wo_n34_f = sam_neg_dmi_pos_wo_n34
        sam_neg_n34_pos_wo_dmi_f = sam_neg_n34_pos_wo_dmi

        sam_f = sam
        dmi_f = dmi
        n34_f = n34


    else:

        dmi_puros_pos_f = ConcatEvent(dmi_puros_pos_f, dmi_puros_pos)
        dmi_puros_neg_f = ConcatEvent(dmi_puros_neg_f, dmi_puros_neg)
        n34_puros_pos_f = ConcatEvent(n34_puros_pos_f, n34_puros_pos)
        n34_puros_neg_f = ConcatEvent(n34_puros_neg_f, n34_puros_neg)
        sam_puros_pos_f = ConcatEvent(sam_puros_pos_f, sam_puros_pos)
        sam_puros_neg_f = ConcatEvent(sam_puros_neg_f, sam_puros_neg)

        dmi_pos_f = ConcatEvent(dmi_pos_f, dmi_pos)
        dmi_neg_f = ConcatEvent(dmi_neg_f, dmi_neg)
        n34_pos_f = ConcatEvent(n34_pos_f, n34_pos)
        n34_neg_f = ConcatEvent(n34_neg_f, n34_neg)
        sam_pos_f = ConcatEvent(sam_pos_f, sam_pos)
        sam_neg_f = ConcatEvent(sam_neg_f, sam_neg)

        neutros_f = ConcatEvent(neutros_f, neutros)

        sim_neg_f = ConcatEvent(sim_neg_f, sim_neg)
        sim_pos_f = ConcatEvent(sim_pos_f, sim_pos)

        dmi_pos_n34_pos_sam_neg_f = ConcatEvent(dmi_pos_n34_pos_sam_neg_f,
                                                dmi_pos_n34_pos_sam_neg)
        dmi_pos_n34_neg_sam_pos_f = ConcatEvent(dmi_pos_n34_neg_sam_pos_f,
                                                dmi_pos_n34_neg_sam_pos)

        dmi_pos_n34_neg_sam_neg_f = ConcatEvent(dmi_pos_n34_neg_sam_neg_f,
                                                dmi_pos_n34_neg_sam_neg)
        dmi_neg_n34_neg_sam_pos_f = ConcatEvent(dmi_neg_n34_neg_sam_pos_f,
                                                dmi_neg_n34_neg_sam_pos)

        dmi_neg_n34_pos_sam_pos_f = ConcatEvent(dmi_neg_n34_pos_sam_pos_f,
                                                dmi_neg_n34_pos_sam_pos)
        dmi_neg_n34_pos_sam_neg_f = ConcatEvent(dmi_neg_n34_pos_sam_neg_f,
                                                dmi_neg_n34_pos_sam_neg)

        dmi_sim_n34_pos_wo_sam_f = ConcatEvent(dmi_sim_n34_pos_wo_sam_f,
                                               dmi_sim_n34_pos_wo_sam)
        dmi_sim_sam_pos_wo_n34_f = ConcatEvent(dmi_sim_sam_pos_wo_n34_f,
                                               dmi_sim_sam_pos_wo_n34)
        n34_sim_sam_pos_wo_dmi_f = ConcatEvent(n34_sim_sam_pos_wo_dmi_f,
                                               n34_sim_sam_pos_wo_dmi)
        n34_sim_dmi_pos_wo_sam_f = ConcatEvent(n34_sim_dmi_pos_wo_sam_f,
                                               n34_sim_dmi_pos_wo_sam)
        sam_sim_n34_pos_wo_dmi_f = ConcatEvent(sam_sim_n34_pos_wo_dmi_f,
                                               sam_sim_n34_pos_wo_dmi)
        sam_sim_dmi_pos_wo_n34_f = ConcatEvent(sam_sim_dmi_pos_wo_n34_f,
                                               sam_sim_dmi_pos_wo_n34)
        dmi_sim_n34_neg_wo_sam_f = ConcatEvent(dmi_sim_n34_neg_wo_sam_f,
                                               dmi_sim_n34_neg_wo_sam)
        n34_sim_sam_neg_wo_dmi_f = ConcatEvent(n34_sim_sam_neg_wo_dmi_f,
                                               n34_sim_sam_neg_wo_dmi)
        n34_sim_dmi_neg_wo_sam_f = ConcatEvent(n34_sim_dmi_neg_wo_sam_f,
                                               n34_sim_dmi_neg_wo_sam)
        sam_sim_n34_neg_wo_dmi_f = ConcatEvent(sam_sim_n34_neg_wo_dmi_f,
                                               sam_sim_n34_neg_wo_dmi)
        sam_sim_dmi_neg_wo_n34_f = ConcatEvent(sam_sim_dmi_neg_wo_n34_f,
                                               sam_sim_dmi_neg_wo_n34)
        dmi_sim_sam_neg_wo_n34_f = ConcatEvent(dmi_sim_sam_neg_wo_n34_f,
                                               dmi_sim_sam_neg_wo_n34)

        dmi_pos_n34_neg_wo_sam_f = ConcatEvent(dmi_pos_n34_neg_wo_sam_f,
                                               dmi_pos_n34_neg_wo_sam)
        dmi_pos_sam_neg_wo_n34_f = ConcatEvent(dmi_pos_sam_neg_wo_n34_f,
                                               dmi_pos_sam_neg_wo_n34)
        n34_pos_sam_neg_wo_dmi_f = ConcatEvent(n34_pos_sam_neg_wo_dmi_f,
                                               n34_pos_sam_neg_wo_dmi)
        n34_pos_dmi_neg_wo_sam_f = ConcatEvent(n34_pos_dmi_neg_wo_sam_f,
                                               n34_pos_dmi_neg_wo_sam)
        sam_pos_dmi_neg_wo_n34_f = ConcatEvent(sam_pos_dmi_neg_wo_n34_f,
                                               sam_pos_dmi_neg_wo_n34)
        sam_pos_n34_neg_wo_dmi_f = ConcatEvent(sam_pos_n34_neg_wo_dmi_f,
                                               sam_pos_n34_neg_wo_dmi)

        dmi_neg_n34_pos_wo_sam_f = ConcatEvent(dmi_neg_n34_pos_wo_sam_f,
                                               dmi_neg_n34_pos_wo_sam)
        dmi_neg_sam_pos_wo_n34_f = ConcatEvent(dmi_neg_sam_pos_wo_n34_f,
                                               dmi_neg_sam_pos_wo_n34)
        n34_neg_sam_pos_wo_dmi_f = ConcatEvent(n34_neg_sam_pos_wo_dmi_f,
                                               n34_neg_sam_pos_wo_dmi)
        n34_neg_dmi_pos_wo_sam_f = ConcatEvent(n34_neg_dmi_pos_wo_sam_f,
                                               n34_neg_dmi_pos_wo_sam)
        sam_neg_dmi_pos_wo_n34_f = ConcatEvent(sam_neg_dmi_pos_wo_n34_f,
                                               sam_neg_dmi_pos_wo_n34)
        sam_neg_n34_pos_wo_dmi_f = ConcatEvent(sam_neg_n34_pos_wo_dmi_f,
                                               sam_neg_n34_pos_wo_dmi)

        sam_f = ConcatEvent(sam_f, sam)
        dmi_f = ConcatEvent(dmi_f, dmi)
        n34_f = ConcatEvent(n34_f, n34)

variables = {'dmi_puros_pos_f': dmi_puros_pos_f,
             'dmi_puros_neg_f': dmi_puros_neg_f,
             'n34_puros_pos_f': n34_puros_pos_f,
             'n34_puros_neg_f': n34_puros_neg_f,
             'sam_puros_pos_f': sam_puros_pos_f,
             'sam_puros_neg_f': sam_puros_neg_f,
             'dmi_pos_f': dmi_pos_f,
             'dmi_neg_f': dmi_neg_f,
             'n34_pos_f': n34_pos_f,
             'n34_neg_f': n34_neg_f,
             'sam_pos_f': sam_pos_f,
             'sam_neg_f': sam_neg_f,
             'neutros_f': neutros_f,
             'sim_neg_f': sim_neg_f,
             'sim_pos_f': sim_pos_f,
             'dmi_pos_n34_pos_sam_neg_f': dmi_pos_n34_pos_sam_neg_f,
             'dmi_pos_n34_neg_sam_pos_f': dmi_pos_n34_neg_sam_pos_f,
             'dmi_pos_n34_neg_sam_neg_f': dmi_pos_n34_neg_sam_neg_f,
             'dmi_neg_n34_neg_sam_pos_f': dmi_neg_n34_neg_sam_pos_f,
             'dmi_neg_n34_pos_sam_pos_f': dmi_neg_n34_pos_sam_pos_f,
             'dmi_neg_n34_pos_sam_neg_f': dmi_neg_n34_pos_sam_neg_f,
             'dmi_sim_n34_pos_wo_sam_f': dmi_sim_n34_pos_wo_sam_f,
             'dmi_sim_sam_pos_wo_n34_f': dmi_sim_sam_pos_wo_n34_f,
             'n34_sim_sam_pos_wo_dmi_f': n34_sim_sam_pos_wo_dmi_f,
             'n34_sim_dmi_pos_wo_sam_f': n34_sim_dmi_pos_wo_sam_f,
             'sam_sim_n34_pos_wo_dmi_f': sam_sim_n34_pos_wo_dmi_f,
             'sam_sim_dmi_pos_wo_n34_f': sam_sim_dmi_pos_wo_n34_f,
             'dmi_sim_n34_neg_wo_sam_f': dmi_sim_n34_neg_wo_sam_f,
             'n34_sim_sam_neg_wo_dmi_f': n34_sim_sam_neg_wo_dmi_f,
             'n34_sim_dmi_neg_wo_sam_f': n34_sim_dmi_neg_wo_sam_f,
             'sam_sim_n34_neg_wo_dmi_f': sam_sim_n34_neg_wo_dmi_f,
             'sam_sim_dmi_neg_wo_n34_f': sam_sim_dmi_neg_wo_n34_f,
             'dmi_sim_sam_neg_wo_n34_f':dmi_sim_sam_neg_wo_n34_f,
             'dmi_pos_n34_neg_wo_sam_f' : dmi_pos_n34_neg_wo_sam_f,
             'dmi_pos_sam_neg_wo_n34_f' : dmi_pos_sam_neg_wo_n34_f,
             'n34_pos_sam_neg_wo_dmi_f' : n34_pos_sam_neg_wo_dmi_f,
             'n34_pos_dmi_neg_wo_sam_f' : n34_pos_dmi_neg_wo_sam_f,
             'sam_pos_dmi_neg_wo_n34_f' : sam_pos_dmi_neg_wo_n34_f,
             'sam_pos_n34_neg_wo_dmi_f' : sam_pos_n34_neg_wo_dmi_f,
             'dmi_neg_n34_pos_wo_sam_f' : dmi_neg_n34_pos_wo_sam_f,
             'dmi_neg_sam_pos_wo_n34_f' : dmi_neg_sam_pos_wo_n34_f,
             'n34_neg_sam_pos_wo_dmi_f' : n34_neg_sam_pos_wo_dmi_f,
             'n34_neg_dmi_pos_wo_sam_f' : n34_neg_dmi_pos_wo_sam_f,
             'sam_neg_dmi_pos_wo_n34_f' : sam_neg_dmi_pos_wo_n34_f,
             'sam_neg_n34_pos_wo_dmi_f' : sam_neg_n34_pos_wo_dmi_f,
             'sam_f':sam_f, 'dmi_f':dmi_f, 'n34_f':n34_f}

print('Saving...')
for nombre, objeto in variables.items():
    objeto.to_netcdf(out_dir + nombre + '_' + s + '.nc')

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