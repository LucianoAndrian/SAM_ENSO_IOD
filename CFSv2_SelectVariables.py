"""
Similar al SelectEvents de ENSO_IOD
"""
# ---------------------------------------------------------------------------- #
data_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
cases_date_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/' \
                 'events_dates/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cases_fields/'
out_dir2 = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cases_index/'
# ---------------------------------------------------------------------------- #
import xarray as xr
from multiprocessing.pool import ThreadPool
from multiprocessing import Process
from ENSO_IOD_Funciones import SelectVariables
# NO se pueden usar todos los cases en SST y HGT
cases = ['sim_pos', 'sim_neg', #triple sim
         'dmi_puros_pos', 'dmi_puros_neg', # puros, sin los otros dos
         'n34_puros_pos', 'n34_puros_neg', # puros, sin los otros dos
         'sam_puros_pos', 'sam_puros_neg', # puros, sin los otros dos
         'dmi_pos_n34_pos_sam_neg', # simultaneo signos opuestos
         'dmi_pos_n34_neg_sam_pos', # simultaneo signos opuestos
         'dmi_pos_n34_neg_sam_neg', # simultaneo signos opuestos
         'dmi_neg_n34_pos_sam_neg', # simultaneo signos opuestos
         'dmi_neg_n34_pos_sam_pos', # simultaneo signos opuestos
         'dmi_neg_n34_neg_sam_pos', # simultaneo signos opuestos
         'dmi_pos', 'dmi_neg', # todos los de uno, sin importar el resto
         'n34_pos', 'n34_neg', # todos los de uno, sin importar el resto
         'sam_pos', 'sam_neg', # todos los de uno, sin importar el resto
         'neutros', # todos neutros
         'dmi_sim_n34_pos_wo_sam', # dos eventos simultaneos sin el otro
         'dmi_sim_sam_pos_wo_n34', # dos eventos simultaneos sin el otro
         'n34_sim_sam_pos_wo_dmi', # dos eventos simultaneos sin el otro
         'n34_sim_dmi_pos_wo_sam', # dos eventos simultaneos sin el otro
         'sam_sim_n34_pos_wo_dmi', # dos eventos simultaneos sin el otro
         'sam_sim_dmi_pos_wo_n34', # dos eventos simultaneos sin el otro
         'dmi_sim_n34_neg_wo_sam', # dos eventos simultaneos sin el otro
         'n34_sim_sam_neg_wo_dmi', # dos eventos simultaneos sin el otro
         'n34_sim_dmi_neg_wo_sam', # dos eventos simultaneos sin el otro
         'sam_sim_n34_neg_wo_dmi', # dos eventos simultaneos sin el otro
         'sam_sim_dmi_neg_wo_n34', # dos eventos simultaneos sin el otro]
         'dmi_sim_sam_neg_wo_n34', # dos eventos simultaneos sin el otro
         'dmi_pos_n34_neg_wo_sam',
         'dmi_pos_sam_neg_wo_n34',
         'n34_pos_sam_neg_wo_dmi',
         'n34_pos_dmi_neg_wo_sam',
         'sam_pos_dmi_neg_wo_n34',
         'sam_pos_n34_neg_wo_dmi',
         'dmi_neg_n34_pos_wo_sam',
         'dmi_neg_sam_pos_wo_n34',
         'n34_neg_sam_pos_wo_dmi',
         'n34_neg_dmi_pos_wo_sam',
         'sam_neg_dmi_pos_wo_n34',
         'sam_neg_n34_pos_wo_dmi',
         'sam', 'dmi', 'n34']

seasons = ['SON']
# ---------------------------------------------------------------------------- #
# DMI ##########################################################################
cases_data_dir_dmi = '/datos/luciano.andrian/ncfiles/NMME_CFSv2/DMI_N34_Leads_r/'
if len(seasons)>1:
    def SelectEventsDMI(c):
        for s in seasons:

            for n in ['__xarray_dataarray_variable__', 'sst', 'sam']:
                try:
                    aux_cases = xr.open_dataset(
                        cases_date_dir + c + '_f_' + s + '.nc') \
                        .rename({n: 'index'})
                except:
                    pass

            data_dmi_s = xr.open_dataset(
                cases_data_dir_dmi + 'DMI_' + s + '_Leads_r_CFSv2.nc')

            case_events = SelectVariables(aux_cases, data_dmi_s)

            case_events.to_netcdf(
                out_dir2 + 'dmi_values_in_' + c + '_' + s + '.nc')

    pool = ThreadPool(4)  # uno por season
    pool.map_async(SelectEventsDMI, [c for c in cases])

else:
    print('one season')
    def SelectEventsDMI(c):
        s = seasons[0]

        for n in ['__xarray_dataarray_variable__', 'sst', 'sam']:
            try:
                aux_cases = xr.open_dataset(
                    cases_date_dir + c + '_f_' + s + '.nc') \
                    .rename({n: 'index'})
            except:
                pass

        data_dmi_s = xr.open_dataset(
            cases_data_dir_dmi + 'DMI_' + s + '_Leads_r_CFSv2.nc')

        case_events = SelectVariables(aux_cases, data_dmi_s)

        case_events.to_netcdf(
            out_dir2 + 'dmi_values_in_' + c + '_' + s + '.nc')

    processes = [Process(target=SelectEventsDMI, args=(c,)) for c in cases]
    for process in processes:
        process.start()

# N34 ##################################################################################################################
if len(seasons)>1:
    def SelectEventsN34(c):
        for s in seasons:

            for n in ['__xarray_dataarray_variable__', 'sst', 'sam']:
                try:
                    aux_cases = xr.open_dataset(
                        cases_date_dir + c + '_f_' + s + '.nc') \
                        .rename({n: 'index'})
                except:
                    pass

            data_n34_s = xr.open_dataset(
                cases_data_dir_dmi + 'N34_' + s + '_Leads_r_CFSv2.nc')

            case_events = SelectVariables(aux_cases, data_n34_s)

            case_events.to_netcdf(
                out_dir2 + 'n34_values_in_' + c + '_' + s + '.nc')

    pool = ThreadPool(4)  # uno por season
    pool.map_async(SelectEventsN34, [c for c in cases])

else:
    print('one season')
    def SelectEventsN34(c):
        s = seasons[0]

        for n in ['__xarray_dataarray_variable__', 'sst', 'sam']:
            try:
                aux_cases = xr.open_dataset(
                    cases_date_dir + c + '_f_' + s + '.nc') \
                    .rename({n: 'index'})
            except:
                pass

        data_n34_s = xr.open_dataset(
            cases_data_dir_dmi + 'N34_' + s + '_Leads_r_CFSv2.nc')

        case_events = SelectVariables(aux_cases, data_n34_s)

        case_events.to_netcdf(
            out_dir2 + 'n34_values_in_' + c + '_' + s + '.nc')

    processes = [Process(target=SelectEventsN34, args=(c,)) for c in cases]
    for process in processes:
        process.start()

# SAM ##########################################################################
cases_data_sam = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
if len(seasons)>1:
    def SelectEventsSAM(c):
        for s in seasons:

            for n in ['__xarray_dataarray_variable__', 'sst', 'sam']:
                try:
                    aux_cases = xr.open_dataset(
                        cases_date_dir + c + '_f_' + s + '.nc') \
                        .rename({n: 'index'})
                except:
                    pass

            data_sam_s = xr.open_dataset(
                cases_data_sam + 'SAM_' + s + '_Leads_r_CFSv2.nc')

            case_events = SelectVariables(aux_cases, data_sam_s)

            case_events.to_netcdf(
                out_dir2 + 'sam_values_in_' + c + '_' + s + '.nc')

    pool = ThreadPool(4)  # uno por season
    pool.map_async(SelectEventsSAM, [c for c in cases])

else:
    print('one season')
    def SelectEventsSAM(c):
        s = seasons[0]

        for n in ['__xarray_dataarray_variable__', 'sst', 'sam']:
            try:
                aux_cases = xr.open_dataset(
                    cases_date_dir + c + '_f_' + s + '.nc') \
                    .rename({n: 'index'})
            except:
                pass

        data_sam_s = xr.open_dataset(
            cases_data_sam + 'SAM_' + s + '_Leads_r_CFSv2.nc')

        case_events = SelectVariables(aux_cases, data_sam_s)

        case_events.to_netcdf(
            out_dir2 + 'sam_values_in_' + c + '_' + s + '.nc')

    processes = [Process(target=SelectEventsSAM, args=(c,)) for c in cases]
    for process in processes:
        process.start()
# ---------------------------------------------------------------------------- #
# # PENDIENTE confirmar primero lo de arriba
# # SST ##########################################################################
# if len(seasons)>1:
#     def SelectEvents(c):
#         for s in seasons:
#             aux_cases = xr.open_dataset(
#                 cases_date_dir + c + '_f_' + s + '.nc')\
#                 .rename({'__xarray_dataarray_variable__': 'index'})
#             data_sst_s = xr.open_dataset(data_dir + 'sst_' + s.lower() + '.nc')
#             case_events = SelectVariables(aux_cases, data_sst_s)
#             case_events.to_netcdf(out_dir + c + '_' + s + '.nc')
#
#
#     pool = ThreadPool(4)  # uno por season
#     pool.map_async(SelectEvents, [c for c in cases])
# else:
#     print('one season')
#     def SelectEvents(c):
#         s=seasons[0]
#         aux_cases = xr.open_dataset(cases_date_dir + c + '_f_' + s + '.nc') \
#             .rename({'__xarray_dataarray_variable__': 'index'})
#         data_sst_s = xr.open_dataset(data_dir + 'sst_' + s.lower() + '.nc')
#         case_events = SelectVariables(aux_cases, data_sst_s)
#         case_events.to_netcdf(out_dir + c + '_' + s + '.nc')
#
#
#     processes = [Process(target=SelectEvents, args=(c,)) for c in cases]
#     for process in processes:
#         process.start()
#
# # HGT ##########################################################################
# if len(seasons)>1:
#     def SelectEventsHGT(c):
#         for s in seasons:
#             try:
#                 aux_cases = xr.open_dataset(
#                     cases_date_dir + c + '_f_' + s + '.nc').\
#                     rename({'__xarray_dataarray_variable__': 'index'})
#             except:
#                 aux_cases = xr.open_dataset(
#                     cases_date_dir + c + '_f_' + s + '.nc')\
#                     .rename({'sst': 'index'})
#
#             data_hgt_s = xr.open_dataset(data_dir + 'hgt_' + s.lower() + '.nc')
#             case_events = SelectVariables(aux_cases, data_hgt_s)
#
#             case_events.to_netcdf(out_dir + 'hgt_' + c + '_' + s + '.nc')
#
#     pool = ThreadPool(4)  # uno por season
#     pool.map_async(SelectEventsHGT, [c for c in cases])
#
# else:
#     print('one season')
#     def SelectEventsHGT(c):
#         s = seasons[0]
#         try:
#             aux_cases = xr.open_dataset(
#                 cases_date_dir + c + '_f_' + s + '.nc')\
#                 .rename({'__xarray_dataarray_variable__': 'index'})
#         except:
#             aux_cases = xr.open_dataset(
#                 cases_date_dir + c + '_f_' + s + '.nc')\
#                 .rename({'sst': 'index'})
#
#         data_hgt_s = xr.open_dataset(data_dir + 'hgt_' + s.lower() + '.nc')
#
#         case_events = SelectVariables(aux_cases, data_hgt_s)
#         case_events.to_netcdf(out_dir + 'hgt_' + c + '_' + s + '.nc')
#
#
#     processes = [Process(target=SelectEventsHGT, args=(c,)) for c in cases]
#     for process in processes:
#         process.start()

# ---------------------------------------------------------------------------- #
# sam_index = xr.open_dataset(path + 'sam_rmon_r_z200.nc').\
#     rename({'time2':'time'})
# hgt = xr.open_dataset(path_aux + 'hgt_mon_anom_d.nc')
#
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
#
# async def SelectFields(event_time):
#     data_field = hgt
#
#     d = event_time[0]
#     l = event_time[1]
#     r = event_time[2]
#
#     aux = data_field.sel(time=data_field['L']==l)
#     aux = aux.sel(time=d, r=r)
#
#     return aux
#
# def SetOrder(dataset):
#     t = dataset['time'].values
#     L = dataset['L'].values
#     r = dataset['r'].values
#
#     values = [[t[i], L[i], r[i]] for i in range(len(t))]
#
#     return values
#
# async def main(index_selected):
#     datos = SetOrder(index_selected)
#     tareas = [SelectFields(event_time) for event_time in datos]
#
#     resultados = await asyncio.gather(*tareas)
#
#     return resultados
# # ---------------------------------------------------------------------------- #
# # Determinar categorias <, >, "neutro" y umbrales.
# sam_neutro = SelectEvents(sam_index, '<', '0.5', True)
# resultados = asyncio.run(main(sam_neutro))
# aux_xr = xr.concat(resultados, dim='time')
# if save:
#     aux_xr.to_netcdf(out_dir + 'sam_neutro_hgt200.nc')
#     sam_neutro.to_netcdf(out_dir + 'sam_neutro.nc')
#
# sam_pos = SelectEvents(sam_index, '>', '0.5')
# resultados = asyncio.run(main(sam_pos))
# aux_xr = xr.concat(resultados, dim='time')
# if save:
#     aux_xr.to_netcdf(out_dir + 'sam_pos_hgt200.nc')
#     sam_pos.to_netcdf(out_dir + 'sam_pos.nc')
#
# sam_neg = SelectEvents(sam_index, '<', '-0.5')
# resultados = asyncio.run(main(sam_neg))
# aux_xr = xr.concat(resultados, dim='time')
# if save:
#     aux_xr.to_netcdf(out_dir + 'sam_neg_hgt200.nc')
#     sam_neg.to_netcdf(out_dir + 'sam_neg.nc')
#
# # ---------------------------------------------------------------------------- #
# # ---------------------------------------------------------------------------- #
