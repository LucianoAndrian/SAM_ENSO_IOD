"""
Test eof indices
"""
# ---------------------------------------------------------------------------- #
save = True
save_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/eof_temporal/'
# ---------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from eofs.xarray import Eof
import matplotlib.dates as mdates
import pandas as pd
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
from ENSO_IOD_Funciones import Nino34CPC, DMI2, ChangeLons, DMI2_singlemonth, \
    Nino34CPC_singlemonth, DMI2_twomonths, Nino34CPC_twomonths, MakeMask, \
    SameDateAs
from cen_funciones import Detrend, Weights
# ---------------------------------------------------------------------------- #
if save:
    dpi = 200
else:
    dpi = 70

# Funciones ------------------------------------------------------------------ #
def SetData(indice):
    indice_norm = indice/indice.std('time')

    try:
        aux = xr.DataArray(
            indice_norm.values.reshape(-1, 12),
            coords=[np.unique(indice.time.dt.year.values), np.arange(1, 13)],
            dims=["time", "month"])
    except:
        aux = xr.DataArray(
            indice_norm.values.reshape(-1, 12),
            coords=[np.unique(indice.time.dt.year.values)[:-1], np.arange(1, 13)],
            dims=["time", "month"])

    return aux

def ApplyEOF(indice, neofs):
    indice_set = SetData(indice)

    solver = Eof(indice_set, center=False)
    eof_obs = solver.eofs(neofs=neofs)
    eof_var = np.around(solver.varianceFraction(neigs=neofs).values*100,1)

    return eof_obs, eof_var


def PlotLS(ln1, ln1_name, ln2, ln2_name, ln3, ln3_name, title, dpi,
           save, name_fig='', save_dir=save_dir, change_year=False):

    if change_year:
        meses = pd.date_range(start='1999-03-01', end='2000-03-01',
                              freq='M') + pd.DateOffset(days=1)
    else:
        meses = pd.date_range(start='1999-12-01', end='2000-12-01',
                              freq='M') + pd.DateOffset(days=1)

    fig = plt.figure(figsize=(8, 3.5), dpi=dpi)
    ax = fig.add_subplot(111)

    # Configurar el eje x para que solo muestre los meses
    ax.xaxis.set_major_locator(
        mdates.MonthLocator())  # Localizador para los meses
    ax.xaxis.set_major_formatter(mdates.DateFormatter(
        '%b'))  # Formato para mostrar solo el nombre corto del mes

    ax.plot(meses, ln1, color='k', label=ln1_name,
                  linewidth=1.5)

    ax.plot(meses, ln2, color='firebrick', label=ln2_name,
                  linewidth=1.5)

    ax.plot(meses, ln3, color='forestgreen', label=ln3_name,
                  linewidth=1.5)

    ax.hlines(y=0, xmin=meses[0], xmax=meses[-1], colors='k', linewidth=0.5)
    plt.legend(loc='upper left')
    plt.grid()
    ax.set_title(title, fontsize=15)

    if save:
        plt.savefig(f'{save_dir}/{name_fig}.jpg')
    else:
        plt.show()

# ---------------------------------------------------------------------------- #

dmi_3rm = DMI2(filter_bwa=False, start_per='1959', end_per='2020',
                  sst_anom_sd=False, opposite_signs_criteria=False)[2]

dmi_1rm = DMI2_singlemonth(filter_bwa=False, start_per='1959', end_per='2020',
                  sst_anom_sd=False, opposite_signs_criteria=False)[2]

dmi_2rm = DMI2_twomonths(filter_bwa=False, start_per='1959', end_per='2020',
                  sst_anom_sd=False, opposite_signs_criteria=False)[2]


sst_aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/"
                      "sst.mnmean.nc")
sst_aux = sst_aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34_3rm = Nino34CPC(sst_aux, start=1920, end=2020)[0]
n34_1rm = Nino34CPC_singlemonth(sst_aux, start=1920, end=2020)[0]
n34_2rm = Nino34CPC_twomonths(sst_aux, start=1920, end=2020)[0]

dmi_1rm = dmi_1rm.sel(time=slice('1959-04-01', '2020-03-01'))

dmi_2rm = SameDateAs(dmi_2rm, dmi_1rm)
dmi_3rm = SameDateAs(dmi_3rm, dmi_1rm)

n34_3rm = SameDateAs(n34_3rm, dmi_3rm)
n34_2rm = SameDateAs(n34_2rm, dmi_2rm)
n34_1rm = SameDateAs(n34_1rm, dmi_1rm)

u50_or = xr.open_dataset('/pikachu/datos/luciano.andrian/observado/'
                         'ncfiles/ERA5/downloaded/ERA5_U50hpa_40-20.mon.nc')
u50_or = u50_or.rename({'u': 'var'})
u50_or = u50_or.rename({'longitude': 'lon'})
u50_or = u50_or.rename({'latitude': 'lat'})
u50_or = Weights(u50_or)
u50_or = u50_or.sel(lat=-60)
#u50_or = u50_or - u50_or.mean('time')
u50_or = (u50_or.groupby('time.month') -
          u50_or.groupby('time.month').mean('time'))

u50_1rm = u50_or.rolling(time=1, center=True).mean()
u50_1rm = Detrend(u50_1rm, 'time')
u50_1rm = u50_1rm.sel(expver=1).drop('expver')
u50_1rm = u50_1rm.mean('lon')
u50_1rm = xr.DataArray(u50_1rm['var'].drop('lat'))

u50_2rm = u50_or.rolling(time=2, center=True).mean()
u50_2rm = Detrend(u50_2rm, 'time')
u50_2rm = u50_2rm.sel(expver=1).drop('expver')
u50_2rm = u50_2rm.mean('lon')
u50_2rm = xr.DataArray(u50_2rm['var'].drop('lat'))

u50_3rm = u50_or.rolling(time=3, center=True).mean()
u50_3rm = Detrend(u50_3rm, 'time')
u50_3rm = u50_3rm.sel(expver=1).drop('expver')
u50_3rm = u50_3rm.mean('lon')
u50_3rm = xr.DataArray(u50_3rm['var'].drop('lat'))

u50_3rm = SameDateAs(u50_3rm, dmi_3rm)
u50_2rm = SameDateAs(u50_2rm, dmi_2rm)
u50_1rm = SameDateAs(u50_1rm, dmi_1rm)
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
dmi_index = [dmi_1rm, dmi_2rm, dmi_3rm]
n34_index = [n34_1rm, n34_2rm, n34_3rm]
u50_index = [u50_1rm, u50_2rm, u50_3rm]

neofs = 2
for rm in [0,1,2]:

    dmi_eof, dmi_eof_var = ApplyEOF(dmi_index[rm], neofs)
    n34_eof, n34_eof_var = ApplyEOF(n34_index[rm], neofs)
    u50_eof, u50_eof_var = ApplyEOF(u50_index[rm], neofs)

    for neof in range(neofs):

        if np.abs(max(u50_eof[neof, :]))<np.abs(min(u50_eof[neof, :])):
            u50_eof[neof, :] = u50_eof[neof, :]*-1
        if np.abs(max(n34_eof[neof, :]))<np.abs(min(n34_eof[neof, :])):
            n34_eof[neof, :] = n34_eof[neof, :]*-1
        if np.abs(max(dmi_eof[neof, :]))<np.abs(min(dmi_eof[neof, :])):
            dmi_eof[neof, :] = dmi_eof[neof, :]*-1

        PlotLS(u50_eof[neof, :], f"u50 var. {np.round(dmi_eof_var[neof],3):.1f}%",
               n34_eof[neof, :], f"n34 var. {np.round(n34_eof_var[neof],3):.1f}%",
               dmi_eof[neof, :], f"dmi var. {np.round(u50_eof_var[neof],3):.1f}%",
               f'Indices {rm+1}rm - {neof+1}º Eof', 300, save,
               f'Indices_{rm+1}_rm-{neof+1}', change_year=True)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #