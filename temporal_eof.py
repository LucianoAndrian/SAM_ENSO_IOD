"""
Test eof indices
"""
# ---------------------------------------------------------------------------- #
save = False
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

    aux = xr.DataArray(
        indice_norm.values.reshape(-1, 12),
        coords=[np.unique(indice.time.dt.year.values), np.arange(1, 13)],
        dims=["time", "month"])

    return aux

def ApplyEOF(indice, neofs):
    indice_set = SetData(indice)

    solver = Eof(indice_set, center=False)
    eof_obs = solver.eofs(neofs=neofs)
    eof_var = np.around(solver.varianceFraction(neigs=neofs).values*100,1)

    return eof_obs, eof_var


def PlotLS(ln1, ln1_name, ln2, ln2_name, ln3, ln3_name, title, dpi):
    meses = pd.date_range(start='1999-12-01', end='2000-12-01',
                          freq='M') + pd.DateOffset(days=1)


    fig = plt.figure(figsize=(7, 3), dpi=dpi)
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

    plt.legend()
    plt.grid()
    ax.set_title(title, fontsize=15)

    plt.show()

# ---------------------------------------------------------------------------- #

dmi_or_3rm = DMI2(filter_bwa=False, start_per='1959', end_per='2020',
                  sst_anom_sd=False, opposite_signs_criteria=False)[2]

dmi_or_1rm = DMI2_singlemonth(filter_bwa=False, start_per='1959', end_per='2020',
                  sst_anom_sd=False, opposite_signs_criteria=False)[2]

dmi_or_2rm = DMI2_twomonths(filter_bwa=False, start_per='1959', end_per='2020',
                  sst_anom_sd=False, opposite_signs_criteria=False)[2]


sst_aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/"
                      "sst.mnmean.nc")
sst_aux = sst_aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34_or_3rm = Nino34CPC(sst_aux, start=1920, end=2020)[0]
n34_or_1rm = Nino34CPC_singlemonth(sst_aux, start=1920, end=2020)[0]
n34_or_2rm = Nino34CPC_twomonths(sst_aux, start=1920, end=2020)[0]

n34_or_3rm = SameDateAs(n34_or_3rm, dmi_or_3rm)
n34_or_2rm = SameDateAs(n34_or_2rm, dmi_or_2rm)
n34_or_1rm = SameDateAs(n34_or_1rm, dmi_or_1rm)

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

u50_3rm = SameDateAs(u50_3rm, dmi_or_3rm)
u50_2rm = SameDateAs(u50_2rm, dmi_or_2rm)
u50_1rm = SameDateAs(u50_1rm, dmi_or_1rm)
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

dmi_eof, dmi_eof_var = ApplyEOF(dmi_or_3rm, 4)
n34_eof, n34_eof_var = ApplyEOF(n34_or_3rm, 4)
u50_eof, u50_eof_var = ApplyEOF(u50_3rm, 4)

PlotLS(u50_eof[0,:], f"u50 var. {dmi_eof_var[0]}%",
       n34_eof[0,:], f"n34 var. {n34_eof_var[0]}%",
       dmi_eof[0,:], f"dmi var. {u50_eof_var[0]}%",
       '1er eof',
       300)

PlotLS(u50_eof[1,:], f"u50 var. {dmi_eof_var[1]}%",
       n34_eof[1,:], f"n34 var. {n34_eof_var[1]}%",
       dmi_eof[1,:], f"dmi var. {u50_eof_var[1]}%",
       '2do eof',
       300)

PlotLS(u50_eof[2,:], f"u50 var. {dmi_eof_var[2]}%",
       n34_eof[2,:], f"n34 var. {n34_eof_var[2]}%",
       dmi_eof[2,:], f"dmi var. {u50_eof_var[2]}%",
       '3er eof',
       300)

PlotLS(u50_eof[3,:], f"u50 var. {dmi_eof_var[3]}%",
       n34_eof[3,:], f"n34 var. {n34_eof_var[3]}%",
       dmi_eof[3,:], f"dmi var. {u50_eof_var[3]}%",
       '4to eof',
       300)
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #