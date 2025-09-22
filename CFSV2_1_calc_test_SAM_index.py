"""
Calculo del indice SAM
El testeo se realizo previamente, decidiendo que el mejor modo era usar
stack en SON.
Se plotea el EOF para verificar.
"""

# ---------------------------------------------------------------------------- #
save_index = True
save_plot = False
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/index/'

# ---------------------------------------------------------------------------- #
import xarray as xr
from seteo_indices.CFSv2_SAM_index import Compute_SAM
from funciones.aux_plot_stereo_simple import plot_stereo

# ---------------------------------------------------------------------------- #
path = '/pikachu/datos/luciano.andrian/cases_fields/'
hgt = xr.open_dataset(path + 'hgt_mon_anom_d.nc') # VER de DONDE SALIO ESTOOO!
#hgt_son = hgt_sea.sel(time=hgt_sea.time.dt.month.isin(10)) # SON
hgt = hgt.interp(lon = hgt.lon.values[::2],
                 lat = hgt.lat.values[::2])
# ---------------------------------------------------------------------------- #
eof, sam = Compute_SAM(hgt)
plot_stereo(eof.sel(mode=0))

# ---------------------------------------------------------------------------- #
if save_index:
    sam.to_netcdf(f'{out_dir}sam_cfsv2_anual_index.nc')
