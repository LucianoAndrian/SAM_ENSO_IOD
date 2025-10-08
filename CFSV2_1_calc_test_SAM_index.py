"""
Calculo del indice SAM
EOF en 200hgt (previamente testeado) en 20ºS-90ºS aplicado sobre
hindcast + realtime para todos los miembros de ensamble y leads
Interpolado a 2º x 2º por RAM del servidor
Se plotea el EOF para verificar.
"""
# ---------------------------------------------------------------------------- #
save_index = True
save_plot = True
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/index/'
out_dir_plot = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas_plots/'

# ---------------------------------------------------------------------------- #
import xarray as xr
from seteo_indices.CFSv2_SAM_index import Compute_SAM
from funciones.aux_plot_stereo_simple import plot_stereo_hgt

# ---------------------------------------------------------------------------- #
path = '/pikachu/datos/luciano.andrian/cases_fields/'
hgt = xr.open_dataset(path + 'hgt_mon_anom_d.nc')

# Por ram del servidor: de 1º x 1º -> 2º x 2º
hgt = hgt.interp(lon = hgt.lon.values[::2], lat = hgt.lat.values[::2])
# ---------------------------------------------------------------------------- #
eof, sam = Compute_SAM(hgt)
plot_stereo_hgt(eof.sel(mode=0), save=save_plot,
                name_fig='eof_mon_anual_hgt200',
                out_dir=out_dir_plot)

# ---------------------------------------------------------------------------- #
if save_index:
    sam.to_netcdf(f'{out_dir}sam_cfsv2_anual_index.nc')
    print(f'SAM index saved: {out_dir}sam_cfsv2_anual_index.nc')
# ---------------------------------------------------------------------------- #