"""
Ploteo de control de seleccion de events
Scatter 3D
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas_plots/'\
          'aux_3dframes/'
# ---------------------------------------------------------------------------- #
import xarray as xr
from funciones.aux_plot_3d_scatter_3index import generate_rotation_sequence

# ---------------------------------------------------------------------------- #
data_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
# ---------------------------------------------------------------------------- #
dmi = xr.open_dataset(data_dir + 'DMI_SON_Leads_r_CFSv2.nc')
ep = xr.open_dataset(data_dir + 'N34_SON_Leads_r_CFSv2.nc')
cp = xr.open_dataset(data_dir + 'SAM_SON_Leads_r_CFSv2.nc')

x = dmi.stack(time2=('time', 'r')).sst.values/dmi.std(['r', 'time']).sst.values
y = ep.stack(time2=('time', 'r')).sst.values/ep.std(['r', 'time']).sst.values
z = cp.stack(time2=('time', 'r')).sam.values/cp.std(['r', 'time']).sam.values

generate_rotation_sequence(x, y, z, out_dir=out_dir,
                           steps_per_transition=45,
                           save=save,
                           xlabel='DMI', ylabel='N34', zlabel='SAM')
# ---------------------------------------------------------------------------- #