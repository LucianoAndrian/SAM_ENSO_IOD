"""
Scatter plots
ENSO-IOD-SAM CFSv2
"""
# ---------------------------------------------------------------------------- #
save = False
out_dir_plot = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas_plots/'

# -
import xarray as xr
from funciones.plots import PlotScatter


events_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/events_variables/'
index_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'

# ---------------------------------------------------------------------------- #
sd_dmi = xr.open_dataset(f'{index_dir}DMI_SON_Leads_r_CFSv2.nc').std()
sd_n34 = xr.open_dataset(f'{index_dir}N34_SON_Leads_r_CFSv2.nc').std()
sd_sam = xr.open_dataset(f'{index_dir}SAM_SON_Leads_r_CFSv2.nc').std()

# DMI vs N34
PlotScatter(idx1_name='DMI', idx2_name='N34',
            idx1_sd=sd_dmi, idx2_sd=sd_n34, save=save,
            name_fig='Scatter_DMI-EP',
            cases_dir=events_dir, index_dir=index_dir,
            remove_cfsv2_str=True)

# DMI vs SAM
PlotScatter(idx1_name='DMI', idx2_name='SAM',
            idx1_sd=sd_dmi, idx2_sd=sd_sam, save=save,
            name_fig='Scatter_DMI-EP',
            cases_dir=events_dir, index_dir=index_dir,
            remove_cfsv2_str=True)

# SAM vs N34
PlotScatter(idx1_name='N34', idx2_name='SAM',
            idx1_sd=sd_n34, idx2_sd=sd_sam, save=save,
            name_fig='Scatter_DMI-EP',
            cases_dir=events_dir, index_dir=index_dir,
            remove_cfsv2_str=True)
# ---------------------------------------------------------------------------- #