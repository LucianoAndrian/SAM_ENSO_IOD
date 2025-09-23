"""
Seteoo de SAM index en CFSv2 igual que DMI y N34
(adaptable para posibles cambios o mas seasons)
"""
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'

# ---------------------------------------------------------------------------- #
import xarray as xr
index_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/index/'

# ---------------------------------------------------------------------------- #
def SelectSeason(data, mm_season):
    aux_l = []
    for l in data.L.values:
        aux_l.append(data.sel(time=data.time.dt.month.isin(mm_season - l), L=l))

    return xr.concat(aux_l, dim='time')
# ---------------------------------------------------------------------------- #
sam = xr.open_dataset(f'{index_dir}sam_cfsv2_anual_index.nc')
sam = sam.rename({'time2': 'time', 'pcs':'sam'})
sam = sam.drop_vars(['mode', 'month'])
sam_3rm = sam.rolling(time=3, center=True).mean() # seasons

sam_3rm_son = SelectSeason(sam_3rm, 10)

sam_3rm_son.to_netcdf(f'{out_dir}SAM_SON_Leads_r_CFSv2.nc')
# ---------------------------------------------------------------------------- #