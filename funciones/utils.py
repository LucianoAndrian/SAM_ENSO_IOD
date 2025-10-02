import xarray as xr
import regionmask
import numpy as np

def MakeMask(DataArray, dataname='mask'):
    mask=regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(DataArray)
    mask = xr.where(np.isnan(mask), mask, 1)
    mask = mask.to_dataset(name=dataname)
    return mask

def SameDateAs(data, datadate):
    """
    En data selecciona las mismas fechas que datadate
    :param data:
    :param datadate:
    :return:
    """
    return data.sel(time=datadate.time.values)

def change_name_dim(data, dim_to_check, dim_to_rename):
    if dim_to_check in list(data.dims):
        out_put =  data.rename({dim_to_check: dim_to_rename})
    else:
        out_put = data
    return out_put

def change_name_variable(data, new_name_var):
    try:
        out_put = data.rename({list(data.data_vars)[0]: new_name_var})
    except:
        out_put = data
        pass
    return out_put