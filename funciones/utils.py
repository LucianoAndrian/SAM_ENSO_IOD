import xarray as xr
import regionmask
import numpy as np

def MakeMask(DataArray, dataname='mask'):
    mask=regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(DataArray)
    mask = xr.where(np.isnan(mask), mask, 1)
    mask = mask.to_dataset(name=dataname)
    return mask