"""
TEST
Calculo SAM en CFSv2
"""
# ---------------------------------------------------------------------------- #
path = '/pikachu/datos/luciano.andrian/cases_fields/'
# ---------------------------------------------------------------------------- #
import xarray as xr
from eofs.xarray import Eof

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath

def plot_stereo(dataarray, variance, n):
    import Scales_Cbars
    cbar = Scales_Cbars.get_cbars('hgt200')
    scale = Scales_Cbars.get_scales('hgt200')
    scale = [-75, -50, -35, -20, -10, -5, 0,
             5, 10, 20, 30, 50, 75]
    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=0)})

    lons = dataarray.lon
    lats = dataarray.lat
    field = dataarray.values
    try:
        cf = ax.contourf(lons, lats, field[0, :, :],
                         transform=ccrs.PlateCarree(),
                         cmap=cbar, levels=scale, extend='both')
        ax.contour(lons, lats, field[0, :, :], transform=ccrs.PlateCarree(),
                   colors='k', levels=scale)
    except:
        cf = ax.contourf(lons, lats, field,
                         transform=ccrs.PlateCarree(),
                         cmap=cbar, levels=scale, extend='both')
        ax.contour(lons, lats, field, transform=ccrs.PlateCarree(),
                   colors='k', levels=scale)


    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', fraction=0.05,
                        pad=0.1)
    cbar.set_label('Values')

    ax.set_extent([-180, 180, -20, -90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    gls = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), lw=0.3,
                           color="gray",
                           y_inline=True, xlocs=range(-180, 180, 30),
                           ylocs=np.arange(-80, -20, 20))
    r_extent = .8e7
    ax.set_xlim(-r_extent, r_extent)
    ax.set_ylim(-r_extent, r_extent)
    circle_path = mpath.Path.unit_circle()
    circle_path = mpath.Path(circle_path.vertices.copy() * r_extent,
                             circle_path.codes.copy())
    ax.set_boundary(circle_path)
    ax.set_frame_on(False)
    plt.draw()
    plt.title('EOF '+ str(n) +' - ' + str(variance[n-1]) + '%')

    plt.show()
# ---------------------------------------------------------------------------- #
hgt = xr.open_dataset(path + 'hgt_mon_anom_d.nc')

hgt = hgt.sel(time=hgt.time.dt.month.isin(10))
weights = np.sqrt(np.abs(np.cos(np.radians(hgt.lat))))
hgt = hgt * weights
#hgt = hgt.compute()
aux_hgt = hgt.rename({'time':'time2'})
hgt_st = aux_hgt.stack(time=('r', 'time2'))
solver = Eof(xr.DataArray(hgt['hgt']))
eof = solver.eofsAsCovariance(neofs=3)
pcs = solver.pcs()

var_per = np.around(solver.varianceFraction(neigs=3).values*100,1)
for n in [0,1,2]:
    aux = eof[n]
    plot_stereo(aux.mean('r'), var_per, n+1)

sam_test = -1*pcs[:,0]/pcs[:,0].std()