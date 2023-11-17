# necesito datos anuales
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath

from eofs.xarray import Eof
from ENSO_IOD_Funciones import SameDateAs
# ---------------------------------------------------------------------------- #
ruta = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
# ---------------------------------------------------------------------------- #
# Funciones ###################################################################
def Detrend(xrda, dim):
    aux = xrda.polyfit(dim=dim, deg=1)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients)
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients)
    dt = xrda - trend
    return dt

def plot_stereo(dataarray, variance, n):
    import Scales_Cbars
    cbar = Scales_Cbars.get_cbars('hgt200')
    #scale = Scales_Cbars.get_scales('hgt200')
    scale = [-300,-250, -200, -150, -100, -50, -25,
             0, 25, 50, 100, 150, 200, 250, 300]
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
################################################################################
hgt = xr.open_dataset(ruta + 'ERA5_HGT750_40-20.nc')
hgt = hgt.rename({'longitude': 'lon', 'latitude': 'lat', 'z': 'var'})
# Interp --------------------------------------------------------------------- #
hgt = hgt.sel(lat=slice(-20, -90))
hgt = hgt.interp(lon=np.arange(0,360,.5), lat=np.arange(-90, 90, .5))

# ---------------------------------------------------------------------------- #
hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
hgt_anom = hgt.groupby('time.month') - \
           hgt_clim.groupby('time.month').mean('time')

weights = np.sqrt(np.abs(np.cos(np.radians(hgt_anom.lat))))
hgt_anom = hgt_anom * weights

hgt_anom = hgt_anom.sel(lat=slice(None, -20))
# test eof seaons
hgt_anom = hgt_anom.rolling(time=3, center=True).mean()
hgt_anom = hgt_anom.sel(time=hgt_anom.time.dt.month.isin(10))

# ---------------------------------------------------------------------------- #
solver = Eof(xr.DataArray(hgt_anom['var']))
eof = solver.eofsAsCovariance(neofs=3)
pcs = solver.pcs()

var_per = np.around(solver.varianceFraction(neigs=3).values*100,1)
for n in [0,1,2]:
    aux = eof[n]
    plot_stereo(aux, var_per, n+1)

sam_test = -1*pcs[:,0]/pcs[:,0].std()
# ---------------------------------------------------------------------------- #
sam_noaa = pd.read_fwf('/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
                       'sam_noaa.txt', skiprows = 0, header=None)
sam_noaa = sam_noaa.loc[sam_noaa[0]<=2020]
year = sam_noaa.iloc[:, 0]
dates = xr.cftime_range(start=str(sam_noaa[0][0]) + '-01-01',
                        end='2020-12-01', freq='MS')
values = sam_noaa.iloc[:, 2:].values.flatten()
sam_noaa = xr.Dataset(
    {'sam': (['time'], values)},
    coords={'time': dates}
)

sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
sam_ec = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']

sam_ec = SameDateAs(sam_ec, sam_noaa)
sam_test = SameDateAs(sam_test, sam_noaa)

plt.figure(figsize=(10, 6))
plt.plot(sam_test, color='k', alpha=0.3)
plt.plot(sam_test.rolling(time=3, center=True).mean(), color='k',
         label='SAM test')
plt.plot(sam_ec, color='lime', alpha=0.3)
plt.plot(sam_ec.rolling(time=3, center=True).mean(), color='lime',
         label='SAM_EC')
plt.plot(sam_noaa.sam, color='tomato', alpha=0.3)
plt.plot(sam_noaa.rolling(time=3, center=True).mean().sam, color='tomato',
         label='SAM NOAA')

plt.title('SAMs, r = .98-.99' )
plt.ylabel('SAM')
plt.legend()
plt.grid(True)
plt.show()

# 3rmean
# 0.98 corr con sam_ec \ 0.99 sin detrned
# 0.96 corr con noaa \ 0.98 sin detrend
#
# monthly
# 0.98 corr con sam_nc \ 0.99 sin detrend
# 0.97 corr con noaa \ 0.98 sin detrend
################################################################################