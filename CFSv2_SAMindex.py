"""
TEST
Calculo SAM en CFSv2
"""
save = False
# ---------------------------------------------------------------------------- #
path = '/pikachu/datos/luciano.andrian/cases_fields/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/eof/'

if save:
    dpi=300
else:
    dpi=100
# ---------------------------------------------------------------------------- #
from eofs.xarray import Eof
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath

import sys
sys.path.append('/home/auri/Facultad/Doc/scrips/ENSO_IOD_SAM/')

def plot_stereo(dataarray, variance, n, lead, save, dpi, aux_name):
    import Scales_Cbars
    cbar = Scales_Cbars.get_cbars('hgt200')
    scale = Scales_Cbars.get_scales('hgt200')
    scale = [-75, -50, -35, -20, -10, -5, 0,
             5, 10, 20, 30, 50, 75]
    fig, ax = plt.subplots(dpi=dpi,
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
    plt.title('EOF '+ str(n) + ' - ' + str(variance[n-1]) + '%' + ' Lead: ' +
              str(lead))
    name_fig = aux_name + 'z200_EOF_' + str(n) + '_Lead_' + str(lead)
    if save:
        print('save: ' + out_dir + name_fig + '.jpg')
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()

    else:
        plt.show()

def Compute(hgt, aux_name_r, aux_name_em, save, modeL=False):

    weights = np.sqrt(np.abs(np.cos(np.radians(hgt.lat))))

    if modeL:
        aux = hgt*weights
        # r y time a a una misma variable para Eof
        aux_st = aux.rename({'time': 'time2'})
        aux_st = aux_st.stack(time=('r', 'time2'))
        aux_st = aux_st.transpose('time', 'lat', 'lon')

        # eof ------------------------------------#
        try:
            solver = Eof(aux_st['hgt'])

        except ValueError as ve:
            if str(ve) == 'all input data is missing':
                print('campos faltantes')
                aux_st = aux_st.where(~np.isnan(aux_st), drop=True)
                solver = Eof(aux_st['hgt'])

        eof_L_r = solver.eofsAsCovariance(neofs=3)
        pcs = solver.pcs()

        # # SAM index -------------------------------#
        # sam_L_r = -1 * pcs[:, 0] / pcs[:, 0].std()
        # sam_L_r = sam_L_r.unstack('time')

        # Plot Eof
        var_per = np.around(solver.varianceFraction(neigs=3).values * 100, 1)
        for n in [0, 1, 2]:
            plot_stereo(eof_L_r[n], var_per, n + 1, 'todos', save, dpi,
                        aux_name_r)

    else:
        # por leads ---------------------------------------------------------- #
        for l in [0, 1, 2, 3]:
            print('L:' + str(l))
            # Todos las runs ------------------------------------------------- #

            aux = hgt.sel(time=hgt['L'] == l) * weights  # .mean('r')

            # r y time a a una misma variable para Eof
            aux_st = aux.rename({'time': 'time2'})
            aux_st = aux_st.stack(time=('r', 'time2'))
            aux_st = aux_st.transpose('time', 'lat', 'lon')

            # eof ------------------------------------#
            try:
                solver = Eof(aux_st['hgt'])

            except ValueError as ve:
                if str(ve) == 'all input data is missing':
                    print('Lead ' + str(l) + ' con campos faltantes')
                    aux_st = aux_st.where(~np.isnan(aux_st), drop=True)
                    solver = Eof(aux_st['hgt'])

            eof_L_r = solver.eofsAsCovariance(neofs=3)
            pcs = solver.pcs()

            # SAM index -------------------------------#
            sam_L_r = -1 * pcs[:, 0] / pcs[:, 0].std()
            sam_L_r = sam_L_r.unstack('time')

            # Plot Eof
            var_per = np.around(solver.varianceFraction(neigs=3).values * 100,
                                1)
            for n in [0, 1, 2]:
                plot_stereo(eof_L_r[n], var_per, n + 1, l, save, dpi,
                            aux_name_r)

            print('Done EOF r')
            del aux_st
            del solver

            # Media del ensamble --------------------------------------------- #
            aux = aux.mean('r')

            # eof ------------------------------------#
            try:
                solver = Eof(xr.DataArray(aux['hgt']))

            except ValueError as ve:
                if str(ve) == 'all input data is missing':
                    print('Lead ' + str(l) + ' con campos faltantes')
                    aux = aux.where(~np.isnan(aux), drop=True)
                    solver = Eof(xr.DataArray(aux['hgt']))

            eof_L_em = solver.eofsAsCovariance(neofs=3)
            pcs = solver.pcs()

            # SAM index -------------------------------#
            sam_L_em = -1 * pcs[:, 0] / pcs[:, 0].std()

            # Plot Eof
            var_per = np.around(solver.varianceFraction(neigs=3).values * 100,
                                1)
            for n in [0, 1, 2]:
                plot_stereo(eof_L_em[n], var_per, n + 1, l, save, dpi,
                            aux_name_em)
            print('Done EOF em')

            # para guardar...
            if l == 0:
                sam_r = sam_L_r.drop('L')
                eof_r = eof_L_em

                sam_em = sam_L_em.drop('L')
                eof_em = eof_L_em
            else:
                eof_r = xr.concat([eof_r, eof_L_r], dim='time')
                sam_r = xr.concat([sam_r, sam_L_r.drop('L')], dim='L')

                eof_em = xr.concat([eof_em, eof_L_em], dim='time')
                sam_em = xr.concat([sam_em, sam_L_em.drop('L')], dim='L')

            print('Done concat')

        # -------------------------------------------------------------------- #
        if save:
            eof_r.to_netcdf(out_dir + 'eof_r' + aux_name_r + '_z200.nc')
            sam_r.to_netcdf(out_dir + 'sam_r' + aux_name_r + '_z200.nc')

            eof_em.to_netcdf(out_dir + 'eof_em' + aux_name_r + '_z200.nc')
            sam_em.to_netcdf(out_dir + 'sam_em' + aux_name_r + '_z200.nc')
        # -------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
hgt = xr.open_dataset(path + 'hgt_mon_anom_d.nc')
hgt_sea = xr.open_dataset(path + 'hgt_seas_anom_d.nc')
hgt_son = hgt_sea.sel(time=hgt_sea.time.dt.month.isin(10)) # SON
# Compute ---------------------------------------------------------------------#
Compute(hgt, 'mon_r', 'mon_em', save)
Compute(hgt_sea, 'sea_r', 'sea_em', save)
Compute(hgt_son, 'SON_r', 'SON_em', False, True)
Compute(hgt_son, 'SON_r', 'SON_em', False, True)
print('#######################################################################')
print('done')
print('#######################################################################')
# Test Select
# sam = xr.open_dataset(out_dir + 'sam_rSON_r_z200.nc').rename({'time2':'time'})
# hgt = xr.open_dataset(path + 'hgt_mon_anom_d.nc')
#
# hgt_l=hgt.sel(time=hgt['L']==3)
# sam_p= hgt_l.where(sam.pcs[3,:,:]>0)
#
# plt.imshow(sam_p.mean(['r','time']).hgt);plt.colorbar();plt.show()
