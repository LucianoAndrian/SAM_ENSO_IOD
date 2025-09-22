import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.path as mpath

def plot_stereo(dataarray):
    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=0)},
        figsize=(6, 6)
    )

    lons = dataarray.lon
    lats = dataarray.lat
    field = dataarray.values

    levels = [-75, -50, -35, -20, -10, -5, 0,
               5, 10, 20, 30, 50, 75]

    cf = ax.contourf(
        lons, lats, field,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r", levels=levels, extend='both'
    )
    ax.contour(
        lons, lats, field,
        transform=ccrs.PlateCarree(),
        colors='k', levels=levels, linewidths=0.5
    )

    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', fraction=0.05,
                        pad=0.1)

    ax.set_extent([-180, 180, -20, -90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                 lw=0.3, color="gray",
                 xlocs=range(-180, 180, 30),
                 ylocs=np.arange(-80, -20, 20))

    r_extent = .8e7
    ax.set_xlim(-r_extent, r_extent)
    ax.set_ylim(-r_extent, r_extent)
    circle_path = mpath.Path.unit_circle()
    circle_path = mpath.Path(circle_path.vertices.copy() * r_extent,
                             circle_path.codes.copy())
    ax.set_boundary(circle_path)
    ax.set_frame_on(False)

    plt.show()