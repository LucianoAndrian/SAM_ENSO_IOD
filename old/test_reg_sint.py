"""
test campos sinteticos a partir de mlr
No hay correlación xq no da nada nuevo
"""
################################################################################
save = True
################################################################################
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/mlr_test/'
era5_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
t_pp_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_obs_d_w_c/'
################################################################################
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2, CreateDirectory, \
    PlotReg
import Scales_Cbars
from eofs.xarray import Eof
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
################################################################################
if save:
    dpi = 100 # no es pa tanto la cosa
else:
    dpi = 100
################################################################################
def NormSD(serie):
    return serie / serie.std('time')

def RemoveEffect(predictand, predictor):
    try:
        model = sm.OLS(predictand.values, predictor.values)
    except:
        model = sm.OLS(predictand, predictor)

    result = model.fit()
    predictand_res = predictand - result.predict()
    return predictand_res

def regression_func(target_values, *predictor_values):
    predictors_list = list(predictor_values)
    predictors = np.column_stack(predictors_list)
    predictors = sm.add_constant(predictors)
    model = sm.OLS(target_values, predictors, hasconst=False)
    results = model.fit()
    return results.params, results.predict(), results.pvalues

def compute_regression(variable, *index):
    input_core_dims = [['time']] + [['time']] * len(index)

    coef_dataset, predict, pvalues = xr.apply_ufunc(
        regression_func,
        variable, *index,
        input_core_dims=input_core_dims,
        output_core_dims=[['coefficients'], ['time'], ['pvalues']],
        output_dtypes=[float, float, float],
        vectorize=True
        #dask='parallelized' +21 seg
    )
    return coef_dataset, predict, pvalues

def MakerMaskSig(data, pvalue):
    mask_sig = data.where(data <= pvalue)
    mask_sig = mask_sig.where(np.isnan(mask_sig), 1)

    return mask_sig

def Plot(data, data_pv, pvalue, title, name_fig, dpi, save, i=0,
         VarName='hgt200'):

    if VarName.lower() == 'hgt200':
        SA_map = False
        pvalue_mask = 0.1
        sig = False
        sig_point = False
        two_variables = True
    else:
        SA_map = True
        pvalue_mask = 1
        sig = True
        sig_point = True
        two_variables = False

    aux = data * MakerMaskSig(data_pv, pvalue_mask)
    aux = aux['var'][:,:,i,i]
    aux2 = data['var'][:,:,i]

    cbar = Scales_Cbars.get_cbars(VarName)
    scale = Scales_Cbars.get_scales(VarName)

    PlotReg(data=aux,
            data_cor=data_pv.sel(pvalues=i), SA=SA_map,
            levels=scale, sig=sig,
            two_variables=two_variables, data2=aux2, sig2=False,
            levels2=scale, title=title, name_fig=name_fig,
            out_dir=out_dir, save=save, cmap=cbar, dpi=dpi,
            color_map='grey', sig_point=sig_point, color_sig='k', r_crit=pvalue)

def SelectSeason_and_Variable(sam, dmi, n34, VarName, mm):

    # Seleccion de seaons en los índices
    sam_season = NormSD(sam.sel(time=sam.time.dt.month.isin(mm)))
    dmi_season = NormSD(SameDateAs(dmi, sam_season))
    n34_season = NormSD(SameDateAs(n34, sam_season))

    # Variable en la season
    if mm==10:
        season_name = 'SON'
    elif mm==7:
        season_name = 'JJA'
    else:
        return print('Season solo JJA o SON')

    if VarName.lower() == 'hgt200':
        file = 'HGT200_' + season_name + '_mer_d_w.nc'
        path = era5_dir
    elif VarName.lower() == 'pp':
        file = 'ppgpcc_w_c_d_1_' + season_name + '.nc'
        path = t_pp_dir
    elif VarName.lower() == 't':
        file = 'tcru_w_c_d_0.25_' + season_name + '.nc'
        path = t_pp_dir

    var_season = xr.open_dataset(path + file)

    return sam_season, dmi_season, n34_season, var_season

def plot_stereo(dataarray, variance, n, title, save, name_fig, dpi):
    import Scales_Cbars
    cbar = Scales_Cbars.get_cbars('hgt200')

    scale = [-300,-250, -200, -150, -100, -50, -25,
             0, 25, 50, 100, 150, 200, 250, 300]
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
    ax.add_feature(cfeature.COASTLINE, edgecolor='#4F514F')
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
    ax.set_frame_on(True)

    plt.draw()
    plt.title(title +' EOF '+ str(n) +' - ' + str(variance[n-1]) + '%')
    if save:
        print('save: ' + out_dir + name_fig + '.jpg')
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

def Compute(indices, indices_name, name, save, dpi):
    if len(indices)==3:
        mlr_params, mlr_pred, mlr_pv = \
            compute_regression(var_p, indices[0], indices[1], indices[2])
        count = [1,2,3]

    elif len(indices)==2:
        mlr_params, mlr_pred, mlr_pv = \
            compute_regression(var_p, indices[0], indices[1])
        count = [1, 2]

    elif len(indices)==1:
        mlr_params, mlr_pred, mlr_pv = \
            compute_regression(var_p, indices[0])
        count = [1]
    else:
        print('Error: indices')
        return

    for n, n_count in zip(indices_name, count):
        Plot(mlr_params, mlr_pv, 0.1,
             f"{VarName} - {name} - {n} Coef. - {s_name}",
             f"{VarName}_{name}_{n.lower()}_c_{s_name}",
             dpi, save, n_count, VarName)

    mlr_pred = mlr_pred.transpose('time', 'lat', 'lon')
    solver = Eof(xr.DataArray(mlr_pred['var']))
    eof = solver.eofsAsCovariance(neofs=3)
    #pcs = solver.pcs()

    var_per = np.around(solver.varianceFraction(neigs=3).values*100,1)
    for i in [0,1,2]:
        aux = eof[i]
        plot_stereo(aux, var_per, i+1,
                    title=f"{VarName} - {name} - {s_name} - ",
                    save=save, dpi=dpi,
                    name_fig=f"{VarName}_EOF{i}_{name}_{s_name}")

################################################################################
# indices
sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam = sam.rolling(time=3, center=True).mean()

dmi = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

aux = xr.open_dataset(
    "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
n34 = Nino34CPC(aux, start=1920, end=2020)[0]
################################################################################
periodos = [[1940, 2020]]
VarName= 'hgt200'
mm = 10
s_name = 'SON'
p = periodos[0]
print('-----------------------------------------------------------------------')
sam_season, dmi_season, n34_season, var_season = \
    SelectSeason_and_Variable(sam, dmi, n34, VarName, mm)

sam_p = sam_season.sel(time=slice(str(p[0]) + '-10-01', str(p[1]) + '-10-01'))
n34_p = SameDateAs(n34_season, sam_p)
dmi_p = SameDateAs(dmi_season, sam_p)
try:
    var_p = SameDateAs(var_season, sam_p)
except:
    var_p = var_season.sel(time=slice(str(p[0]) + '-10-01',
                                      str(p[1]) + '-10-31'))
    var_p['time'] = sam_p.time.values

var_p = var_p.interp(lon=np.arange(0,360,.5), lat=np.arange(-80, 20, .5)[::-1])
weights = np.sqrt(np.abs(np.cos(np.radians(var_p.lat))))
var_p = var_p * weights

print('Regression ------------------------------------------------------------')
dmi_wo_n34 = RemoveEffect(dmi_p, n34_p)
sam_wo_n34 = RemoveEffect(sam_p, n34_p)
n34_wo_dmi = RemoveEffect(n34_p, dmi_p)
sam_wo_dmi = RemoveEffect(sam_p, dmi_p)
dmi_wo_sam = RemoveEffect(dmi_p, sam_p)
n34_wo_sam = RemoveEffect(n34_p, sam_p)
print('-----------------------------------------------------------------------')
print('DMI + N34 + SAM -------------------------------------------------------')
Compute([dmi_p, n34_p, sam_p], ['DMI', 'N34', 'SAM'], 'MLR', save, dpi)

print('DMI + N34--------------------------------------------------------------')
Compute([dmi_p, n34_p], ['DMI', 'N34'], 'MLR_DMI-N34', save, dpi)

print('DMI + N34 wo. SAM -----------------------------------------------------')
Compute([dmi_wo_sam, n34_wo_sam], ['DMI', 'N34'], 'MLR_DMI-N34_woSAM', save, dpi)

print('DMI + SAM--------------------------------------------------------------')
Compute([dmi_p, sam_p], ['DMI', 'SAM'], 'MLR_DMI-SAM', save, dpi)

print('DMI + SAM wo. N34------------------------------------------------------')
Compute([dmi_wo_n34, sam_wo_n34], ['DMI', 'SAM'], 'MLR_DMI-SAM_woN34', save, dpi)

print('N34 + SAM--------------------------------------------------------------')
Compute([n34_p, sam_p], ['N34', 'SAM'], 'MLR_N34-SAM', save, dpi)

print('N34 + SAM wo. DMI------------------------------------------------------')
Compute([n34_wo_dmi, sam_wo_dmi], ['N34', 'SAM'], 'MLR_N34-SAM_woDMI', save, dpi)
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------')
print('done')
print('-----------------------------------------------------------')


