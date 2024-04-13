from tigramite.pcmci import PCMCI


"""
TEST Algoritmo de descubrimiento causal
en UN punto de grilla
"""

################################################################################
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
import pandas as pd
pd.options.mode.chained_assignment = None
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import xarray as xr
from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
################################################################################
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import xarray as xr
import pandas as pd
pd.options.mode.chained_assignment = None

from ENSO_IOD_Funciones import Nino34CPC, SameDateAs, DMI2
from PCMCI import PCMCI

import statsmodels.api as sm
import concurrent.futures
from datetime import datetime
import time

import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
################################################################################
save=False
use_sam=False
################################################################################
# ---------------------------------------------------------------------------- #
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
hgt_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/downloaded/'
#out_dir = ''
# ---------------------------------------------------------------------------- #
hgt = xr.open_dataset(hgt_dir + 'ERA5_HGT200_40-20.nc')
hgt = hgt.rename({'longitude': 'lon', 'latitude': 'lat', 'z': 'var'})
hgt = hgt.interp(lon=np.arange(0,360,2), lat=np.arange(-90, 90, 2))

hgt_clim = hgt.sel(time=slice('1979-01-01', '2000-12-01'))
hgt_anom = hgt.groupby('time.month') - \
           hgt_clim.groupby('time.month').mean('time')

if use_sam:
    weights = np.sqrt(np.abs(np.cos(np.radians(hgt_anom.lat))))
    hgt_anom = hgt_anom * weights

#hgt_anom2 = hgt_anom.sel(lat=slice(-80, 0), lon=slice(60, 70))
hgt_anom = hgt_anom.rolling(time=3, center=True).mean()
hgt_anom = hgt_anom.sel(time=slice('1940-02-01', '2020-11-01'))
hgt_anom = hgt_anom.sel(time=hgt_anom.time.dt.month.isin([8,9,10,11]))

################################################################################
# indices
# ---------------------------------------------------------------------------- #
sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam = sam.rolling(time=3, center=True).mean()

dmi = DMI2(filter_bwa=False, start_per='1920', end_per='2020',
           sst_anom_sd=False, opposite_signs_criteria=False)[2]

aux = xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
aux = aux.sel(time=slice('1920-01-01', '2020-12-01'))
n34 = Nino34CPC(aux, start=1920, end=2020)[0]
# ---------------------------------------------------------------------------- #
dmi2 = SameDateAs(dmi, hgt_anom)
n342 = SameDateAs(n34, hgt_anom)
sam2 = SameDateAs(sam, hgt_anom)

#sam3 = sam2
#c = c/c.std()
dmi3 = dmi2/dmi2.std()
n343 = n342/n342.std()
sam3 = sam2/sam2.std()

amd = hgt_anom.sel(lon=slice(210,290), lat=slice(-90,-50)).mean(['lon', 'lat'])
amd = amd/amd.std()
################################################################################
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sklearn
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb

from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys

# np.random.seed(42)     # Fix random seed
# links_coeffs = {0: [((0, -1), 0.7), ((1, -1), -0.8)],
#                 1: [((1, -1), 0.8), ((3, -1), 0.8)],
#                 2: [((2, -1), 0.5), ((1, -2), 0.5), ((3, -3), 0.6)],
#                 3: [((3, -1), 0.4)],
#                 }
# T = 100     # time series length
# data, true_parents_neighbors = toys.var_process(links_coeffs, T=T)
# T, N = data.shape

# Initialize dataframe object, specify time axis and variable names



data = np.array([amd['var'].values, dmi3.values, n343.values, sam3.values]).T
# T=100
# N=4
var_names = ['c', 'dmi3', 'n34', 'sam3']
dataframe = pp.DataFrame(data,
                         datatime = np.arange(len(data)),
                         var_names=var_names)

tp.plot_timeseries(dataframe); plt.show()

parcorr = ParCorr(significance='analytic')

pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)

results=pcmci.run_pcmci(tau_max=2,
                        tau_min=0,
                        pc_alpha=0.05,
                        max_combinations=1,
                        max_conds_px=None,
                        max_conds_py=None,
                        max_conds_dim=None,
                        selected_links=None,
                        alpha_level=0.05
                        )

correlations = pcmci.get_lagged_dependencies(tau_max=2, val_only=True)['val_matrix']
lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations,
                                   setup_args={'var_names':var_names,
                                    'x_base':5, 'y_base':.5}); plt.show()


print("p-values")
print (results['p_matrix'].round(3))
print("MCI partial correlations")
print (results['val_matrix'].round(2))
#
q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=2,
                                       fdr_method='fdr_bh')
pcmci.print_significant_links(
        p_matrix = results['p_matrix'],
        val_matrix = results['val_matrix'],
        alpha_level = 0.05)

tp.plot_graph(
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-corr'
    ); plt.show()

# med=LinearMediation(dataframe)
# med.fit_model(all_parents=parents, tau_max=3)

