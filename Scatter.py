"""
Scatters que no dicen nada...
Replicando los scatter plots anteriores
"""
################################################################################
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/scatter/'

################################################################################
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from ENSO_IOD_Funciones import DMI, Nino34CPC, SameDateAs
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
################################################################################
save = False
# seasons = ['JJA', 'SON']
# mmonth = [7, 10]

seasons = ['SON']
mmonth = [10]


if save:
    dpi = 300
else:
    dpi = 100
################################################################################
def ScatterPlot(xserie, yserie, xlabel, ylabel, title, name_fig, dpi, save):
    fig, ax = plt.subplots(dpi=dpi)

    plt.scatter(x=xserie, y=yserie, marker='o', s=20,
                edgecolor='k', color='dimgray', alpha=1)

    plt.ylim((-5, 5))
    plt.xlim((-5, 5))
    plt.axhspan(-.5, .5, alpha=0.2, color='black', zorder=0)
    plt.axvspan(-.5, .5, alpha=0.2, color='black', zorder=0)
    ax.grid()
    fig.set_size_inches(6, 6)
    plt.xlabel(xlabel, size=15)
    plt.ylabel(ylabel, size=15)
    plt.title(title)
    plt.tight_layout()
    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
    else:
        plt.show()

def NormSD(serie):
    return serie / serie.std('time')
################################################################################

# indices
sam = xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']

dmi = DMI(filter_bwa=False, start_per='1920', end_per='2020')[2]
n34 = Nino34CPC( xr.open_dataset("/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc"),
                 start=1920, end=2020)[0]

sam = sam.rolling(time=3, center=True).mean()
sam = NormSD(sam)
dmi = NormSD(SameDateAs(dmi, sam))
n34 = NormSD(SameDateAs(n34, sam))


#------------------------------------------------------------------------------#
# by seasons
for sam_component in ['sam', 'asam', 'ssam']:
    sam = xr.open_dataset(sam_dir + sam_component +'_700.nc')['mean_estimate']
    sam = sam.rolling(time=3, center=True).mean()
    sam = NormSD(sam)



    for s, mm in zip(seasons, mmonth):
        aux_dmi = dmi.sel(time=dmi.time.dt.month.isin([mm]))
        aux_n34 = n34.sel(time=n34.time.dt.month.isin([mm]))
        aux_sam = sam.sel(time=sam.time.dt.month.isin([mm]))

        # DMI vs N34
        aux_r = np.round(pearsonr(aux_dmi, aux_n34), 3)
        ScatterPlot(aux_dmi, aux_n34, 'DMI', 'N34',
                    'DMI vs N34 - ' + s + ' - r = ' + str(aux_r[0]) +
                    ' pvalue =' + str(aux_r[1]), 'DMI_N34_' + s, dpi, save)

        # SAM vs N34
        aux_r = np.round(pearsonr(aux_sam, aux_n34), 3)
        ScatterPlot(aux_sam, aux_n34, sam_component, 'N34',
                    sam_component + ' vs N34 - ' + s + ' - r = ' + str(aux_r[0]) +
                    ' pvalue =' + str(aux_r[1]),  sam_component + '_N34_' + s, dpi, save)

        # SAM vs DMI
        aux_r = np.round(pearsonr(aux_sam, aux_dmi), 3)
        ScatterPlot(aux_sam, aux_dmi, sam_component, 'DMI',
                    sam_component + ' vs DMI - ' + s + ' - r = ' + str(aux_r[0]) +
                    ' pvalue = ' + str(aux_r[1]), sam_component + '_DMI_' + s, dpi, save)

# # Full
# # DMI vs N34
# ScatterPlot(dmi, n34, 'DMI', 'N34', 'DMI vs N34', 'DMI_N34_', dpi, save)
#
# # SAM vs N34
# ScatterPlot(sam, n34, 'SAM', 'N34', 'SAM vs N34', 'SAM_N34_', dpi, save)
#
# # SAM vs DMI
# ScatterPlot(sam, dmi, 'SAM', 'DMI', 'SAM vs DMI', 'SAM_DMI_', dpi, save)
################################################################################
