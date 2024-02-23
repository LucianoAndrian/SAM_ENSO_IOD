"""
Scatter plot 2D: SAM vs N34 y SAM vs DMI
"""
# ---------------------------------------------------------------------------- #
save = False
# ---------------------------------------------------------------------------- #
dmi_n34_dir = '/datos/luciano.andrian/ncfiles/NMME_CFSv2/DMI_N34_Leads_r/'
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
cases_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cases_index/'

out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/scatter/'
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

if save:
    dpi=300
else:
    dpi=100
# ---------------------------------------------------------------------------- #
def SelectParIndex(x, y, s):

    x_name = x.split('_')[0]
    y_name = y.split('_')[0]

    if x_name == y.split('_')[1]:
        y = y.split('_', 1)[1]

    if x_name.lower() == 'sam':
        x_xr_name = 'sam'
        x_sd_aux_path = \
            '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cases_index/'
    else:
        x_xr_name = 'sst'
        x_sd_aux_path = \
            '/datos/luciano.andrian/ncfiles/NMME_CFSv2/DMI_N34_Leads_r/'

    if y_name.lower() == 'sam':
        y_xr_name = 'sam'
        y_sd_aux_path = \
            '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/cases_index/'
    else:
        y_xr_name = 'sst'
        y_sd_aux_path = \
            '/datos/luciano.andrian/ncfiles/NMME_CFSv2/DMI_N34_Leads_r/'

    sd_x_s = xr.open_dataset(
        x_sd_aux_path + x_name.upper() + '_' + s + '_Leads_r_CFSv2.nc')\
        .mean('r').std()
    sd_y_s = xr.open_dataset(
        y_sd_aux_path + y_name.upper() +'_' + s + '_Leads_r_CFSv2.nc')\
        .mean('r').std()

    try:
        aux_x = xr.open_dataset(
            cases_dir + x_name + '_values_in_' + x + '_' + s + '.nc') \
            .__mul__(1 / sd_x_s)

        aux_y = xr.open_dataset(
            cases_dir + y_name + '_values_in_' + y + '_' + s + '.nc') \
            .__mul__(1 / sd_y_s)
    except:
        aux_x = xr.open_dataset(
            cases_dir + x_name + '_values_in_neutros_' + s + '.nc') \
            .__mul__(1 / sd_x_s)

        aux_y = xr.open_dataset(
            cases_dir + y_name + '_values_in_neutros_' + s + '.nc') \
            .__mul__(1 / sd_y_s)


    return aux_x[x_xr_name].values.round(2), \
               aux_y[y_xr_name].values.round(2)

def PlotScatter(first_index, second_index, excluded_index,
                save=False, dpi=100):
    #seasons = ['JJA', 'JAS', 'ASO', 'SON']
    seasons = ['SON']
    for s in seasons:
        # Sim_pos
        x = f"{first_index}_sim_{second_index}_pos_wo_{excluded_index}"
        y = f"{second_index}_sim_{first_index}_pos_wo_{excluded_index}"
        fi_sim_pos, si_sim_pos = SelectParIndex(x, y, s)

        # sim_neg
        x = f"{first_index}_sim_{second_index}_neg_wo_{excluded_index}"
        y = f"{second_index}_sim_{first_index}_neg_wo_{excluded_index}"
        fi_sim_neg, si_sim_neg = SelectParIndex(x, y, s)

        # fi_puros_pos
        x = f"{first_index}_puros_pos"
        y = f"{second_index}_{x}"
        fi_puros_pos, si_in_fi_puros_pos = SelectParIndex(x, y, s)

        # fi_puros_neg
        x = f"{first_index}_puros_neg"
        y = f"{second_index}_{x}"
        fi_puros_neg, si_in_fi_puros_neg = SelectParIndex(x, y, s)

        # si_puros_pos
        x = f"{second_index}_puros_pos"
        y = f"{first_index}_{x}"
        si_puros_pos, fi_in_si_puros_pos = SelectParIndex(x, y, s)

        # si_puros_neg
        x = f"{second_index}_puros_neg"
        y = f"{first_index}_{x}"
        si_puros_neg, fi_in_si_puros_neg = SelectParIndex(x, y, s)

        # fi_pos_si_neg # Corrgeir los dobles 'WO' en SelectEvents
        x = f"{first_index}_pos_{second_index}_neg_wo_{excluded_index}"
        y = f"{second_index}_{x}"
        fi_pos_si_neg, si_in_fi_pos_si_neg = SelectParIndex(x, y, s)

        # fi_neg_si_pos
        x = f"{first_index}_neg_{second_index}_pos_wo_{excluded_index}"
        y = f"{second_index}_{x}"
        fi_neg_si_pos, si_in_fi_neg_si_pos = SelectParIndex(x, y, s)

        # neutros
        x = f"{first_index}_neutros"
        y = f"{second_index}_neutros"
        fi_neutro, si_neutro = SelectParIndex(x, y, s)

        # -------------------------------------------------------------------- #
        # Plot --------------------------------------------------------------- #
        # -------------------------------------------------------------------- #
        fig, ax = plt.subplots(dpi=dpi, figsize=(5, 5))
        # Simultaneos positivos ---------------------------------------------- #
        plt.scatter(x=fi_sim_pos, y=si_sim_pos,
                    marker='o', s=20, edgecolor='k', color='red', alpha=.5)
        # Simultaneos negativos ---------------------------------------------- #
        plt.scatter(x=fi_sim_neg, y=si_sim_neg,
                    marker='o', s=20, edgecolor='k', color='lightseagreen',
                    alpha=.5)
        # fi puros positivos ------------------------------------------------- #
        plt.scatter(x=fi_puros_pos, y=si_in_fi_puros_pos, marker='>',
                    s=20, edgecolor='k', color='firebrick', alpha=.5)
        # fi puros negativos ------------------------------------------------- #
        plt.scatter(x=fi_puros_neg, y=si_in_fi_puros_neg, marker='<',
                    s=20, edgecolor='k', color='lime', alpha=.5)
        # si puros positivos ------------------------------------------------- #
        plt.scatter(x=fi_in_si_puros_pos, y=si_puros_pos, marker='^',
                    s=20, edgecolor='k', color='darkorange', alpha=.5)
        # si puros negativos ------------------------------------------------- #
        plt.scatter(x=fi_in_si_puros_neg, y=si_puros_neg, marker='v',
                    s=20, edgecolor='k', color='blue', alpha=.5)
        # fi positivos si negativos ------------------------------------------ #
        plt.scatter(x=fi_pos_si_neg, y=si_in_fi_pos_si_neg, marker='o',
                    s=20, edgecolor='k', color='purple', alpha=.5)
        # fi negativos si positivos ------------------------------------------ #
        plt.scatter(x=fi_neg_si_pos, y=si_in_fi_neg_si_pos, marker='o',
                    s=20, edgecolor='k', color='orange', alpha=.5)
        # Neutro ------------------------------------------------------------- #
        plt.scatter(y=fi_neutro, x=si_neutro, marker='o',
                    s=20, edgecolor='k', color='gray', alpha=.5)

        plt.ylim((-4, 4));
        plt.xlim((-4, 4))
        plt.axhspan(-.5, .5, alpha=0.2, color='black', zorder=0)
        plt.axvspan(-.5, .5, alpha=0.2, color='black', zorder=0)
        # ax.grid(True)
        fig.set_size_inches(6, 6)
        plt.xlabel(first_index.upper(), size=14)
        plt.ylabel(second_index.upper(), size=14)

        plt.text(-3.9, 3.6, f"{second_index}+/{first_index}-", dict(size=14))
        plt.text(-.4, 3.6, f"{second_index}+", dict(size=14))
        plt.text(+2.1, 3.6, f"{second_index}+/{first_index}+", dict(size=14))
        plt.text(+3, -.1, f"{first_index}+", dict(size=14))
        plt.text(+2.1, -3.9, f"{second_index}-/{first_index}+", dict(size=14))
        plt.text(-.3, -3.9, f"{second_index}-", dict(size=14))
        plt.text(-3.9, -3.9, f"{second_index}-/{first_index}-", dict(size=14))
        plt.text(-3.9, -.1, f"{first_index}-", dict(size=14))
        plt.title('CFSv2 - ' + s, fontsize=14)
        plt.tight_layout()
        if save:
            plt.savefig(
                f"{out_dir}scatter_CFSv2_{first_index}-{second_index}_{s}.jpg")
            plt.close()
        else:
            plt.show()

################################################################################
PlotScatter(first_index='dmi', second_index='n34', excluded_index='sam',
            save=save, dpi=dpi)
PlotScatter(first_index='dmi', second_index='sam', excluded_index='n34',
            save=save, dpi=dpi)
PlotScatter(first_index='n34', second_index='sam', excluded_index='dmi',
            save=save, dpi=dpi)
################################################################################
