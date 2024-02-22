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
        x_sd_aux_path + x_name.upper() + '_' + s + '_Leads_r_CFSv2.nc').std()
    sd_y_s = xr.open_dataset(
        y_sd_aux_path + y_name.upper() +'_' + s + '_Leads_r_CFSv2.nc').std()

    aux_x = xr.open_dataset(
        cases_dir + x_name + '_values_in_' + x + '_' + s + '.nc')\
        .__mul__(1 / sd_x_s)

    aux_y = xr.open_dataset(
        cases_dir + y_name + '_values_in_' + y + '_' + s + '.nc') \
        .__mul__(1 / sd_y_s)

    return aux_x[x_xr_name].values.round(2), \
               aux_y[y_xr_name].values.round(2)





########################################################################################################################
seasons = ['JJA', 'JAS', 'ASO', 'SON']
seasons = ['SON']

cases = ['sim_pos', 'sim_neg', #triple sim
         'dmi_puros_pos', 'dmi_puros_neg', # puros, sin los otros dos
         'n34_puros_pos', 'n34_puros_neg', # puros, sin los otros dos
         'sam_puros_pos', 'sam_puros_neg', # puros, sin los otros dos
         'dmi_pos_n34_pos_sam_neg', # simultaneo signos opuestos
         'dmi_pos_n34_neg_sam_pos', # simultaneo signos opuestos
         'dmi_pos_n34_neg_sam_neg', # simultaneo signos opuestos
         'dmi_neg_n34_pos_sam_neg', # simultaneo signos opuestos
         'dmi_neg_n34_pos_sam_pos', # simultaneo signos opuestos
         'dmi_neg_n34_neg_sam_pos', # simultaneo signos opuestos
         'dmi_pos', 'dmi_neg', # todos los de uno, sin importar el resto
         'n34_pos', 'n34_neg', # todos los de uno, sin importar el resto
         'sam_pos', 'sam_neg', # todos los de uno, sin importar el resto
         'neutros', # todos neutros
         'dmi_sim_n34_pos_wo_sam', # dos eventos simultaneos sin el otro
         'dmi_sim_sam_pos_wo_n34', # dos eventos simultaneos sin el otro
         'n34_sim_sam_pos_wo_dmi', # dos eventos simultaneos sin el otro
         'n34_sim_dmi_pos_wo_sam', # dos eventos simultaneos sin el otro
         'sam_sim_n34_pos_wo_dmi', # dos eventos simultaneos sin el otro
         'sam_sim_dmi_pos_wo_n34', # dos eventos simultaneos sin el otro
         'dmi_sim_n34_neg_wo_sam', # dos eventos simultaneos sin el otro
         'n34_sim_sam_neg_wo_dmi', # dos eventos simultaneos sin el otro
         'n34_sim_dmi_neg_wo_sam', # dos eventos simultaneos sin el otro
         'sam_sim_n34_neg_wo_dmi', # dos eventos simultaneos sin el otro
         'sam_sim_dmi_neg_wo_n34',
         'dmi_sim_sam_neg_wo_n34'] # dos eventos simultaneos sin el otro

first_index = 'dmi'
second_index = 'sam'
excluded_index = 'n34'
sign = 'pos'

sam_puros = xr.open_dataset(cases_dir + 'sam_values_in_sam_puros_pos_SON.nc')
dmi_for_sam_puros = xr.open_dataset(cases_dir + 'dmi_values_in_sam_puros_pos_SON.nc')
for s in seasons:
    #Sim_pos
    x = f"{first_index}_sim_{second_index}_pos_wo_{excluded_index}"
    y = f"{second_index}_sim_{first_index}_pos_wo_{excluded_index}"
    fi_sim_pos, si_sim_pos = SelectParIndex(x, y, s)

    #sim_neg
    x = f"{first_index}_sim_{second_index}_neg_wo_{excluded_index}"
    y = f"{second_index}_sim_{first_index}_neg_wo_{excluded_index}"
    fi_sim_neg, si_sim_neg = SelectParIndex(x, y, s)

    # fi_puros_pos
    x = f"{first_index}_puros_pos"
    y = f"{second_index}_{x}"
    fi_puros_pos, si_in_fi_puros_pos = SelectParIndex(x, y, s)

    #fi_puros_neg
    x = f"{first_index}_puros_neg"
    y = f"{second_index}_{x}"
    fi_puros_neg, si_in_fi_puros_neg = SelectParIndex(x, y, s)

    #si_puros_pos
    x = f"{second_index}_puros_pos"
    y = f"{first_index}_{x}"
    si_puros_pos, fi_in_si_puros_pos = SelectParIndex(x, y, s)

    #si_puros_neg
    x = f"{second_index}_puros_neg"
    y = f"{first_index}_{x}"
    si_puros_neg, fi_in_si_puros_neg = SelectParIndex(x, y, s)

    # fi_pos_si_neg # Corrgeir los dobles 'WO' en SelectEvents
    x = f"{first_index}_pos_{second_index}_neg_wo_{excluded_index}"
    y = f"{second_index}_neg_{first_index}_pos_wo_{excluded_index}"
    fi_sim_pos, si_sim_pos = SelectParIndex(x, y, s)

    # #n34_puros_neg
    # dmi_in_n34_puros_neg, n34_puros_neg = SelectParIndex('neutros', 'n34_puros_neg',
    #                                                      sd_dmi_s, sd_n34_s, s,
    #                                                      by_r=True, open_dmi=True, open_n34=False)
    # #dmi_pos_n34_neg
    # dmi_pos_n34_neg, n34_in_dmi_pos_n34_neg = SelectParIndex('dmi_pos_n34_neg', 'dmi_pos_n34_neg',
    #                                                          sd_dmi_s, sd_n34_s, s,
    #                                                          by_r=False, open_dmi=False, open_n34=False)
    # #dmi_neg_n34_pos
    # dmi_neg_n34_pos, n34_in_dmi_neg_n34_pos = SelectParIndex('dmi_neg_n34_pos', 'dmi_neg_n34_pos',
    #                                                          sd_dmi_s, sd_n34_s, s,
    #                                                          by_r=False, open_dmi=False, open_n34=False)
    # #neutros
    # dmi_neutro, n34_neutro = SelectParIndex('neutros', 'neutros',
    #                                         sd_dmi_s, sd_n34_s, s,
    #                                         by_r=False)
    #------------------------------------------------------------------------------------------------------------------#
    # Plot ------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------------------------------------------------------#
    fig, ax = plt.subplots(dpi=dpi, figsize=(5, 5))
    # Simultaneos positivos --------------------------------------------------------------------------------------------#
    plt.scatter(x=fi_sim_pos, y=si_sim_pos,
                marker='o', s=20, edgecolor='k', color='red', alpha=.5)
    # Simultaneos negativos --------------------------------------------------------------------------------------------#
    plt.scatter(x=fi_sim_neg, y=si_sim_neg,
                marker='o', s=20, edgecolor='k', color='lightseagreen', alpha=.5)
    # fi puros positivos ----------------------------------------------------------------------------------------------#
    plt.scatter(x=fi_puros_pos, y=si_in_fi_puros_pos, marker='>',
                s=20, edgecolor='k', color='firebrick', alpha=.5)
    # fi puros negativos ----------------------------------------------------------------------------------------------#
    plt.scatter(x=fi_puros_neg, y=si_in_fi_puros_neg, marker='<',
                s=20, edgecolor='k', color='lime', alpha=.5)
    # si puros positivos ----------------------------------------------------------------------------------------------#
    plt.scatter(x=fi_in_si_puros_pos, y=si_puros_pos, marker='^',
                s=20, edgecolor='k', color='darkorange', alpha=.5)
    # si puros negativos ----------------------------------------------------------------------------------------------#
    plt.scatter(x=si_in_fi_puros_neg, y=fi_puros_neg, marker='v',
                 s=20, edgecolor='k', color='blue', alpha=.5)
    # #dmi positivos n34 negativos --------------------------------------------------------------------------------------#
    # plt.scatter(x=dmi_pos_n34_neg, y=n34_in_dmi_pos_n34_neg, marker='o',
    #             s=20, edgecolor='k', color='purple', alpha=.5)
    # #dmi negativos n34 positivos --------------------------------------------------------------------------------------#
    # plt.scatter(x=dmi_neg_n34_pos, y=n34_in_dmi_neg_n34_pos, marker='o',
    #             s=20, edgecolor='k', color='orange', alpha=.5)
    # #Neutro -----------------------------------------------------------------------------------------------------------#
    # plt.scatter(y=n34_neutro, x=dmi_neutro, marker='o',
    #             s=20, edgecolor='k', color='gray', alpha=.5)

    plt.ylim((-4, 4));
    plt.xlim((-4, 4))
    plt.axhspan(-.5, .5, alpha=0.2, color='black', zorder=0)
    plt.axvspan(-.5, .5, alpha=0.2, color='black', zorder=0)
    # ax.grid(True)
    fig.set_size_inches(6, 6)
    plt.xlabel('IOD', size=14)
    plt.ylabel('Niño 3.4', size=14)

    plt.text(-3.9, 3.6, 'EN/IOD-', dict(size=14))
    plt.text(-.3, 3.6, 'EN', dict(size=14))
    plt.text(+2.5, 3.6, 'EN/IOD+', dict(size=14))
    plt.text(+3, -.1, 'IOD+', dict(size=14))
    plt.text(+2.5, -3.9, 'LN/IOD+', dict(size=14))
    plt.text(-.3, -3.9, 'LN', dict(size=14))
    plt.text(-3.9, -3.9, ' LN/IOD-', dict(size=14))
    plt.text(-3.9, -.1, 'IOD-', dict(size=14))
    plt.title('CFSv2 - ' + s, fontsize=14)
    plt.tight_layout()
    if save:
        plt.savefig(out_dir + 'scatter_CFSv2_' + s + '.jpg')
        plt.close()
    else:
        plt.show()
################################################################################################################
