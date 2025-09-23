########################################################################################################################
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
########################################################################################################################
index_dir = '/datos/luciano.andrian/ncfiles/NMME_CFSv2/DMI_N34_Leads_r/'
cases_dir = "/pikachu/datos/luciano.andrian/cases_fields/"
out_dir = '/home/luciano.andrian/doc/salidas/ENSO_IOD/Modelos/Scatter/'

save = False
dpi = 100
########################################################################################################################

def SelectParIndex(dmi_case, n34_case, sd_dmi_s, sd_n34_s, s, by_r=True, open_n34=True, open_dmi=False):
    aux = xr.open_dataset(cases_dir + 'dmi_values_' + dmi_case + '_' + s + '.nc').__mul__(1 / sd_dmi_s)
    aux2 = xr.open_dataset(cases_dir + 'N34_values_' + n34_case + '_' + s + '.nc').__mul__(1 / sd_n34_s)

    if by_r:
        if open_n34:
            n34_s = xr.open_dataset(index_dir + 'N34_' + s + '_Leads_r_CFSv2.nc').__mul__(1/sd_n34_s)
            aux3 = n34_s.sel(r=aux.r, time=aux.time)

            if len(np.where(aux3.L.values==aux.L.values)[0]):
                #return aux.sst.values, aux3.sst.values
                return aux.sst.values.round(2), aux3.sst.values.round(2)
            else:
                print('Error: CASES')
                return [], []

        if open_dmi:
            dmi_s = xr.open_dataset(index_dir + 'DMI_' + s + '_Leads_r_CFSv2.nc').__mul__(1/sd_dmi_s)
            aux3 = dmi_s.sel(r=aux2.r, time=aux2.time)

            if len(np.where(aux3.L.values == aux2.L.values)[0]):
                return aux3.sst.values.round(2), aux2.sst.values.round(2)
            else:
                print('Error: CASES')
                return [], []
    else:
        aux2 = aux2.sel(time=aux2.time.isin([aux.time.values]))

        if len(aux2.time) == len(aux.time):
            return aux.sst.values.round(2), aux2.sst.values.round(2)
        else:
            print('Error: CASES')
            return [], []
########################################################################################################################
seasons = ['JJA', 'JAS', 'ASO', 'SON']

for s in seasons:
    sd_dmi_s = xr.open_dataset(index_dir + 'DMI_' + s + '_Leads_r_CFSv2.nc').std()
    sd_n34_s = xr.open_dataset(index_dir + 'N34_' + s + '_Leads_r_CFSv2.nc').std()

    #Sim_pos
    c = 'sim_pos'
    dmi_sim_pos, n34_sim_pos = SelectParIndex(c, c, sd_dmi_s, sd_n34_s, s)
    #sim_neg
    c = 'sim_neg'
    dmi_sim_neg, n34_sim_neg = SelectParIndex(c, c, sd_dmi_s, sd_n34_s, s)
    #dmi_puros_pos
    dmi_puros_pos, n34_in_dmi_puros_pos = SelectParIndex('dmi_puros_pos', 'neutros',
                                                         sd_dmi_s, sd_n34_s, s,
                                                         by_r=True, open_dmi=False, open_n34=True)
    #dmi_puros_neg
    dmi_puros_neg, n34_in_dmi_puros_neg = SelectParIndex('dmi_puros_neg', 'neutros',
                                                         sd_dmi_s, sd_n34_s, s,
                                                         by_r=True, open_dmi=False, open_n34=True)
    #n34_puros_pos
    dmi_in_n34_puros_pos, n34_puros_pos = SelectParIndex('neutros', 'n34_puros_pos',
                                                         sd_dmi_s, sd_n34_s, s,
                                                         by_r=True, open_dmi=True, open_n34=False)
    #n34_puros_neg
    dmi_in_n34_puros_neg, n34_puros_neg = SelectParIndex('neutros', 'n34_puros_neg',
                                                         sd_dmi_s, sd_n34_s, s,
                                                         by_r=True, open_dmi=True, open_n34=False)
    #dmi_pos_n34_neg
    dmi_pos_n34_neg, n34_in_dmi_pos_n34_neg = SelectParIndex('dmi_pos_n34_neg', 'dmi_pos_n34_neg',
                                                             sd_dmi_s, sd_n34_s, s,
                                                             by_r=False, open_dmi=False, open_n34=False)
    #dmi_neg_n34_pos
    dmi_neg_n34_pos, n34_in_dmi_neg_n34_pos = SelectParIndex('dmi_neg_n34_pos', 'dmi_neg_n34_pos',
                                                             sd_dmi_s, sd_n34_s, s,
                                                             by_r=False, open_dmi=False, open_n34=False)
    #neutros
    dmi_neutro, n34_neutro = SelectParIndex('neutros', 'neutros',
                                            sd_dmi_s, sd_n34_s, s,
                                            by_r=False)
    #------------------------------------------------------------------------------------------------------------------#
    # Plot ------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------------------------------------------------------#
    fig, ax = plt.subplots(dpi=dpi, figsize=(5, 5))
    #Simultaneos positivos --------------------------------------------------------------------------------------------#
    plt.scatter(x=dmi_sim_pos, y=n34_sim_pos,
                marker='o', s=20, edgecolor='k', color='red', alpha=.5)
    #Simultaneos negativos --------------------------------------------------------------------------------------------#
    plt.scatter(x=dmi_sim_neg, y=n34_sim_neg,
                marker='o', s=20, edgecolor='k', color='lightseagreen', alpha=.5)
    #dmi puros positivos ----------------------------------------------------------------------------------------------#
    plt.scatter(x=dmi_puros_pos, y=n34_in_dmi_puros_pos, marker='>',
                s=20, edgecolor='k', color='firebrick', alpha=.5)
    #dmi puros negativos ----------------------------------------------------------------------------------------------#
    plt.scatter(x=dmi_puros_neg, y=n34_in_dmi_puros_neg, marker='<',
                s=20, edgecolor='k', color='lime', alpha=.5)
    #n34 puros positivos ----------------------------------------------------------------------------------------------#
    plt.scatter(x=dmi_in_n34_puros_pos, y=n34_puros_pos, marker='^',
                s=20, edgecolor='k', color='darkorange', alpha=.5)
    #n34 puros negativos ----------------------------------------------------------------------------------------------#
    plt.scatter(x=dmi_in_n34_puros_neg, y=n34_puros_neg, marker='v',
                s=20, edgecolor='k', color='blue', alpha=.5)
    #dmi positivos n34 negativos --------------------------------------------------------------------------------------#
    plt.scatter(x=dmi_pos_n34_neg, y=n34_in_dmi_pos_n34_neg, marker='o',
                s=20, edgecolor='k', color='purple', alpha=.5)
    #dmi negativos n34 positivos --------------------------------------------------------------------------------------#
    plt.scatter(x=dmi_neg_n34_pos, y=n34_in_dmi_neg_n34_pos, marker='o',
                s=20, edgecolor='k', color='orange', alpha=.5)
    #Neutro -----------------------------------------------------------------------------------------------------------#
    plt.scatter(y=n34_neutro, x=dmi_neutro, marker='o',
                s=20, edgecolor='k', color='gray', alpha=.5)

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
########################################################################################################################