"""
Regresion similar a ENSO_IOD pero con SAM.
1. SAM vs N34
2. SAM vs DMI
3. DMI vs N34 (identico a ENSO_IOD)
----- / ------
Tiene sentido esto?:
4. SAM|dmi (2) vs N34|dmi (3)
5. SAM|n34 (1) vs DMI|n34 (3)
6. DMI|sam (2) vs N34|sam (1)
"""
################################################################################
sam_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'
era5_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/regression/'
################################################################################
from itertools import groupby
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
import xarray as xr
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from matplotlib import colors
import matplotlib.pyplot as plt

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from ENSO_IOD_Funciones import DMI, Nino34CPC, ComputeWithEffect, \
    ComputeWithoutEffect, PlotReg, SameDateAs
################################################################################
save = True
sig = True
sa = True # Solo para T y PP
periodos = [[1940,2020]]
p = periodos[0]
#------------------------------------------------------------------------------#
if save:
    dpi = 300
else:
    dpi = 100
full_season = False
text = False
################################################################################
def MakerMaskSig(data, r_crit):
    mask_sig = data.where((data < -1 * r_crit) | (data > r_crit))
    mask_sig = mask_sig.where(np.isnan(mask_sig), 1)

    return mask_sig

def RegressionPlots(dataindex1, corrindex1, nameindex1,
                   dataindex2, corrindex2, nameindex2,
                   prename_fig, v, v_count, s, p, dpi, save,
                    without=False, out_dir = out_dir,
                    t_pp=False, sa=False):

    #--------------------------------------------------------------------------#
    cbar = colors.ListedColormap(['#9B1C00', '#B9391B', '#CD4838', '#E25E55',
                                  '#F28C89', '#FFCECC',
                                  'white',
                                  '#B3DBFF', '#83B9EB', '#5E9AD7', '#3C7DC3',
                                  '#2064AF', '#014A9B'][::-1])
    cbar.set_over('#641B00')
    cbar.set_under('#012A52')
    cbar.set_bad(color='white')

    cbar_pp = colors.ListedColormap(['#003C30', '#004C42', '#0C7169',
                                     '#79C8BC', '#B4E2DB',
                                     'white',
                                     '#F1DFB3', '#DCBC75', '#995D13',
                                     '#6A3D07', '#543005'][::-1])
    cbar_pp.set_under('#3F2404')
    cbar_pp.set_over('#00221A')
    cbar_pp.set_bad(color='white')
    #--------------------------------------------------------------------------#
    y1 = 0
    t_critic = 1.66
    r_crit = np.sqrt(
        1 / (((np.sqrt((p[1] - p[0] + y1) - 2) / t_critic) ** 2) + 1))
    # -------------------------------------------------------------------------#
    if t_pp:
        scales = [np.linspace(-15, 15, 13),  # pp
                  [-.6, -.4, -.2, -.1, -.05, 0, 0.05, 0.1, 0.2, 0.4, 0.6]]  # t
        cmap = [cbar_pp, cbar]
        title_var = ['PP GPCC', 'T Cru']
        two_variables = False
        sig = True
        sig_point = True
        r_crit2 = 0
    else:
        scales = [[-150, -100, -75, -50, -25, -15, 0, 15, 25, 50, 75, 100, 150],
                  [-150, -100, -75, -50, -25, -15, 0, 15, 25, 50, 75, 100, 150]]
        title_var = ['HGT200 ERA5', 'HGT750 ERA5']
        cmap = [cbar, cbar]
        two_variables = True
        sig = False
        sig_point = False
        r_crit2 = r_crit
        sa = False

    # Reg index1 --------------------------------------------------------------#
    if without:
        title = title_var[v_count] + '_' + s + '_' + str(p[0] + y1) + \
                '_' + str(p[1]) + '_' + nameindex1 + ' -{' + nameindex2 + '}'
        name_fig = v + '_' + s + '_' + str(p[0] + y1) + '_' + str(p[1]) + \
                   prename_fig + '_' + nameindex1 + '_WO_' + nameindex2
    else:
        title = title_var[v_count] + '_' + s + '_' + str(p[0] + y1) + \
                '_' + str(p[1]) + '_' + nameindex1 + '_' + prename_fig
        name_fig = v + '_' + s + '_' + str(p[0] + y1) + '_' + str(p[1]) + \
                   prename_fig + '_' + nameindex1

    PlotReg(data=dataindex1 * MakerMaskSig(corrindex1, r_crit2),
            data_cor=corrindex1, SA=sa,
            levels=scales[v_count],
            two_variables=two_variables, data2=dataindex1, sig2=False,
            levels2=scales[v_count], title=title, name_fig=name_fig,
            out_dir=out_dir, save=save, cmap=cmap[v_count], dpi=dpi,
            color_map='grey', sig=sig, sig_point=sig_point, color_sig='k',
            r_crit=r_crit)

    # Reg index2 --------------------------------------------------------------#
    if without:
        title = title_var[v_count] + '_' + s + '_' + str(p[0] + y1) + \
                '_' + str(p[1]) + '_' + nameindex2 + ' -{' + nameindex1 + '}'
        name_fig = v + '_' + s + '_' + str(p[0] + y1) + '_' + str(p[1]) + \
                   prename_fig + '_' + nameindex2 + '_WO_' + nameindex1
    else:
        title = title_var[v_count] + '_' + s + '_' + str(p[0] + y1) + \
                '_' + str(p[1]) + '_' + nameindex2 + '_' + prename_fig
        name_fig = v + '_' + s + '_' + str(p[0] + y1) + '_' + str(p[1]) + \
                   prename_fig + '_' + nameindex2

    PlotReg(data=dataindex2 * MakerMaskSig(corrindex2, r_crit2),
            data_cor=corrindex2, SA=sa,
            levels=scales[v_count],
            two_variables=two_variables, data2=dataindex2, sig2=False,
            levels2=scales[v_count], title=title, name_fig=name_fig,
            out_dir=out_dir, save=save, cmap=cmap[v_count], dpi=dpi,
            color_map='grey', sig=sig, sig_point=sig_point, color_sig='k',
            r_crit=r_crit)
    #--------------------------------------------------------------------------#
    #--------------------------------------------------------------------------#


def PreProcIndex(dmi_or, n34_or, sam_or, mm):
    dmi = SameDateAs(dmi_or, sam_or)
    dmi = dmi.sel(time=dmi.time.dt.month.isin(mm))

    n34 = SameDateAs(n34_or, sam_or)
    n34 = n34.sel(time=n34.time.dt.month.isin(mm))

    sam = SameDateAs(sam_or, sam_or) # Esto no hace nada
    sam = sam.sel(time=sam.time.dt.month.isin(mm))

    return dmi, n34, sam

def OpenObsDataSet(name, sa=True,
                   dir='/pikachu/datos/luciano.andrian/observado/ncfiles'
                        '/data_obs_d_w_c/'):

    aux = xr.open_dataset(dir + name + '.nc')
    if sa:
        aux2 = aux.sel(lon=slice(270, 330), lat=slice(15, -60))
        if len(aux2.lat) > 0:
            return aux2
        else:
            aux2 = aux.sel(lon=slice(270, 330), lat=slice(-60, 15))
            return aux2
    else:
        return aux
################################################################################
seasons = [10] # main month
seasons_name = ['SON']
interp = False
two_variables=False

################################################################################
################################################################################
# indices
dmi_or = DMI(filter_bwa=False, start_per='1920', end_per='2020')[2]

n34_or = Nino34CPC(
    xr.open_dataset("/pikachu/datos4/Obs/sst/sst.mnmean_2020.nc"),
    start=1920, end=2020)[0]

sam_or = -1 * xr.open_dataset(sam_dir + 'sam_700.nc')['mean_estimate']
sam_or = sam_or.rolling(time=3, center=True).mean()
################################################################################
print('<<<<<<<<<<<<<<<<<<<<<<<<<<<< Regression >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
################################################################################
#-------------------------------- z200, z750 ----------------------------------#
variables = ['HGT200_', 'HGT750_']
title_var = ['HGT200 ERA5', 'HGT750 ERA5']

# for test:
v = 'HGT200_'
v_count = 0
hpalevel = 200
s = 'SON'
mm = 10

for v, v_count, hpalevel in zip(variables, [0,1], [200,750]):
    print('Variable: ' + v)
    print('Level: ' + str(hpalevel))

    # by seasons
    for s, mm in zip(seasons_name, seasons):
        print(s)
        # data ----------------------------------------------------------------#
        data = xr.open_dataset(era5_dir + v + s + '_mer_d_w.nc')
        time_original = data.time  # ya esta desde 1940-2020

        # periodo y estacion de los indices -----------------------------------#
        dmi, n34, sam = PreProcIndex(dmi_or, n34_or, sam_or, mm)

        #----------------------------------------------------------------------#
        # AVISO: las funciones fueron pensadas para usar dmi y n34
        # por eso los argumentos de los indices se llaman "n34" y "dmi"
        # pero funciona igual con cualquier serie temporal que ingrese
        #
        # ---------------------------------------------------------------------#
        # En cada paso todas las variables podrian sobreescribirse menos los
        # indices n34_wo_sam, sam_wo_n34, etc
        # ---------------------------------------------------------------------#

        print('1. SAM vs N34 #################################################')
        reg1_n34, reg1_corr_n34, reg1_sam, reg1_corr_sam = \
            ComputeWithEffect(data=data, n34=n34, dmi=sam,
                              m=mm, time_original=time_original)

        RegressionPlots(reg1_n34, reg1_corr_n34, 'N34',
                       reg1_sam, reg1_corr_sam, 'SAM',
                       'reg1', v, v_count, s, p, dpi, save)


        reg1_n34_wosam, reg1_corr_n34, reg1_sam_won34, reg1_corr_sam, \
        n34_wo_sam, sam_wo_n34 = ComputeWithoutEffect(
            data, n34=n34, dmi=sam, m=mm, time_original=time_original)

        RegressionPlots(reg1_n34_wosam, reg1_corr_n34, 'N34',
                       reg1_sam_won34, reg1_corr_sam, 'SAM',
                       'reg1', v, v_count, s, p, dpi, save, without=True)

        print('2. SAM vs DMI #################################################')
        reg2_dmi, reg2_corr_dmi, reg2_sam, reg2_corr_sam = \
            ComputeWithEffect(data=data, n34=dmi, dmi=sam,
                              m=mm, time_original=time_original)
        RegressionPlots(reg2_dmi, reg2_corr_dmi, 'DMI',
                       reg2_sam, reg2_corr_sam, 'SAM',
                       'reg2', v, v_count, s, p, dpi, save)


        reg2_dmi_wosam, reg2_corr_dmi, reg2_sam_wodmi, reg2_corr_sam, \
        dmi_wo_sam, sam_wo_dmi= ComputeWithoutEffect(
            data, n34=dmi, dmi=sam, m=mm, time_original=time_original)

        RegressionPlots(reg2_dmi_wosam, reg2_corr_dmi, 'DMI',
                       reg2_sam_wodmi, reg2_corr_sam, 'SAM',
                       'reg2', v, v_count, s, p, dpi, save, without=True)

        print('3. N34 vs DMI #################################################')
        reg3_n34, reg3_corr_n34, reg3_dmi, reg3_corr_dmi = \
            ComputeWithEffect(data=data, n34=n34, dmi=dmi,
                              m=mm, time_original=time_original)

        RegressionPlots(reg3_n34, reg3_corr_n34, 'N34',
                       reg3_dmi, reg3_corr_dmi, 'DMI',
                       'reg3', v, v_count, s, p, dpi, save)


        reg3_n34_wodmi, reg3_corr_n34, reg3_dmi_won34, reg3_corr_dmi, \
        n34_wo_dmi, dmi_wo_n34 = ComputeWithoutEffect(
            data, n34=n34, dmi=dmi, m=mm, time_original=time_original)

        RegressionPlots(reg3_n34_wodmi, reg3_corr_n34, 'N34',
                       reg3_dmi_won34, reg3_corr_dmi, 'DMI',
                       'reg2', v, v_count, s, p, dpi, save, without=True)

        ########################################################################
        # partial regression

        print('4. SAM|dmi vs N34|dmi #########################################')
        reg4_n34_wosam_wodmi, reg1_corr_n34, \
        reg4_sam_won34_wodmi, reg1_corr_sam, aux1, aux2 = \
            ComputeWithoutEffect(data, n34=n34_wo_dmi, dmi=sam_wo_dmi,
                                 m=mm, time_original=time_original)

        RegressionPlots(reg4_n34_wosam_wodmi, reg1_corr_n34, 'N34_woDMI',
                       reg4_sam_won34_wodmi, reg1_corr_sam, 'SAM_woDMI',
                       'reg4', v, v_count, s, p, dpi, save, without=True)

        print('5. SAM|n34 vs DMI|n34 #########################################')
        reg5_sam_wodmi_won34, reg2_corr_sam, \
        reg5_dmi_wosam_won34, reg2_corr_dmi, aux1, aux2 = \
            ComputeWithoutEffect(data, n34=sam_wo_n34, dmi=dmi_wo_n34,
                                 m=mm, time_original=time_original)

        RegressionPlots(reg5_sam_wodmi_won34, reg2_corr_sam, 'SAM_woN34',
                       reg5_dmi_wosam_won34, reg2_corr_dmi, 'DMI_woN34',
                       'reg5', v, v_count, s, p, dpi, save, without=True)

        print('6. N34|sam vs DMI|sam #########################################')
        reg6_n34_wodmi_wosam, reg3_corr_n34, \
        reg6_dmi_won34_wosam, reg3_corr_dmi, aux1, aux2 = \
            ComputeWithoutEffect(data, n34=n34_wo_sam, dmi=dmi_wo_sam,
                                 m=mm, time_original=time_original)

        RegressionPlots(reg6_n34_wodmi_wosam, reg3_corr_n34, 'N34_woSAM',
                       reg6_dmi_won34_wosam, reg3_corr_dmi, 'DMI_woSAM',
                       'reg6', v, v_count, s, p, dpi, save, without=True)
################################################################################
#---------------------------------- PP y T-------------------------------------#
variables_tpp = ['ppgpcc_w_c_d_1', 'tcru_w_c_d_0.25']
plt.rcParams['hatch.linewidth'] = 2

# for test
v = 'tcru_w_c_d_0.25'
v_count = 1
s = 'SON'
mm = 10
for v_count, v in enumerate(variables_tpp):
    print('Variable: ' + v)

    # by seasons
    for s, mm in zip(seasons_name, seasons):
        print(s)
        # data ----------------------------------------------------------------#
        data = OpenObsDataSet(name=v + '_' + s, sa=sa)
        time_original = data.time # ya esta desde 1940-2020

        # periodo y estacion de los indices -----------------------------------#
        dmi, n34, sam = PreProcIndex(dmi_or, n34_or, sam_or, mm)

        #----------------------------------------------------------------------#
        # AVISO: las funciones fueron pensadas para usar dmi y n34
        # por eso los argumentos de los indices se llaman "n34" y "dmi"
        # pero funciona igual con cualquier serie temporal que ingrese
        #
        # ---------------------------------------------------------------------#
        # En cada paso todas las variables podrian sobreescribirse menos los
        # indices n34_wo_sam, sam_wo_n34, etc
        # ---------------------------------------------------------------------#

        print('1. SAM vs N34 #################################################')
        reg1_n34, reg1_corr_n34, reg1_sam, reg1_corr_sam = \
            ComputeWithEffect(data=data, n34=n34, dmi=sam,
                              m=mm, time_original=time_original)

        RegressionPlots(reg1_n34, reg1_corr_n34, 'N34',
                       reg1_sam, reg1_corr_sam, 'SAM',
                       'reg1', v, v_count, s, p, dpi, save,
                        t_pp=True, sa=sa)


        reg1_n34_wosam, reg1_corr_n34, reg1_sam_won34, reg1_corr_sam, \
        n34_wo_sam, sam_wo_n34 = ComputeWithoutEffect(
            data, n34=n34, dmi=sam, m=mm, time_original=time_original)

        RegressionPlots(reg1_n34_wosam, reg1_corr_n34, 'N34',
                       reg1_sam_won34, reg1_corr_sam, 'SAM',
                       'reg1', v, v_count, s, p, dpi, save, without=True,
                        t_pp=True, sa=sa)

        print('2. SAM vs DMI #################################################')
        reg2_dmi, reg2_corr_dmi, reg2_sam, reg2_corr_sam = \
            ComputeWithEffect(data=data, n34=dmi, dmi=sam,
                              m=mm, time_original=time_original)
        RegressionPlots(reg2_dmi, reg2_corr_dmi, 'DMI',
                       reg2_sam, reg2_corr_sam, 'SAM',
                       'reg2', v, v_count, s, p, dpi, save,
                        t_pp=True, sa=sa)


        reg2_dmi_wosam, reg2_corr_dmi, reg2_sam_wodmi, reg2_corr_sam, \
        dmi_wo_sam, sam_wo_dmi= ComputeWithoutEffect(
            data, n34=dmi, dmi=sam, m=mm, time_original=time_original)

        RegressionPlots(reg2_dmi_wosam, reg2_corr_dmi, 'DMI',
                       reg2_sam_wodmi, reg2_corr_sam, 'SAM',
                       'reg2', v, v_count, s, p, dpi, save, without=True,
                        t_pp=True, sa=sa)

        print('3. N34 vs DMI #################################################')
        reg3_n34, reg3_corr_n34, reg3_dmi, reg3_corr_dmi = \
            ComputeWithEffect(data=data, n34=n34, dmi=dmi,
                              m=mm, time_original=time_original)

        RegressionPlots(reg3_n34, reg3_corr_n34, 'N34',
                       reg3_dmi, reg3_corr_dmi, 'DMI',
                       'reg3', v, v_count, s, p, dpi, save,
                        t_pp=True, sa=sa)


        reg3_n34_wodmi, reg3_corr_n34, reg3_dmi_won34, reg3_corr_dmi, \
        n34_wo_dmi, dmi_wo_n34 = ComputeWithoutEffect(
            data, n34=n34, dmi=dmi, m=mm, time_original=time_original)

        RegressionPlots(reg3_n34_wodmi, reg3_corr_n34, 'N34',
                       reg3_dmi_won34, reg3_corr_dmi, 'DMI',
                       'reg2', v, v_count, s, p, dpi, save, without=True,
                        t_pp=True, sa=sa)

        ########################################################################
        # partial regression

        print('4. SAM|dmi vs N34|dmi #########################################')
        reg4_n34_wosam_wodmi, reg1_corr_n34, \
        reg4_sam_won34_wodmi, reg1_corr_sam, aux1, aux2 = \
            ComputeWithoutEffect(data, n34=n34_wo_dmi, dmi=sam_wo_dmi,
                                 m=mm, time_original=time_original)

        RegressionPlots(reg4_n34_wosam_wodmi, reg1_corr_n34, 'N34_woDMI',
                       reg4_sam_won34_wodmi, reg1_corr_sam, 'SAM_woDMI',
                       'reg4', v, v_count, s, p, dpi, save, without=True,
                        t_pp=True, sa=sa)

        print('5. SAM|n34 vs DMI|n34 #########################################')
        reg5_sam_wodmi_won34, reg2_corr_sam, \
        reg5_dmi_wosam_won34, reg2_corr_dmi, aux1, aux2 = \
            ComputeWithoutEffect(data, n34=sam_wo_n34, dmi=dmi_wo_n34,
                                 m=mm, time_original=time_original)

        RegressionPlots(reg5_sam_wodmi_won34, reg2_corr_sam, 'SAM_woN34',
                       reg5_dmi_wosam_won34, reg2_corr_dmi, 'DMI_woN34',
                       'reg5', v, v_count, s, p, dpi, save, without=True,
                        t_pp=True, sa=sa)

        print('6. N34|sam vs DMI|sam #########################################')
        reg6_n34_wodmi_wosam, reg3_corr_n34, \
        reg6_dmi_won34_wosam, reg3_corr_dmi, aux1, aux2 = \
            ComputeWithoutEffect(data, n34=n34_wo_sam, dmi=dmi_wo_sam,
                                 m=mm, time_original=time_original)

        RegressionPlots(reg6_n34_wodmi_wosam, reg3_corr_n34, 'N34_woSAM',
                       reg6_dmi_won34_wosam, reg3_corr_dmi, 'DMI_woSAM',
                       'reg6', v, v_count, s, p, dpi, save, without=True,
                        t_pp=True, sa=sa)
################################################################################
print('done')
################################################################################
