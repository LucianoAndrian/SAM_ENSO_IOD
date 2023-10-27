"""
Barras de colores y escalas varias
"""
################################################################################
from matplotlib import colors
import numpy as np
################################################################################
################################################################################
# Colorbars ####################################################################
def get_cbars(VarName):
    cbar_hgt = colors.ListedColormap(['#9B1C00', '#B9391B', '#CD4838',
                                      '#E25E55', '#F28C89', '#FFCECC',
                                      'white',
                                      '#B3DBFF', '#83B9EB', '#5E9AD7',
                                      '#3C7DC3', '#2064AF', '#014A9B'][::-1])
    cbar_hgt.set_over('#641B00')
    cbar_hgt.set_under('#012A52')
    cbar_hgt.set_bad(color='white')

    cbar_pp = colors.ListedColormap(['#003C30', '#004C42', '#0C7169',
                                     '#79C8BC', '#B4E2DB',
                                     'white',
                                     '#F1DFB3', '#DCBC75', '#995D13',
                                     '#6A3D07', '#543005'][::-1])
    cbar_pp.set_under('#3F2404')
    cbar_pp.set_over('#00221A')
    cbar_pp.set_bad(color='white')

    cbar_t = colors.ListedColormap(['#9B1C00','#B9391B', '#CD4838',
                                    '#E25E55',  '#F28C89','#FFCECC',
                                    'white',
                                    '#B3DBFF', '#83B9EB', '#5E9AD7',
                                    '#3C7DC3','#014A9B','#2064AF'][::-1])
    cbar_t.set_over('#9B1C00')
    cbar_t.set_under('#014A9B')
    cbar_t.set_bad(color='white')

    if VarName.lower() == 'hgt200':
        return cbar_hgt
    elif VarName.lower() == 'pp':
        return cbar_pp
    elif VarName.lower() == 't':
        return cbar_t

# scales #######################################################################
def get_scales(VarName):
    scale_reg_hgt = [-150, -100, -75, -50, -25, -15, 0, 15, 25, 50, 75, 100, 150]

    scale_reg_pp = np.linspace(-15, 15, 13)

    scale_reg_t =  [-.6,-.4,-.2,-.1,-.05,0,0.05,0.1,0.2,0.4,0.6]

    if VarName.lower() == 'hgt200':
        return scale_reg_hgt
    elif VarName.lower() == 'pp':
        return scale_reg_pp
    elif VarName.lower() == 't':
        return scale_reg_t

################################################################################
if __name__ == "__main__":
    cbar = get_cbars(VarName)
    scale = get_scales(VarName)
################################################################################