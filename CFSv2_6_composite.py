"""
Composites ENSO-IOD-SAM
Mirando ENSO, IOD y ENSO-IOD en la misma fase (como Andrian et al. 2024) y sus
combinaciones con SAM+ y SAM-
Dos figuras, dos fases ENSO-IOD, por variable
"""
# ---------------------------------------------------------------------------- #
save=True
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas_plots/comps/'
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np

from funciones.aux_select_events_CFSv2_to_plot import SelectEvents_to_Composite
from funciones.plots import Plot_ENSO_IOD_SAM_comp
from funciones.scales_and_cbars import get_scales, get_cbars

# Aux: funciones ------------------------------------------------------------- #
def set_fix(variable):
    if variable == 'prec':
        fix = 30
    elif 'hgt' in variable:
        fix = 9.8
    elif variable == 'tref':
        fix = 1
    else:
        fix = 1

    return fix

def Composite(data_to_anom, data_ref):
    num_events = len(data_to_anom.time)
    composite = data_to_anom.time('time') - data_ref.mean('time')

    return composite, num_events

def Composites_to_plot(variable, phase, data_dir,
                       plot_vertical=False):
    fix = set_fix(variable)
    events_ordenados = SelectEvents_to_Composite(data_dir, variable, phase,
                                                 plot_vertical=plot_vertical)

    composites = []
    num_events = []

    neutros = xr.open_dataset(
        f'{data_dir}{events_ordenados[-1]}').mean('time') * fix

    composites.append(neutros*np.nan)
    num_events.append('None')
    for event in events_ordenados[:-1]:
        event_select = xr.open_dataset(f'{data_dir}{event}') * fix
        num_events.append(len(event_select.time.values))

        composites.append(event_select.mean('time') - neutros)

    composites = xr.concat(composites, dim='plots')

    return composites, num_events

def Set_title(phase, plot_vertical=False):
    if phase=='pos':
        sign = '+'
        enso_phase = 'El Niño'
    else:
        sign = '-'
        enso_phase = 'La Niña'

    if plot_vertical:
        titles = ['', enso_phase, f'IOD{sign}', f'{enso_phase} & IOD{sign}',
                  f'SAM+', f'{enso_phase} & SAM+',  f'IOD{sign} & SAM+',
                  f'{enso_phase} & IOD{sign} & SAM+',
                  f'SAM-', f'{enso_phase} & SAM-',  f'IOD{sign} & SAM-',
                  f'{enso_phase} & IOD{sign} & SAM-']
    else:
        titles = ['', f'SAM+', f'SAM-',
                  enso_phase, f'{enso_phase} & SAM+', f'{enso_phase} & SAM-',
                  f'IOD{sign}', f'IOD{sign} & SAM+', f'IOD{sign} & SAM-',
                  f'{enso_phase} & IOD{sign}',
                  f'{enso_phase} & IOD{sign} & SAM+',
                  f'{enso_phase} & IOD{sign} & SAM-']

    return titles
# ---------------------------------------------------------------------------- #
data_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/events_variables/'
for phase in ['neg', 'pos']:

    hgt_comp, hgt_num = Composites_to_plot(variable='hgt', phase=phase,
                                           data_dir=data_dir)
    cbar = get_cbars('cbar_rdbu')
    scale_hgt = get_scales('hgt_comp_cfsv2')
    Plot_ENSO_IOD_SAM_comp(data=hgt_comp,
                           levels=scale_hgt, cmap=cbar,
                           titles=hgt_num, namefig=f'hgt_comp_{phase}',
                           map='hs', plots_titles=Set_title(phase),
                           save=save, out_dir=out_dir,
                           data_ctn=hgt_comp,
                           levels_ctn=None, color_ctn='k',
                           high=1, width=7.08661,
                           cbar_pos='H', plot_step=1,
                           pdf=False, ocean_mask=False,
                           data_ctn_no_ocean_mask=False,
                           sig_data=None, hatches=None)

    for plot_vertical in [False, True]:
        if plot_vertical:
            end_name = 'h'
            num_cols = 4
            num_rows = 3
            high = 2.5
            width = 7.08661
            cbar_pos = 'v'
        else:
            end_name = 'v'
            num_cols = 3
            num_rows = 4
            high = 3
            width = 5
            cbar_pos = 'H'

        prec_comp, prec_num = Composites_to_plot(variable='prec', phase=phase,
                                                 data_dir=data_dir,
                                                 plot_vertical=plot_vertical)
        cbar = get_cbars('pp')
        scale = get_scales('pp_comp_cfsv2')
        scale_hgt = get_scales('hgt_regre')
        Plot_ENSO_IOD_SAM_comp(data=prec_comp,
                               levels=scale, cmap=cbar, titles=prec_num,
                               namefig=f'prec_comp_{phase}_{end_name}',
                               map='sa',
                               plots_titles=Set_title(phase, plot_vertical),
                               save=save, out_dir=out_dir,
                               data_ctn=hgt_comp,
                               levels_ctn=scale_hgt, color_ctn='k',
                               high=high, width=width,
                               cbar_pos=cbar_pos, plot_step=1,
                               pdf=False, ocean_mask=True,
                               data_ctn_no_ocean_mask=True,
                               sig_data=None, hatches=None,
                               num_cols=num_cols, num_rows=num_rows)

        tref_comp, tref_num = Composites_to_plot(variable='tref', phase=phase,
                                                 data_dir=data_dir,
                                                 plot_vertical=plot_vertical)
        cbar = get_cbars('cbar_rdbu')
        scale = get_scales('t_comp_cfsv2')
        scale_hgt = get_scales('hgt_regre')
        Plot_ENSO_IOD_SAM_comp(data=tref_comp,
                               levels=scale, cmap=cbar, titles=tref_num,
                               namefig=f'tref_comp_{phase}_{end_name}',
                               map='sa',
                               plots_titles=Set_title(phase, plot_vertical),
                               save=save, out_dir=out_dir,
                               data_ctn=hgt_comp,
                               levels_ctn=scale_hgt, color_ctn='k',
                               high=high, width=width,
                               cbar_pos=cbar_pos, plot_step=1,
                               pdf=False, ocean_mask=True,
                               data_ctn_no_ocean_mask=True,
                               num_cols=num_cols, num_rows=num_rows)

# ---------------------------------------------------------------------------- #