"""
CEN
ENSO-IOD-U 50hPa
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas_cen/'

# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import pandas as pd
pd.options.mode.chained_assignment = None
from cen.cen_funciones import set_actor_effect_dict, apply_cen_1d

# indices -------------------------------------------------------------------- #
from CEN_set_actors import n34_or, dmi_or, u50_or

# CEN set -------------------------------------------------------------------- #
indices = {'n34':n34_or, 'dmi':dmi_or, 'u50':u50_or}

# lags, lag_variable + lags orden como indices
lags = {'SON': [10, 10, 10, 10],
        #'ASO--SON': [10, 9, 9, 9],
        'ASO': [9, 9, 9, 9],
        'JAS_ASO--SON': [10, 8, 8, 9],
        'JAS--SON': [10, 8, 8, 8]}

# CEN ------------------------------------------------------------------------ #
effects_dict_n34_dmi = set_actor_effect_dict(target='dmi',
                                     totales=['n34:n34'],
                                     directos=['n34'])
df_n34_dmi = apply_cen_1d(variable_referencia=dmi_or,
                          effects_dict=effects_dict_n34_dmi,
                          indices=indices,
                          lags=lags,
                          alpha=[0.15, 0.1, 0.05],
                          years_to_remove=[2002, 2019],
                          verbose=0)

effects_dict = set_actor_effect_dict(target='u50',
                                     totales=['dmi:dmi+n34','n34:n34',],
                                     directos=['dmi', 'n34'])
df = apply_cen_1d(variable_referencia=dmi_or,
                  effects_dict=effects_dict,
                  indices=indices,
                  lags=lags,
                  alpha=[0.15, 0.1, 0.05],
                  years_to_remove=[2002, 2019],
                  verbose=0)

# ---------------------------------------------------------------------------- #
if save:
    df_n34_dmi.to_csv(f'{out_dir}cen_n34-dmi_trinity_son.csv', index=False)
    df.to_csv(f'{out_dir}cen_trinity_son.csv', index=False)
    print('Saved')
    print(f'{out_dir}cen_n34-dmi_trinity_son.csv')
    print(f'{out_dir}cen_trinity_son.csv')

# ---------------------------------------------------------------------------- #