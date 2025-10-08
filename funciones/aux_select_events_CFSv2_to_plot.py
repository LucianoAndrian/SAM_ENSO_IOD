"""
funcion especifica para CFSv2_6_composite.py
"""
# ---------------------------------------------------------------------------- #
import os

# ---------------------------------------------------------------------------- #
def SelectEvents_to_Composite(data_dir, variable, phase='pos',
                              plot_transpuesta=False):

    idx1 = 'n34'
    idx2 = 'dmi'
    idx3 = 'sam'

    if phase == 'pos':
        op_phase = 'neg'
    elif phase == 'neg':
        op_phase = 'pos'

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.nc')]
    files = [f for f in files if f'{variable}_' in f]

    # Puros: 4
    idx1_puro = [f for f in files if ('puros' in f)
                 and (f'{idx1}_{phase}' in f)][0]
    idx2_puro = [f for f in files if ('puros' in f)
                 and (f'{idx2}_{phase}' in f)][0]
    idx3_puro_phase = [f for f in files if ('puros' in f)
                       and (f'{idx3}_pos' in f)][0]
    idx3_puro_op_phase = [f for f in files if ('puros' in f)
                          and (f'{idx3}_neg' in f)][0]

    # Dobles: 4
    idx1_phase_idx3_phase = [f for f in files if
                             ('dobles' in f) and
                             ('op' not in f) and
                             (f'{idx2}' not in f) and
                             (f'{phase}' in f)][0]

    idx1_phase_idx3_op_phase = [f for f in files if
                                ('dobles_op' in f) and
                                (f'{idx2}' not in f) and
                                (f'{idx1}_{phase}' in f) and
                                (f'{idx3}_{op_phase}' in f)][0]

    idx2_phase_idx3_phase = [f for f in files if
                             ('dobles' in f) and
                             ('op' not in f) and
                             (f'{idx1}' not in f) and
                             (f'{phase}' in f)][0]

    idx2_phase_idx3_op_phase = [f for f in files if
                                ('dobles_op' in f) and
                                (f'{idx1}' not in f) and
                                (f'{idx2}_{phase}' in f) and
                                (f'{idx3}_{op_phase}' in f)][0]

    idx1_idx2_same_phase = [f for f in files if
                            ('dobles' in f) and
                            ('op' not in f) and
                            (f'{idx3}' not in f) and
                            (f'{phase}' in f)][0]

    # Triples: 2
    triple_same_phase = [f for f in files if
                         (f'triples_' in f) and
                         (f'opuestos' not in f) and
                         (f'{phase}' in f)][0]

    triple_op_phase = [f for f in files if
                       (f'triples_opuestos' in f) and
                       (f'{idx1}_{phase}' in f) and
                       (f'{idx2}_{phase}' in f)][0]

    # neutros
    neutros = [f for f in files if 'neutros' in f][0]

    # orden para plotear
    if plot_transpuesta is True:
        if 'pos' in phase.lower():
            orden = [idx1_puro, idx2_puro, idx1_idx2_same_phase,
                     idx3_puro_phase, idx1_phase_idx3_phase,
                     idx2_phase_idx3_phase,  triple_same_phase,
                     idx3_puro_op_phase, idx1_phase_idx3_op_phase,
                     idx2_phase_idx3_op_phase, triple_op_phase,
                     neutros]
        else:
            orden = [idx1_puro, idx2_puro, idx1_idx2_same_phase,
                     idx3_puro_phase, idx1_phase_idx3_op_phase,
                     idx2_phase_idx3_op_phase, triple_op_phase,
                     idx3_puro_op_phase, idx1_phase_idx3_phase,
                     idx2_phase_idx3_phase,  triple_same_phase,
                     neutros]

    else:
        if 'pos' in phase.lower():
            orden = [idx3_puro_phase, idx3_puro_op_phase,
                     idx1_puro, idx1_phase_idx3_phase, idx1_phase_idx3_op_phase,
                     idx2_puro, idx2_phase_idx3_phase, idx2_phase_idx3_op_phase,
                     idx1_idx2_same_phase, triple_same_phase, triple_op_phase,
                     neutros]
        else:
            orden = [idx3_puro_phase, idx3_puro_op_phase,
                     idx1_puro, idx1_phase_idx3_op_phase, idx1_phase_idx3_phase,
                     idx2_puro, idx2_phase_idx3_op_phase, idx2_phase_idx3_phase,
                     idx1_idx2_same_phase, triple_op_phase, triple_same_phase,
                     neutros]

    return orden
# ---------------------------------------------------------------------------- #