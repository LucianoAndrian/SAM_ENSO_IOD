
u50_aux = u50_or.sel(
    time=~u50_or.time.dt.year.isin([2002,2019]))
u50_aux =u50_aux.sel(time=u50_aux.time.dt.year.isin(range(1959,2021)))
#u50_aux = u50_aux.sel(time=u50_aux.time.dt.month.isin([7,8,9,10,11, 12]))
dmi_aux = SameDateAs(dmi_or, u50_aux)
n34_aux = SameDateAs(n34_or, u50_aux)
asam_aux = SameDateAs(asam_or, u50_aux)
ssam_aux = SameDateAs(ssam_or, u50_aux)
u50_aux = u50_aux / u50_aux.std()
dmi_aux = dmi_aux / dmi_aux.std()
n34_aux = n34_aux / n34_aux.std()
asam_aux = asam_aux / asam_aux.std()
ssam_aux = ssam_aux / ssam_aux.std()

series = {'dmi':dmi_aux.values, 'n34':n34_aux.values, 'u50':u50_aux.values,
          'asam':asam_aux.values, 'ssam':ssam_aux.values}

from PCMCI import PCMCI
PCMCI(series=series, tau_max=3, pc_alpha=0.05, mci_alpha=0.1, mm=10, w=0,
      autocorr=True)
#
# series = {'dmi':dmi_aux.values, 'n34':n34_aux.values, 'u50':u50_aux.values,
#           'asam':asam_aux.values}
# series = {'dmi':dmi_aux.values, 'n34':n34_aux.values,
#           'asam':asam_aux.values}
#
#
# series = {'asam':asam_aux.values,'u50':u50_aux.values,
#           'ssam':ssam_aux.values}
#
# series = {'dmi':dmi_aux.values, 'n34':n34_aux.values}
# PCMCI(series=series, tau_max=5, pc_alpha=0.2, mci_alpha=0.05, mm=10, w=0)


hgt200_anom, pp, asam, ssam, u50, strato_indice2, dmi, n34, actor_list, \
    dmi_aux, n34_aux, u50_aux, sam_aux, asam_aux, ssam_aux = \
    auxSetLags_ActorList(lag_target=10,
                         lag_dmin34=9,
                         lag_strato=9,
                         hgt200_anom_or=hgt200_anom_or, pp_or=pp_or,
                         dmi_or=dmi_or, n34_or=n34_or, asam_or=asam_or,
                         ssam_or=ssam_or, sam_or=sam_or, u50_or=u50_or,
                         strato_indice=None,
                         years_to_remove=[2002, 2019])
# print('DMI, N34 - U50 ----------------------------------------------------')
# aux_alpha_CN_Effect_2(actor_list,
#                       set_series_directo=['u50', 'n34'],
#                       set_series_totales={'u50': ['u50', 'n34'],
#                                           'n34': ['n34']},
#                       variables={'dmi': dmi},
#                       sig=True, alpha_sig=[0.05, 0.1, 0.15, 1])

aux_alpha_CN_Effect_2(actor_list,
                      set_series_directo=['dmi', 'n34', 'u50', 'ssam',
                                          'aux_ssam'],
                      set_series_totales={'dmi': ['dmi', 'n34', 'aux_ssam'],
                                          'n34': ['n34', 'dmi', 'aux_ssam'],
                                          'u50': ['u50', 'n34', 'dmi', 'ssam',
                                                  'aux_ssam'],
                                          'ssam':['ssam', 'u50', 'n34', 'dmi',
                                                  'aux_ssam'],
                                          'aux_ssam':['aux_ssam']},
                      variables={'asam': asam},
                      sig=True, alpha_sig=[0.05, 0.1, 0.15, 1])



aux_alpha_CN_Effect_2(actor_list,
                      set_series_directo=['dmi', 'n34', 'u50', 'ssam',
                                          ],
                      set_series_totales={'dmi': ['dmi', 'n34', ],
                                          'n34': ['n34', 'dmi', ],
                                          'u50': ['u50', 'n34', 'dmi', 'ssam',
                                                  ],
                                          'ssam':['ssam', 'u50', 'n34', 'dmi',
                                                  ]},
                      variables={'asam': asam},
                      sig=True, alpha_sig=[0.05, 0.1, 0.15, 1],
                      set_series_directo_particulares=None)


aux_alpha_CN_Effect_2(actor_list,
                      set_series_directo=['dmi', 'n34', 'u50', 'ssam'],
                      set_series_totales={'dmi': ['n34', 'aux_ssam'],
                                          'n34': ['n34'],
                                          'u50': ['u50', 'n34', 'dmi', 'u50_aux', 'aux_ssam'],
                                          'ssam':['ssam', 'u50', 'n34', 'dmi',
                                                  'aux_ssam', 'u50_aux']},
                      variables={'asam': asam},
                      sig=True, alpha_sig=[0.05, 0.1, 0.15, 1],
                      set_series_directo_particulares=
                      {'dmi':['dmi','n34', 'ssam', 'u50'],
                       'n34':['n34', 'dmi', 'ssam', 'u50'],
                       'ssam':['ssam', 'u50', 'n34', 'dmi', 'aux_ssam', 'u50_aux'],
                       'u50':['u50', 'ssam', 'n34', 'dmi', 'aux_ssam', 'u50_aux']})


aux_alpha_CN_Effect_2(actor_list,
                      set_series_directo=['dmi', 'n34', 'u50', 'ssam'],
                      set_series_totales={'dmi': ['n34'],
                                          'n34': ['n34'],
                                          'u50': ['u50', 'n34', 'dmi'],
                                          'ssam':['ssam', 'u50', 'n34', 'dmi']},
                      variables={'asam': asam},
                      sig=True, alpha_sig=[0.05, 0.1, 0.15, 1],
                      set_series_directo_particulares=
                      {'dmi':['dmi','n34', 'ssam', 'u50'],
                       'n34':['n34', 'dmi', 'ssam', 'u50'],
                       'ssam':['ssam', 'u50', 'n34', 'dmi'],
                       'u50':['u50', 'ssam', 'n34', 'dmi']})