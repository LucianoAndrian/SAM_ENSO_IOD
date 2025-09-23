"""

"""
# ---------------------------------------------------------------------------- #
from Funcion_CFSv2_SelectEvents import Compute
# ---------------------------------------------------------------------------- #
out_dir = ('/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/IOD_hLat_cic'
           '/cases_events/')
dates_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
# Eventos DMI y U50 - umbral 0.5*SD ------------------------------------------ #
Compute(['DMI', 'U50'], ['MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND'],
        dates_dir=dates_dir, out_dir=out_dir)
# ---------------------------------------------------------------------------- #