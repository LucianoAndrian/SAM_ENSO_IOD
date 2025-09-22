"""
Que tan bien el modelo simula el patrón del ENSO en comparación con su patrón
historico?
"""
################################################################################
save = False
################################################################################
out_dir = None
data_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
index_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
################################################################################
# import
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ENSO_IOD_Funciones import ChangeLons, Nino34CPC
################################################################################
data_em = xr.open_dataset(data_dir + 'hgt_son.nc').mean('r')
n34_em = xr.open_dataset(index_dir + 'N34_SON_Leads_r_CFSv2.nc').mean('r')