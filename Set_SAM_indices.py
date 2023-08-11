"""
Seleccion y conversion a .nc del índice SAM.
"""
################################################################################
sam_dir = '/home/luciano.andrian/doc/SAM_ENSO_IOD/sam_monthly.csv'
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/'

# data from
# https://www.cima.fcen.uba.ar/~elio.campitelli/asymsam/data/sam_monthly.csv
################################################################################
import pandas as pd
import xarray as xr
import numpy as np
import sys
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
################################################################################
level = 700 #
sam_component_name = ['sam', 'ssam', 'asam']
test = False
################################################################################
try:
    data = pd.read_csv(sam_dir, sep=',', header=0)
except:
    print('-----------------------')
    print('el directorio no existe')
    sys.exit(1)
#------------------------------------------------------------------------------#
data['time'] = pd.to_datetime(data['time'])
data = data.loc[data.time.dt.year<=2020]

print('# Control #############################################################')
skip_anios = []
l_count = 0
j_count = 0
for s in sam_component_name:
    sam_component = data.loc[(data['index'] == s) &
                             (data['lev'] == level)]

    for i in range(0, 80):
        anioi = 1940 + i
        aniof = 1941 + i
        aux = sam_component.time.loc[(data.time.dt.year > anioi) &
                                     (data.time.dt.year <= aniof)]
        l = len(aux)

        # Todos los años tienen 12 meses?
        if l != 12:
            l_count = 1
            # print(str(aniof))
            # print('Meses: ' + str(l))
            # print(pd.to_datetime(aux.values).month)
            skip_anios.append(aniof)

        else:
            # Los que tienen los 12 meses
            # Estan en orden?
            for j in range(0, 11):
                m0 = pd.to_datetime(aux.values).month[j]
                m1 = pd.to_datetime(aux.values).month[j + 1]
                if m0 - m1 != 1:
                    j_count = 1
                    print(str(m0) + ' - ' + str(m1))
                    skip_anios.append(aniof)

if (l_count == 0) & (j_count == 0):
    print('Control OK')
else:
    print('Años salteados: ' + str(skip_anios) )

print('#######################################################################')
################################################################################

if len(sam_component_name) > 0:
    skip_anios = np.unique(skip_anios)

for y in skip_anios:
    data = data.loc[data.time.dt.year != y]

dates = xr.cftime_range(start='1940-01-01', end='2020-12-01', freq='MS')
dates = [date for date in dates if date.year not in skip_anios]

print('-----------------------------------------------------------------------')
for s in sam_component_name:
    sam_component = data.loc[(data['index'] == s) &
                             (data['lev'] == level)]

    sam_component = sam_component.drop(columns=['lev', 'index', 'time'])
    sam_component = sam_component.iloc[::-1]
    sam_component = sam_component.to_xarray()

    sam_component['index'] = dates
    sam_component = sam_component.rename({'index': 'time'})

    if test:
        print('#----- Test: ' + s + ' -----#')
        print(sam_component)
    else:
        sam_component.to_netcdf(out_dir + s + '_' + str(level) + '.nc')

################################################################################
print('#######################################################################')
print('done')
################################################################################


