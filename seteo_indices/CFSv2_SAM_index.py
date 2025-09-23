"""
Testeo: diferentes formas de calcular el SAM en CFSv2
"""
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/SAM_ENSO_IOD/salidas/eof/'

# ---------------------------------------------------------------------------- #
from eofs.xarray import Eof
import numpy as np

# ---------------------------------------------------------------------------- #
def Compute_SAM(hgt, model_stack=True):
    """

    :param hgt: xr.dataarray
    :param model_stack: bool, True, r y time a una misma dim, False, por leads
    :param save: bool
    :return:
    """

    weights = np.sqrt(np.abs(np.cos(np.radians(hgt.lat))))

    if model_stack:
        aux = hgt*weights
        aux_st = aux.rename({'time': 'time2'})
        aux_st = aux_st.set_index(time=('time2', 'L')).unstack('time')
        aux_st = aux_st.stack(
            time=('r', 'time2', 'L')).transpose('time', 'lat', 'lon')

        # eof ------------------------------------#
        try:
            solver = Eof(aux_st['hgt'])
        except ValueError as ve:
            if str(ve) == 'all input data is missing':
                print('campos faltantes')
                aux_st = aux_st.where(~np.isnan(aux_st), drop=True)
                solver = Eof(aux_st['hgt'])

        eof_stack = solver.eofsAsCovariance(neofs=2)
        pcs_stack = solver.pcs()

        # SAM index
        sam_stack = pcs_stack[:, 0] / pcs_stack[:, 0].std()
        sam_stack = sam_stack.unstack('time')

        return eof_stack, sam_stack





