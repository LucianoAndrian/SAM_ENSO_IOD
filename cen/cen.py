import logging
from cen.cen_funciones import AUX_select_actors, regre_forplot
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class CEN_ufunc:

    def __init__(self, aux_actor_list):
        self.aux_actor_list = aux_actor_list
        logger.info("Inicializando CEN_ufunc con lista de actores: %s", aux_actor_list)

    def pre_regre_ufunc(self, x, sets, coef, alpha):
        logger.debug("Entrando en pre_regre_ufunc con coef=%s, sets=%s, alpha=%s",
                     coef, sets, alpha)
        try:
            pre_serie = {'c': x}

            if isinstance(sets, str):
                parts = sets.split('+')
            else:
                parts = sets

            sets_list = parts#[part for part in parts if len(part) > 0]
            actor_list = self.aux_actor_list
            series_select = AUX_select_actors(actor_list, sets_list, pre_serie)
            efecto_sig, efecto_all = regre_forplot(series_select, True, coef, alpha)

            logger.debug("pre_regre_ufunc finalizado correctamente para coef=%s", coef)
            return efecto_sig, efecto_all

        except Exception as e:
            logger.exception("Error en pre_regre_ufunc con coef=%s, sets=%s", coef, sets)
            raise

    def compute_regression_in_parallel(self, x, sets, coef, alpha):
        logger.info("Iniciando compute_regression_in_parallel con coef=%s", coef)
        try:
            coef_dataset, pval_dataset = xr.apply_ufunc(
                self.pre_regre_ufunc, x, sets, coef, alpha,
                input_core_dims=[['time'], [], [], []],
                output_core_dims=[[], []],
                output_dtypes=[float, float],
                vectorize=True)

            logger.info("compute_regression_in_parallel finalizado para coef=%s", coef)
            return coef_dataset, pval_dataset

        except Exception as e:
            logger.exception("Error en compute_regression_in_parallel con coef=%s", coef)
            raise

    def run_ufunc_cen(self, variable_target, effects, alpha=0.05):
        logger.info("Ejecutando run_ufunc_cen con alpha=%s", alpha)

        try:
            variable_name = list(variable_target.data_vars)[0]
            logger.debug("Variable objetivo detectada: %s", variable_name)
        except Exception as e:
            logger.error("No se pudo encontrar nombre de variable en variable_target")
            variable_name = list(variable_target.data_vars)

        efectos_directos = effects['efectos_directos']
        efectos_totales = effects['efectos_totales']
        efectos_directos_particulares = effects['efectos_directos_particulares']

        regre_efectos_totales = {}
        for a in efectos_totales.keys():
            logger.info("Procesando efecto total: %s", a)
            try:
                aux_efectos_totales = efectos_totales[a]
                a_regre_total_sig, a_regre_total = \
                    self.compute_regression_in_parallel(
                        x=variable_target[variable_name],
                        sets=aux_efectos_totales,
                        coef=a,
                        alpha=alpha)

                regre_efectos_totales[a] = a_regre_total
                regre_efectos_totales[f'{a}_sig'] = a_regre_total_sig
            except Exception:
                logger.exception("Fallo al calcular efectos_totales para %s", a)
                raise

        regre_efectos_directos = {}
        for ad in efectos_directos.keys():
            logger.info("Procesando efecto directo: %s", ad)
            try:
                aux_efectos_directos = efectos_directos[ad]
                ad_regre_directo_sig, ad_regre_directo = \
                    self.compute_regression_in_parallel(
                        x=variable_target[variable_name],
                        sets=aux_efectos_directos,
                        coef=ad,
                        alpha=alpha)

                regre_efectos_directos[ad] = ad_regre_directo
                regre_efectos_directos[f'{ad}_sig'] = ad_regre_directo_sig
            except Exception:
                logger.exception("Fallo al calcular efectos_directos para %s", ad)
                raise

        logger.info("Finalizado run_ufunc_cen correctamente")
        return regre_efectos_totales, regre_efectos_directos
