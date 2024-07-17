
from cen_funciones import AUX_select_actors, regre_forplot, Plot
import xarray as xr
from Scales_Cbars import get_cbars

class CEN_ufunc:

    def __init__(self, aux_actor_list):
        self.aux_actor_list = aux_actor_list

    def pre_regre_ufunc(self, x, sets, coef, alpha):
        # , sig=False, alpha=0.05):
        """
        :param x: target, punto de grilla
        :param sets: str de actores separados por : eg. sets = 'dmi:n34'
        :param coef: 'str' coef del que se quiere beta. eg. 'dmi'
        :return: regre coef (beta)
        """
        pre_serie = {'c': x}

        parts = sets.split(':')
        sets_list = [part for part in parts if len(part) > 0]

        actor_list = self.aux_actor_list
        series_select = AUX_select_actors(actor_list, sets_list, pre_serie)
        efecto_sig, efecto_all = regre_forplot(series_select, True, coef, alpha)

        return efecto_sig, efecto_all

    def compute_regression(self, x, sets, coef, alpha):
        coef_dataset, pval_dataset = xr.apply_ufunc(
            self.pre_regre_ufunc, x, sets, coef, alpha,
            input_core_dims=[['time'], [], [], []],
            output_core_dims=[[], []],
            output_dtypes=[float, float],
            vectorize=True)

        return coef_dataset, pval_dataset


    def Compute_CEN_and_Plot(self, variables, name_variables, maps,
                             actors_and_sets_total, actors_and_sets_direc,
                             save=False, factores_sp=None, aux_name='',
                             alpha=0.05, out_dir='', actors_to_plot=None):
        if save:
            dpi = 100
        else:
            dpi = 70

        if actors_to_plot is None:
            actors_to_plot = list(actors_and_sets_total.keys())

        for v, v_name, mapa in zip(variables,
                                   name_variables,
                                   maps):

            v_cmap = get_cbars(v_name)

            for a in actors_and_sets_total:

                if a in actors_to_plot:

                    sets_total = actors_and_sets_total[a]
                    aux_sig, aux_all = (
                        self.compute_regression(v['var'], sets_total,
                                                coef=a, alpha=alpha))

                    titulo = f"{v_name} - {a} efecto total  {aux_name}"
                    name_fig = f"{v_name}_{a}_efecto_TOTAL_{aux_name}"

                    Plot(aux_sig, v_cmap, mapa, save, dpi, titulo, name_fig,
                         out_dir, data_ctn=aux_all)

                    try:
                        sets_direc = actors_and_sets_direc[a]

                        aux_sig, aux_all = (
                            self.compute_regression(v['var'], sets_direc,
                                                    coef=a,
                                                    alpha=alpha))
                        titulo = f"{v_name} - {a} efecto directo  {aux_name}"
                        name_fig = f"{v_name}_{a}_efecto_DIRECTO_{aux_name}"

                        Plot(aux_sig, v_cmap, mapa, save, dpi, titulo, name_fig,
                             out_dir, data_ctn=aux_all)

                        if factores_sp is not None:
                            sp_cmap = get_cbars('snr2')

                            try:
                                factores_sp_a = factores_sp[a]

                                for f_sp in factores_sp_a.keys():
                                    aux_f_sp = factores_sp_a[f_sp]

                                    titulo = (f"{v_name} - {a} SP Indirecto "
                                              f"via {f_sp} {aux_name}")

                                    name_fig = (f"{v_name}_{a}_SP_indirecto_"
                                                f"{f_sp}_{aux_name}")

                                    Plot(aux_f_sp * aux_sig, sp_cmap, mapa,
                                         save, dpi, titulo, name_fig, out_dir,
                                         data_ctn=aux_all)
                            except:
                                pass

                    except:
                        print('Sin efecto directo')


