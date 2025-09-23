"""
Versión simple del algoritmo PCMCI (Runge et al., 2019)

Funcionamiento restringido respecto a tigramite:
 - Sólo funciona en el modo de series enmascaradas de tigramite
Tener en cuenta:
 - el testeo de los parents en la parte de PC puede resultar más restrictiva que
 en tigramite.
"""
################################################################################
import warnings
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
from scipy.stats import pearsonr
################################################################################
def select_positions(serie, pos, window, lag=0):
    """
    Selecciona las posiciones de la serie de entrada en función de la ventana y
    el lag
    :param serie: numpy.ndarray, serie 1D, preferentemente de jan to dec
    :param pos: int, MES del año
    :param window: int o list. Si es int. del mes elegido se va tomar +-1 mes a
                   atras y adelante, si es una lista, el primer elemento va
                   indicar cuantos meses hacia atras y el 2do cuantos meses
                   hacia adelante
    :param lag: int, cuanto se desplaza el mes elegido
    :return: serie con los valores seleccionados

    ej:
    pos=10 (octubre), window=1 (+-1 mes), lag=1 (sep)
    return A, S, O, A, S, O....

    pos=10 (octubre), window=[1,0] (1 mes atras y 0 adelante), lag=0 (sep)
    return A, S, A, S, ...
    """

    selected_values = []
    length = len(serie)
    step = 12 # saltando de a un año
    pos -= lag

    try:
        if len(window) == 2:
            w_b = window[0]
            w_f = window[1]
        elif len(window) > 2:
            w_b = window[0]
            w_f = window[0]
            print("Error, window must be int. or list < 2, using window[0]")
    except:
        w_b = window
        w_f = window

    for i in range(0, length//12 + 1):
        selected_values.extend(serie[pos-w_b-1+step*i:pos+w_f+step*i])

    return selected_values

def SetLags2(x, y, ty, series, parents, mm, w):
    """

    :param x:  numpy.ndarray, serie 1D que no va ser lagueada
    :param y:  numpy.ndarray, serie 1D que va ser lagueada
    :param ty: int, lag de la serie y
    :param series: dict, todas las series que entran al algoritmo
    :param parents: dict, parents de las series x e y
    :param pos: int, MES del año
    :param window: int o list. Si es int. del mes elegido se va tomar +-1 mes a
                   atras y adelante, si es una lista, el primer elemento va
                   indicar cuantos meses hacia atras y el 2do cuantos meses
                   hacia adelante
    :return: dataframe con las series x e y seteadas respecto al lag de la
             serie y y sus respectivos parents.
    """
    x = select_positions(x, mm, w, 0)
    y = select_positions(y, mm, w, ty)

    z_series = []
    for i, p  in enumerate(parents):
        zn = series[p.split('_lag_')[0]]
        tz = np.int(p.split('_lag_')[1])

        z_series.append(select_positions(zn, mm, w, tz))

    z_columns = [f'parent{i}' for i in range(1, len(z_series) + 1)]
    z_data = {column: z_series[i] for i, column in enumerate(z_columns)}

    return pd.DataFrame({'x':x, 'y':y, **z_data})

def PartialCorrelation(df):
    """
    Correlación parcial a partir de los residuos de las series x e y, calculados
    a partir de la regresión lineal de los parents de ambas.

    (La primera versión de esta funcion era muy lenta en xarray_ufunc)

    :param df: dataframe de SetLags2
    :return: int. r coef. de correlacion y pv pvalue
    """
    x = df['x'].values
    y = df['y'].values

    X = np.column_stack((np.ones_like(x), df[df.columns[2:]].values))
    beta_x = np.linalg.lstsq(X, x, rcond=None)[0]
    x_res = x - np.dot(X, beta_x)

    beta_y = np.linalg.lstsq(X, y, rcond=None)[0]
    y_res = y - np.dot(X, beta_y)
    r, pv = pearsonr(x_res, y_res)

    return r, pv

def SetParents(parents, alpha, withtarget=False):
    """
    Testea y ordena los parents resultado de PartialCorrelation.

    :param parents: dataframe, con el nombre del parents, r y pv
    :param alpha: int, valor de significancia
    :param withtarget: bool, para el paso MCI y dar los valores por pantalla
    :return: dataframe, con target, actor, r y pv
    """
    parents = parents[parents['pval'] < alpha]
    if withtarget:
        parents = parents.assign(abs_r=parents['r'].abs()).sort_values(
            by=['Target', 'abs_r'], ascending=[True, False])
        parents = parents.drop(columns=['abs_r'])
    else:
        parents = parents.iloc[parents['r'].abs().argsort()[::-1]]
    parents['r'] = parents['r'].round(3)
    parents['pval'] = parents['pval'].round(3)

    return parents

def PC(series, target, tau_max, pc_alpha, mm, w, autocorr):
    """
    Algoritmo PC (o parte de el)

    :param series: dict, todas las series que se quieren evaluar
    :param target: character, key de una de las entradas de series
    :param tau_max: int, valor maximo del lag. Es positivo pero va para atras
    :param pc_alpha: int, valor de significancia
    :param mm: int, MES idem anteriores
    :param w: int o list, idem anteriores
    :param: autocorr: bool, si es True la autocorrelación va ser tenida en
            cuenta al igual que en el PCMCM original. Si es False no lo hará
    :return: list, nombre de los parents
    """
    taus = np.arange(1, tau_max + 1)
    # Set preliminary parents ------------------------------------------------ #

    if autocorr:
        no_tau = [0]  # solo por las dudas si algun dia se usa taus desde 0 en PC
    else:
        no_tau = taus

    # Correlation
    first = True
    target_serie = select_positions(series[target], mm, w, 0)
    for k in series.keys():
            for t in taus:
                if t in no_tau and k==target:
                    # NO si mismo.
                    pass
                else:
                    k_serie = select_positions(series[k], mm, w, t)

                    r, pv = pearsonr(target_serie, k_serie)

                    d = {'pparents': k + '_lag_' + str(t), 'r': [r],
                         'pval': [pv]}

                    if first:
                        first = False
                        parents0 = pd.DataFrame(d)
                    else:
                        parents0 = pd.concat([parents0, pd.DataFrame(d)],
                                             axis=0)
    parents = SetParents(parents0, pc_alpha)
    # ------------------------------------------------------------------------ #
    # Partial correlation
    i = 0
    while len(parents) > 2:
        strong_parents = parents['pparents'].head(i+2).tolist()
        first = True
        for p in parents['pparents']:

            # Select strong parent/s for partial correlation
            aux_strong_parents = strong_parents[:i+1] if \
                all(parent != p for parent in strong_parents) else \
                [parent for parent in strong_parents if parent != p]

            serie_p = p.split('_lag_')[0]
            t_p = np.int(p.split('_lag_')[1])

            df = SetLags2(series[target], series[serie_p], ty=t_p,
                          series=series, parents=aux_strong_parents,
                          mm=mm, w=w)

            r, pv = PartialCorrelation(df)

            d = {'pparents': serie_p + '_lag_' + str(t_p),
                 'r': [r], 'pval': [pv]}

            if first:
                first = False
                parents1 = pd.DataFrame(d)
            else:
                parents1 = pd.concat([parents1, pd.DataFrame(d)], axis=0)

        parents = SetParents(parents1, pc_alpha)
        i += 1
        if i > 5:
            break

    parents_name=[]
    for p in parents['pparents']:
        parents_name.append(p)

    return parents_name

def add_lag(parents, plus_lag=1):
    """
    Agrega lag al nombre de los parents en MCI
    """
    parents_add_lag = []
    for p in parents:
        pre, lag = p.split('_lag_')
        lag = int(lag) + plus_lag
        parents_add_lag.append(pre + '_lag_' + str(lag))

    return parents_add_lag

def MCI(series, targets, tau_max, parents, mci_alpha, mm, w, autocorr):
    """
    Algoritmo MCI (o parte de el)

    :param series: dict, todas las series que se quieren evaluar
    :param target: dict, target y parents en cada elemento del dict
    :param tau_max: int, valor maximo del lag. Es positivo pero va para atras
    :param pc_alpha: int, valor de significancia
    :param mm: int, MES idem anteriores
    :param w: int o list, idem anteriores
    :param: autocorr: bool, si es True la autocorrelación va ser tenida en
            cuenta al igual que en el PCMCM original. Si es False no lo hará
    :return: dataframe, con target, actor, r y pv
    """

    lags = np.arange(0, tau_max + 1)

    if autocorr:
        no_lag = [0]
    else:
        no_lag = lags

    first = True
    for target in targets:
        target_parents_original = parents[target].copy()

        for l in lags:
            for a in targets:
                if l in no_lag and a == target:
                    # NO si mismo.
                    pass
                else:
                    actor_parents = parents[a].copy()
                    actor_as_target_parent = a + '_lag_' + str(l)

                    target_parents = target_parents_original.copy()

                    # esto esta OK.
                    if actor_as_target_parent in target_parents_original:
                        target_parents.remove(actor_as_target_parent)

                    target_actor_parents = target_parents + \
                                           add_lag(actor_parents, l)

                    # Test
                    target_actor_parents = list(set(target_actor_parents))

                    # esto es debido a la longitud de las series en los casos
                    # donde el add_lag se va a la mierda falla pero no tienen
                    # mucho sentido por ahora esos lags
                    try:
                        df = SetLags2(series[target], series[a], ty=l,
                                      series=series,
                                      parents=target_actor_parents, mm=mm, w=w)
                        r, pv = PartialCorrelation(df)
                    except:
                        r, pv = 0, 1

                    d = {'Target': target, 'Actor': a + '_lag_' + str(l),
                         'r': [r], 'pval': [pv]}

                    if first:
                        first = False
                        parents_f = pd.DataFrame(d)
                    else:
                        parents_f = pd.concat([parents_f, pd.DataFrame(d)],
                                              axis=0)


    links = SetParents(parents_f, mci_alpha, True)

    return links


def PCMCI(series, tau_max, pc_alpha, mci_alpha, mm, w, autocorr):
    """
    algoritmo PCMCI
    :param series: dict, todas las series que se quieren evaluar
    :param target: dict, target y parents en cada elemento del dict
    :param tau_max: int, valor maximo del lag. Es positivo pero va para atras
    :param pc_alpha: int, valor de significancia
    :param mm: int, MES idem anteriores
    :param w: int o list, idem anteriores
    :param: autocorr: bool, si es True la autocorrelación va ser tenida en
            cuenta al igual que en el PCMCM original. Si es False no lo hará
    :return: dataframe, con target, actor, r y pv
    """
    targets = []
    targets_parents= {}
    # PC --------------------------------------------------------------------- #
    for s in series.keys():
        targets.append(s)
        targets_parents.update({s:PC(series, s, tau_max, pc_alpha, mm, w,
                                     autocorr)})
    print(targets_parents)
    # MCI -------------------------------------------------------------------- #
    links = MCI(series,targets, tau_max, targets_parents, mci_alpha, mm, w,
                autocorr)
    return links

# ---------------------------------------------------------------------------- #
def example():
    print("Ejemplo de uso PCMCI ----------------------------------------------")
    print("import numpy as np")
    print("from PCMCI import PCMCI")

    print("np.random.seed(0)")
    print("x = np.random.randn(100)")
    print("y = np.zeros_like(x)")
    print("y[3:] = 0.8 * x[:-3] + 0.2 * np.random.randn(100 - 3)")
    print("z = np.zeros_like(x)")
    print("z[3:] = np.sin(x[:-3]) + 0.5 * np.random.randn(100 - 3)")
    print("")

    print("")
    print("series = {'x': x, 'y': y, 'z': z}")
    print("")
    print("PCMCI(series=series, tau_max=3, pc_alpha=0.1, mci_alpha=0.1, "
          "mm=10, w=1, autocorr=True)")
    print("")
    np.random.seed(0)
    x = np.random.randn(100)
    y = np.zeros_like(x)
    y[3:] = 0.8 * x[:-3] + 0.2 * np.random.randn(100 - 3)
    z = np.zeros_like(x)
    z[3:] = np.sin(x[:-3]) + 0.5 * np.random.randn(100 - 3)

    from PCMCI import PCMCI
    print("resultado: ")
    series = {'x':x, 'y':y, 'z':z}
    return PCMCI(series=series, tau_max=3, pc_alpha=0.1, mci_alpha=0.1,
                 mm=10, w=1, autocorr=True)
# ---------------------------------------------------------------------------- #