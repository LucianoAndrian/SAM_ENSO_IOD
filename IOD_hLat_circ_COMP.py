"""
IOD - SAM/U50 comp
"""
# ---------------------------------------------------------------------------- #
save = True

out_dir = '/pikachu/datos/luciano.andrian/IOD_vs_hLat/comp/'
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import IOD_hLat_cic_SET_VAR
from Scales_Cbars import get_cbars

# ---------------------------------------------------------------------------- #
if save:
    dpi = 200
else:
    dpi = 100
# ---------------------------------------------------------------------------- #
def SelectDMI_SAM_u50(dmi_df, indice_serie, dmi_serie, mm_dmi, mm_ind,
                      use_dmi_df = False, ind_name='IND'):

    if use_dmi_df:
        # DMI pos
        dmi_pos = dmi_df[
            dmi_df['DMI'] > 0]  # acá estan los que cumple DMIIndex()
        # DMI neg
        dmi_neg = dmi_df[dmi_df['DMI'] < 0]
    else:
        # dmi pos
        dmi_pos = dmi_serie.where(dmi_serie > .5, drop=True)
        # dmi neg
        dmi_neg = dmi_serie.where(dmi_serie < -.5, drop=True)

    # indice pos
    ind_pos = indice_serie.where(indice_serie > .5, drop=True)
    # indice neg
    ind_neg = indice_serie.where(indice_serie < -.5, drop=True)

    m_count = 0
    output = {}
    for m_dmi, m_ind in zip(mm_dmi, mm_ind):
        ar_result = np.zeros((13,1))

        dmi_serie_sel = dmi_serie.where(dmi_serie.time.dt.month == m_dmi, drop=True)
        indice_serie_sel = indice_serie.where(indice_serie.time.dt.month == m_ind,
                                              drop=True)
        if use_dmi_df:
            dmi_pos_sel = dmi_pos[dmi_pos['Mes'] == m_dmi]
            dmi_neg_sel = dmi_neg[dmi_neg['Mes'] == m_dmi]
        else:

            dmi_pos_sel = dmi_pos.where(dmi_pos.time.dt.month == m_dmi,
                                        drop=True)
            dmi_neg_sel = dmi_neg.where(dmi_neg.time.dt.month == m_dmi,
                                        drop=True)


        ind_pos_sel = ind_pos.where(ind_pos.time.dt.month == m_ind, drop=True)
        ind_neg_sel = ind_neg.where(ind_neg.time.dt.month == m_ind, drop=True)

        if use_dmi_df:
            ind_dmi_pos = ind_pos_sel.where(
                ind_pos_sel.time.dt.year.isin(dmi_pos_sel['Años'].values),
                drop=True)

            ind_dmi_neg = ind_neg_sel.where(
                ind_neg_sel.time.dt.year.isin(dmi_neg_sel['Años'].values),
                drop=True)

            ind_pos_dmi_neg = ind_pos_sel.where(
                ind_pos_sel.time.dt.year.isin(dmi_neg_sel['Años'].values),
                drop=True)

            ind_neg_dmi_pos = ind_neg_sel.where(
                ind_neg_sel.time.dt.year.isin(dmi_pos_sel['Años'].values),
                drop=True)

            dmi_pos_sel_puros = dmi_pos_sel[~dmi_pos_sel['Años'].isin(
                ind_dmi_pos.time.dt.year.values)]
            dmi_pos_sel_puros = dmi_pos_sel_puros[
                ~dmi_pos_sel_puros['Años'].isin(
                    ind_neg_dmi_pos.time.dt.year.values)]

            dmi_neg_sel_puros = dmi_neg_sel[~dmi_neg_sel['Años'].isin(
                ind_dmi_neg.time.dt.year.values)]
            dmi_neg_sel_puros = dmi_neg_sel_puros[
                ~dmi_neg_sel_puros['Años'].isin(
                    ind_pos_dmi_neg.time.dt.year.values)]

        else:

            ind_dmi_pos = ind_pos_sel.where(
                ind_pos_sel.time.dt.year.isin(dmi_pos_sel.time.dt.year.values),
                drop=True)

            ind_dmi_neg = ind_neg_sel.where(
                ind_neg_sel.time.dt.year.isin(dmi_neg_sel.time.dt.year.values),
                drop=True)

            ind_pos_dmi_neg = ind_pos_sel.where(
                ind_pos_sel.time.dt.year.isin(dmi_neg_sel.time.dt.year.values),
                drop=True)

            ind_neg_dmi_pos = ind_neg_sel.where(
                ind_neg_sel.time.dt.year.isin(dmi_pos_sel.time.dt.year.values),
                drop=True)

            dmi_pos_sel_puros = dmi_pos_sel.sel(
                time=~dmi_pos_sel.time.dt.year.isin(
                    ind_dmi_pos.time.dt.year.values))
            dmi_pos_sel_puros = dmi_pos_sel_puros.sel(
                time=~dmi_pos_sel_puros.time.dt.year.isin(
                    ind_neg_dmi_pos.time.dt.year.values))

            dmi_neg_sel_puros = dmi_neg_sel.sel(
                time=~dmi_neg_sel.time.dt.year.isin(
                    ind_dmi_neg.time.dt.year.values))
            dmi_neg_sel_puros = dmi_neg_sel_puros.sel(
                time=~dmi_neg_sel_puros.time.dt.year.isin(
                    ind_pos_dmi_neg.time.dt.year.values))

        ind_pos_sel_puros = ind_pos_sel.sel(
            time=~ind_pos_sel.time.dt.year.isin(
                ind_dmi_pos.time.dt.year.values))
        ind_pos_sel_puros = ind_pos_sel_puros.sel(
            time=~ind_pos_sel_puros.time.dt.year.isin(
                ind_pos_dmi_neg.time.dt.year.values))

        ind_neg_sel_puros = ind_neg_sel.sel(
            time=~ind_neg_sel.time.dt.year.isin(
                ind_dmi_neg.time.dt.year.values))
        ind_neg_sel_puros = ind_neg_sel_puros.sel(
            time=~ind_neg_sel_puros.time.dt.year.isin(
                ind_neg_dmi_pos.time.dt.year.values))

        # ind_events = (len(ind_pos_sel_puros.time.values) +
        #               len(ind_neg_sel_puros.time.values))
        #
        # dmi_events = (len(dmi_pos_sel_puros) +
        #               len(dmi_neg_sel_puros))
        #
        # in_phase = len(ind_dmi_pos.time.values) + len(ind_dmi_neg.time.values)
        #
        # out_phase = (len(ind_pos_dmi_neg.time.values) +
        #              len(ind_neg_dmi_pos.time.values))

        ind_neutros = indice_serie_sel.where(abs(indice_serie_sel) < .5,
                                             drop=True)
        if use_dmi_df:
            dmi_neutros = dmi_serie_sel.sel(
                time=~dmi_serie_sel.time.dt.year.isin(
                    dmi_df['Años'].values))
        else:
            dmi_neutros = dmi_serie_sel.sel(
                time=~dmi_serie_sel.time.dt.year.isin(
                    dmi_pos_sel.time.dt.year.values))
            dmi_neutros = dmi_neutros.sel(
                time=~dmi_neutros.time.dt.year.isin(
                    dmi_neg_sel.time.dt.year.values))
        try:
            neutros = xr.concat([ind_neutros, dmi_neutros], dim='time')
        except:
            ind_neutros = ind_neutros.drop('month')
            neutros = xr.concat([ind_neutros, dmi_neutros], dim='time')

        ds = xr.Dataset(
            data_vars={
                'Neutral': (neutros.time.values),
                "DMI_sim_pos": (ind_dmi_pos.time.values),
                "DMI_sim_neg": (ind_dmi_neg.time.values),
                "DMI_puros_pos": (dmi_pos_sel_puros.time.values),
                "DMI_puros_neg": (dmi_neg_sel_puros.time.values),
                f"{ind_name}_puros_pos": (ind_pos_sel_puros.time.values),
                f"{ind_name}_puros_neg": (ind_neg_sel_puros.time.values),
                f'{ind_name}_pos_DMI_neg': (ind_pos_dmi_neg.time.values),
                f'{ind_name}_neg_DMI_pos': (ind_neg_dmi_pos.time.values),
            }
        )

        output[str(m_dmi)] = ds

    return output

def PlotComposite(variable, data_dates, cases,
                  scale, cmap, save, title, out_dir=out_dir):

    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs

    for m , m_values in zip(data_dates.keys(), data_dates.values()):

        neutro = m_values.Neutral

        for c_count, c in enumerate(cases):
            case = m_values[c]

            if c != 'Neutral':
                var_neutro = variable.sel(time=neutro)
                var_case = variable.sel(time=case)

                comp = var_case.mean(c) - var_neutro.mean('Neutral')

                crs_latlon = ccrs.PlateCarree()

                if save:
                    dpi = 200
                else:
                    dpi = 80

                ratio = len(comp.lat) / len(comp.lon)
                fig = plt.figure(figsize=(10, 10 * ratio), dpi=dpi)

                ax = plt.axes(
                    projection=ccrs.PlateCarree(central_longitude=180))
                crs_latlon = ccrs.PlateCarree()
                ax.set_extent([comp.lon.values[0],
                               comp.lon.values[-1],
                               min(comp.lat.values),
                               max(comp.lat.values)], crs=crs_latlon)

                im = ax.contourf(comp.lon, comp.lat,
                                 comp[list(comp.data_vars)[0]],
                                 levels=scale,
                                 transform=crs_latlon, cmap=cmap, extend='both')

                cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
                cb.ax.tick_params(labelsize=8)

                ax.coastlines(color='k', linestyle='-', alpha=1)
                ax.set_xticks(np.arange(comp.lon.values[0],
                                        comp.lon.values[-1], 20),
                              crs=crs_latlon)
                ax.set_yticks(np.arange(min(comp.lat.values),
                                        max(comp.lat.values), 10),
                              crs=crs_latlon)
                lon_formatter = LongitudeFormatter(zero_direction_label=True)
                lat_formatter = LatitudeFormatter()
                ax.xaxis.set_major_formatter(lon_formatter)
                ax.yaxis.set_major_formatter(lat_formatter)

                ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
                ax.tick_params(labelsize=10)

                plt.title(f'{cases[c_count]} - rm{title} - mm: {m} - '
                          f'N: {len(var_case[c].values)}',
                          fontsize=12)

                namefig = f'{cases[c_count]}_rm-{title}_mm-{m}'

                plt.tight_layout()
                if save:
                    plt.savefig(f'{out_dir}{namefig}.jpg')
                    print(f'Saved {out_dir}{namefig}.jpg')
                    plt.close()
                else:
                    plt.show()

# ---------------------------------------------------------------------------- #
(dmi_or_1rm, dmi_or_2rm, dmi_or_3rm, sam_or_1rm, sam_or_2rm, sam_or_3rm,
 u50_or_1rm, u50_or_2rm, u50_or_3rm, hgt200_anom_or_1rm, hgt200_anom_or_2rm,
 hgt200_anom_or_3rm) = IOD_hLat_cic_SET_VAR.compute()
# ---------------------------------------------------------------------------- #
cbar = get_cbars('hgt')

meses = [8, 9, 10]

dmis = [dmi_or_1rm, dmi_or_2rm, dmi_or_3rm]
sams = [sam_or_1rm, sam_or_2rm, sam_or_3rm]
u50s = [u50_or_1rm['var'], u50_or_2rm['var'], u50_or_3rm['var']]
data = [hgt200_anom_or_1rm, hgt200_anom_or_2rm, hgt200_anom_or_3rm]

for rl in [0,1,2]:
    aux_indices = [sams[rl], u50s[rl]]
    aux_dmi = dmis[rl]
    aux_data = data[rl]

    for index_name, index in zip(['sam', 'u50'], aux_indices):
        aux = SelectDMI_SAM_u50(None, index, aux_dmi, meses, meses,
                                ind_name=index_name)

        PlotComposite(variable=hgt200_anom_or_1rm, data_dates=aux,
                      cases=list(aux[list(aux.keys())[0]].dims),
                      scale=np.arange(-1,1.1,0.1), cmap=cbar, save=save,
                      title=f'{rl+1}',out_dir=out_dir)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #