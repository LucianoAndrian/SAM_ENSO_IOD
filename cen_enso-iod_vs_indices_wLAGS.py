# if use_strato_index:
#     print(f"use_lags = False, no se puede usar Strato_index")
#
# hgt_anom_lag1 = hgt_anom_or.sel(time=hgt_anom_or.time.values[1:-1])
# hgt_anom_lag2 = hgt_anom_or.sel(time=hgt_anom_or.time.values[:-2])
# hgt_anom = hgt_anom_or.sel(time=hgt_anom_or.time.values[2:])
#
# dmi = SameDateAs(dmi_or, hgt_anom)
# dmi_lag1 = SameDateAs(dmi_or, hgt_anom_lag1)
# dmi_lag2 = SameDateAs(dmi_or, hgt_anom_lag2)
#
# n34 = SameDateAs(n34_or, hgt_anom)
# n34_lag1 = SameDateAs(n34_or, hgt_anom_lag1)
# n34_lag2 = SameDateAs(n34_or, hgt_anom_lag2)
#
# sam = SameDateAs(sam_or, hgt_anom)
# sam_lag1 = SameDateAs(sam_or, hgt_anom_lag1)
# sam_lag2 = SameDateAs(sam_or, hgt_anom_lag2)
#
# asam = SameDateAs(asam_or, hgt_anom)
# asam_lag1 = SameDateAs(asam_or, hgt_anom_lag1)
# asam_lag2 = SameDateAs(asam_or, hgt_anom_lag2)
#
# ssam = SameDateAs(ssam_or, hgt_anom)
# ssam_lag1 = SameDateAs(ssam_or, hgt_anom_lag1)
# ssam_lag2 = SameDateAs(ssam_or, hgt_anom_lag2)
#
# pp = SameDateAs(pp_or, hgt_anom)
# pp_lag1 = SameDateAs(pp_or, hgt_anom_lag1)
# pp_lag2 = SameDateAs(pp_or, hgt_anom_lag2)
#
# pp_caja = SameDateAs(pp_caja_or, hgt_anom)
# pp_caja_lag1 = SameDateAs(pp_caja_or, hgt_anom_lag1)
# pp_caja_lag2 = SameDateAs(pp_caja_or, hgt_anom_lag2)
#
# amd = hgt_anom.sel(lon=slice(210, 270), lat=slice(-80, -50)).mean(
#     ['lon', 'lat'])
# amd_lag1 = hgt_anom_lag1.sel(lon=slice(210, 270), lat=slice(-80, -50)).mean(
#     ['lon', 'lat'])
# amd_lag2 = hgt_anom_lag1.sel(lon=slice(210, 270), lat=slice(-80, -50)).mean(
#     ['lon', 'lat'])
#
# amd_sd = amd.std()
# amd = amd / amd_sd
# amd_lag1 = amd_lag1 / amd_sd
# amd_lag2 = amd_lag2 / amd_sd
#
# hgt_sd = hgt_anom.std()
# hgt_anom = hgt_anom / hgt_sd
# hgt_anom_lag1 = hgt_anom_lag1 / hgt_sd
# hgt_anom_lag2 = hgt_anom_lag2 / hgt_sd
#
# dmi = dmi / dmi.std()
# dmi_lag1 = dmi_lag1 / dmi.std()
# dmi_lag2 = dmi_lag2 / dmi.std()
#
# n34 = n34 / n34.std()
# n34_lag1 = n34_lag1 / n34.std()
# n34_lag2 = n34_lag2 / n34.std()
#
# sam = sam / sam.std()
# sam_lag1 = sam_lag1 / sam.std()
# sam_lag2 = sam_lag2 / sam.std()
#
# asam = asam / asam.std()
# asam_lag1 = asam_lag1 / asam.std()
# asam_lag2 = asam_lag2 / asam.std()
#
# ssam = ssam / ssam.std()
# ssam_lag1 = ssam_lag1 / ssam.std()
# ssam_lag2 = ssam_lag2 / ssam.std()
#
# pp_caja_sd = pp_caja.std()
# pp_caja3 = pp_caja / pp_caja_sd
# pp_caja3_lag1 = pp_caja_lag1 / pp_caja_sd
# pp_caja_lag2 = pp_caja_lag2 / pp_caja_sd
#
# pp_sd = pp.std()
# pp3 = pp / pp_sd
# pp_lag1 = pp_lag1
# pp_lag2 = pp_lag2 / pp.std()