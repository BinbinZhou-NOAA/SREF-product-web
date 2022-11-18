wgrib2=$WGRIB2
#$wgrib2 sref.mean.t03z.pgrb212.grib2.f36 -set_grib_type c3 -grib_out sref.mean_c3.t03z.pgrb212.grib2.f36
#$wgrib2 sref.prob.t03z.pgrb212.grib2.f36 -set_grib_type c3 -grib_out sref.prob_c3.t03z.pgrb212.grib2.f36

python ./plot_sref_mean_prob.py 2022060303 36
