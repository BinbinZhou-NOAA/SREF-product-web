#!/bin/sh


export work=/lfs/h2/emc/ptmp/binbin.zhou/sref/plot_py/YYYYMMDDCYC
export TODAY=YYYYMMDD

$WGRIB2  $work/sref.mean.tCYCz.pgrb212.grib2.fFHR -set_grib_type c3 -grib_out $work/sref.mean_c3.tCYCz.pgrb212.fFHR.grib2
$WGRIB2  $work/sref.prob.tCYCz.pgrb212.grib2.fFHR -set_grib_type c3 -grib_out $work/sref.prob_c3.tCYCz.pgrb212.fFHR.grib2

python /lfs/h2/emc/vpppg/noscrub/binbin.zhou/sref/plot_py/scripts/plot_sref_mean_prob.py YYYYMMDDCYC FHR

exit

