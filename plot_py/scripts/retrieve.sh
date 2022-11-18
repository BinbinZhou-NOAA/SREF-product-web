#!/bin/ksh

TODAY=$1
cyc=$2
grid=212
data=/lfs/h1/ops/prod/com/sref/v7.1/sref.$TODAY/$cyc/ensprod
ptmp=/lfs/h2/emc/ptmp/binbin.zhou/sref/plot_py/${TODAY}${cyc}
mkdir -p $ptmp
cd $ptmp


for ens in mean spread prob ; do

bigfile=$data/sref.t${cyc}z.pgrb${grid}.${ens}_3hrly.grib2

$WGRIB2 ${bigfile} |grep ":anl:" |$WGRIB2 -i ${bigfile} -grib sref.${ens}.t${cyc}z.pgrb${grid}.grib2.f00

for t in  3 6 9 12 15 18 21 24 27 30 33 36 ; do
  if [ $t -le 9 ] ; then
   hh="0"$t
  else
   hh=$t
  fi
$WGRIB2 ${bigfile} |grep ":${t} hour fcst:" |$WGRIB2 -i ${bigfile} -grib sref.${ens}.t${cyc}z.pgrb${grid}.grib2.f${hh}
done

if [ $grid = '212' ] ; then
bigfile=$data/sref.t${cyc}z.pgrb${grid}.${ens}_1hrly.grib2
for t in 1 2 4 5 7 8 10 11 13 14 16 17 19 20 22 23 25 26 28 29 31 32 34 35 ; do
  if [ $t -le 9 ] ; then
   hh="0"$t
  else
   hh=$t
  fi
$WGRIB2 ${bigfile} |grep ":${t} hour fcst:" |$WGRIB2 -i ${bigfile} -grib sref.${ens}.t${cyc}z.pgrb${grid}.grib2.f${hh}
done
fi

done



exit


