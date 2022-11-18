#!/bin/bash -l

set -x 

TODAY=`$NDATE`
yyyymmdd=${TODAY:0:8}
cyc=$1

PAST=`$NDATE -48 ${yyyymmdd}01`
PASTDAY=${PAST:0:8}
#ssh -l wd20bz emcrzdm "rm -rf /home/www/emc/htdocs/mmb/SREF_avia/FCST/NARRE/$PASTDAY"


export script=/lfs/h2/emc/vpppg/noscrub/binbin.zhou/sref/plot_py/scripts
export dev=/lfs/h2/emc/vpppg/noscrub/binbin.zhou/sref/plot_py/dev
export work=/lfs/h2/emc/ptmp/binbin.zhou/sref/plot_py/${yyyymmdd}${cyc}
mkdir -p $work
mkdir -p $work/ms
mkdir -p $work/mw
mkdir -p $work/ne
mkdir -p $work/nw
mkdir -p $work/se
mkdir -p $work/sw
mkdir -p $work/us

$script/retrieve.sh ${yyyymmdd} ${cyc}

for fhr in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 ; do
 
 sed -e "s!YYYYMMDD!$yyyymmdd!g" -e "s!CYC!$cyc!g" -e "s!FHR!$fhr!g" $dev/sref_plot_py.ecf.base > $dev/sref_plot_py.t${cyc}z.f${fhr}.ecf
 sed -e "s!YYYYMMDD!$yyyymmdd!g" -e "s!CYC!$cyc!g" -e "s!FHR!$fhr!g" $script/run_sref_plot_py.FHR.sh > $work/run_sref_plot_py_${fhr}.sh
 
 chmod +x $work/run_sref_plot_py_${fhr}.sh

 qsub $dev/sref_plot_py.t${cyc}z.f${fhr}.ecf

done

sleep 2400 

/lfs/h2/emc/vpppg/noscrub/binbin.zhou/sref/plot_py/scripts/sftp.sh $yyyymmdd $cyc
/lfs/h2/emc/vpppg/noscrub/binbin.zhou/sref/plot_py/web_grib2/base/get_html_days.sh $yyyymmdd $cyc 

