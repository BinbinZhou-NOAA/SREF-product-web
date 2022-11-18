#!/bin/ksh

TODAY=`date +%Y%m%d`
CYC=$1

. /u/Binbin.Zhou/.kshrc


dev=`cat /etc/dev`
if [ $dev = 'mars' ] ; then

t=`date +%H`
if [ $CYC -eq 21 ]  ; then #run at evening and  machine time has become next day
x=` /gpfs/dell1/nco/ops/nwprod/prod_util.v1.1.0/exec/ips/ndate -24 ${TODAY}01`
TODAY=`echo ${x} | cut -c 1-8`
fi


GRID=$2



/gpfs/dell2/emc/verification/noscrub/Binbin.Zhou/sref/web_grib2/script/get_ctl_gs_run.sh $TODAY $CYC $GRID
/gpfs/dell2/emc/verification/noscrub/Binbin.Zhou/sref/web_grib2/base/get_html_days.sh $TODAY $CYC $GRID 

if [ $cyc -eq 3 ] ; then
PAST=` /gpfs/dell1/nco/ops/nwprod/prod_util.v1.1.0/exec/ips/ndate -24 ${TODAY}12`
PASTDAY=`echo ${PAST} | cut -c 1-8`
ssh -l wd20bz emcrzdm "rm -rf /home/www/emc/htdocs/mmb/wd20bz/SREF_aviation/$PASTDAY"
fi

fi 
