#!/bin/sh

TODAY=$1
CYC=$2

export work=/lfs/h2/emc/ptmp/binbin.zhou/sref/plot_py/${TODAY}${CYC}

LOCATION=SREF_aviation
GRID=212

for reg in us ne se mw ms nw sw ; do


sftp wd20bz@vm-lnx-emcrzdm01.ncep.noaa.gov<<EOF

mkdir /home/www/emc/htdocs/mmb/wd20bz/$LOCATION/$TODAY
cd  /home/www/emc/htdocs/mmb/wd20bz/$LOCATION/$TODAY
mkdir /home/www/emc/htdocs/mmb/wd20bz/$LOCATION/$TODAY/$CYC
cd  /home/www/emc/htdocs/mmb/wd20bz/$LOCATION/$TODAY/$CYC
mkdir /home/www/emc/htdocs/mmb/wd20bz/$LOCATION/$TODAY/$CYC/$GRID
cd /home/www/emc/htdocs/mmb/wd20bz/$LOCATION/$TODAY/$CYC/$GRID
mkdir /home/www/emc/htdocs/mmb/wd20bz/$LOCATION/$TODAY/$CYC/$GRID/${reg}
cd /home/www/emc/htdocs/mmb/wd20bz/$LOCATION/$TODAY/$CYC/$GRID/${reg}

lcd ${work}/${reg}
mput *.png

EOF

done

