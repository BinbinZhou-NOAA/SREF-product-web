#!/bin/ksh

s=/lfs/h2/emc/vpppg/noscrub/binbin.zhou/sref/plot_py/web_grib2
cd $s/base

TODAY=$1
PAST=`$NDATE -24 ${TODAY}12`
PSTDY=`echo ${PAST} | cut -c 1-8`

cyc=$2
GRID=212
begin=$cyc
typeset -Z2 cyc
typeset -Z2 begin
cycles=""
while [ $begin -ge 0 ]  
do 
  if [ $begin -lt $cyc ] ; then
    cycles="$cycles  <option value=${begin} > ${begin}Z"
  else
    cycles="$cycles  <option value=${begin} selected > ${begin}Z"
  fi 
  begin=`expr $begin - 6 `
done     

echo ${cycles}

if [ $GRID = '212' ] ; then
   sub="<option value=us selected > Entire <option value=ne > N.East <option value=se > S. East <option value=mw > Mid-West <option value=ms > South <option value=nw > N. West <option value=sw > S. West"
   
   sed -e "s!YYYYMMDD!$TODAY!g" -e "s!PASTDAY1!$PSTDY!g"  -e "s!CYCLES!${cycles}!g" -e "s!IMAGE!.t${cyc}z.01.png!g" -e "s!> CONUS!selected > CONUS!g" -e "s!SUBS!${sub}!g"  $s/base/html.base > $s/html/sref.html.base

elif [ $GRID = '216' ] ; then  
   sub="<option value=alaska  selected > Entire" 
   sed -e "s!sref.js!sref_3hr.js!g" -e "s!YYYYMMDD!$TODAY!g" -e "s!PASTDAY1!$PSTDY!g" -e "s!CYCLES!${cycles}!g" -e "s!IMAGE!.t${cyc}z.01.png!g" -e "s!> Alaska!selected > Alaska!g" -e "s!SUBS!${sub}!g"  $s/base/html.base > $s/html/sref.html.base
else
   sub="<option value=hawaii  selected > Entire"
   sed -e "s!sref.js!sref_3hr.js!g" -e "s!YYYYMMDD!$TODAY!g" -e "s!PASTDAY1!$PSTDY!g" -e "s!CYCLES!${cycles}!g" -e "s!IMAGE!.t${cyc}z.01.png!g" -e "s!> Hawaii!selected > Hawaii!g" -e "s!SUBS!${sub}!g"  $s/base/html.base > $s/html/sref.html.base
fi 

sed -e "s! > Icing! selected > Icing!g" -e "s!PRODS!<option value=prb_icing.at.FL240 > Prob: FL240 <option value=prb_icing.at.FL180 > Prob: FL180 <option value=prb_icing.at.FL150 > Prob: FL150 <option value=prb_icing.at.FL120 > Prob: FL120 <option value=prb_icing.at.FL090 > Prob: FL090 <option value=prb_icing.at.FL060 > Prob: FL060 <option value=prb_icing.at.FL030 > Prob: FL030 !g" $s/html/sref.html.base > $s/html/icing.html

sed -e "s! > Turbulence! selected > Turbulence!g" -e "s!PRODS! <option value=prb_3cat8.FL390 > Prob: severe FL390 <option value=prb_3cat8.FL360 > Prob: severe FL360 <option value=prb_3cat8.FL330 > Prob: severe FL330 <option value=prb_3cat8.FL300 > Prob: severe FL300 <option value=prb_3cat8.FL270 > Prob: severe FL270 <option value=prb_3cat8.FL240 > Prob: severe FL240 <option value=prb_3cat8.FL210 > Prob: severe FL210 <option value=prb_3cat8.FL180 > Prob: severe FL180  <option value=prb_2cat8.FL390 > Prob: moderate FL390 <option value=prb_2cat8.FL360 > Prob: moderate FL360 <option value=prb_2cat8.FL330 > Prob: moderate FL330 <option value=prb_2cat8.FL300 > Prob: moderate FL300 <option value=prb_2cat8.FL270 > Prob: moderate FL270 <option value=prb_2cat8.FL240 > Prob: moderate FL240 <option value=prb_2cat8.FL210 > Prob: moderate FL210 <option value=prb_2cat8.FL180 > Prob: moderate FL180  <option value=prb_1cat8.FL390 > Prob: light FL390 <option value=prb_1cat8.FL360 > Prob: light FL360 <option value=prb_1cat8.FL330 > Prob: light FL330 <option value=prb_1cat8.FL300 > Prob: light FL300 <option value=prb_1cat8.FL270 > Prob: light FL270 <option value=prb_1cat8.FL240 > Prob: light FL240 <option value=prb_1cat8.FL210 > Prob: light FL210 <option value=prb_1cat8.FL180 > Prob: light FL180 !g" $s/html/sref.html.base > $s/html/cat.html

sed -e "s! > Convection! selected > Convection!g"  $s/html/sref.html.base > $s/html/conv.html 

sed -e "s! > Ceiling! selected  > Ceiling!g" -e "s!PRODS!<option value=ceiling> Ceiling mean <option value=prb_ceiling.lt.500ft >Prob: Ceiling<500 ft <option value=prb_ceiling.lt.1000ft >Prob: Ceiling<1000 ft <option value=prb_ceiling.lt.2000ft > Prob: Ceiling<2000 ft <option value=prb_ceiling.lt.3000ft > Prob: Ceiling<3000 ft <option value=prb_ceiling.lt.4500ft >Prob: Ceiling<4500 ft  <option value=prb_ceiling.lt.6000ft > Prob: Ceiling<6000 ft !g"   $s/html/sref.html.base > $s/html/ceiling.html

sed -e "s! > Visibility! selected > Visibility!g"  -e "s!PRODS!<option value=visb > Visibility mean <option value=prb_vis.lt.400m > Prob: Vis < 1/4 mile <option value=prb_vis.lt.800m > Prob: Vis < 1/2 mile <option value=prb_vis.lt.1600m > Prob: Vis < 1 mile <option value=prb_vis.lt.3200m > Prob: Vis < 2 miles <option value=prb_vis.lt.6400m > Prob: Vis < 4 miles <option value=prb_vis.lt.8000m > Prob: Vis < 5 miles <option value=prb_vis.lt.9600m > Prob: Vis < 6 miles !g"  $s/html/sref.html.base > $s/html/visib.html

sed -e "s! > Low level wind shear! selected > Low level wind shear!g" -e "s!PRODS!<option value=llws>LLWS mean/spread <option value=prb_llws.gt.20knt > Prob: LLWS>20knt/2000ft<option value=prb_jet10.gt.20knt > Prob: > 20 knt at Surface <option value=prb_jet10.gt.40knt > Prob: > 40 knt at Surface <option value=prb_jet10.gt.60knt > Prob: > 60 knt at Surface   !g" $s/html/sref.html.base > $s/html/llws.html

#sed -e "s! > Jet Stream! selected > Jet Stream!g" -e "s!PRODS!<option value=prb_jet30000.gt.60knt > Prob: > 60 knt at 30000ft <option value=prb_jet30000.gt.80knt > Prob: > 80 knt at 30000ft <option value=prb_jet30000.gt.100knt > Prob: > 100 knt at 30000ft <option value=prb_jet15000.gt.60knt > Prob: > 60 knt at 15000ft <option value=prb_jet15000.gt.80knt > Prob: > 80 knt at 15000ft <option value=prb_jet15000.gt.100knt > Prob: > 100 knt at 15000ft   <option value=prb_jet4500.gt.20knt > Prob: > 20 knt at 4500ft <option value=prb_jet4500.gt.40knt > Prob: > 40 knt at 4500ft <option value=prb_jet4500.gt.60knt > Prob: > 60 knt at 4500ft <option value=prb_jet10.gt.20knt > Prob: > 20 knt at Surface <option value=prb_jet10.gt.40knt > Prob: > 40 knt at Surface <option value=prb_jet10.gt.60knt > Prob: > 60 knt at Surface!g" $s/html/sref.html.base > $s/html/wind.html


sed -e "s! > Reflectivity! selected > Reflectivity!g" -e "s!PRODS!<option value=prb_comprefl.gt.10 > Prob: Reflectivity>10 dBZ <option value=prb_comprefl.gt.20 > Prob: Reflectivity>20 dBZ <option value=prb_comprefl.gt.30 > Prob: Reflectivity>30 dBZ <option value=prb_comprefl.gt.40 > Prob: Reflectivity>40 dBZ!g"  $s/html/sref.html.base > $s/html/refl.html 

sed -e "s! > Echo-top! selected > Echo-top!g" -e "s!PRODS!<option value=prb_etop.gt.3000feet > Prob: Echo-top>3000 ft <option value=prb_etop.gt.9000feet > Prob: Echo-top>9000 ft <option value=prb_etop.gt.15000feet > Prob: Echo-top>15000 ft  <option value=prb_etop.gt.21000feet > Prob: Echo-top>21000 ft <option value=prb_etop.gt.30000feet > Prob: Echo-top>30000 ft!g" $s/html/sref.html.base > $s/html/etop.html 

#Sed -e "s! > Tropopause! selected > Tropopause!g" -e "s!PRODS!<option value=trop > Tropopause height mean/spread!g" $s/html/sref.html.base > $s/html/trop.html

#sed -e "s! > Freezing! selected > Freezing!g" -e "s!PRODS!<option value=frzh > Freezing height mean/spread!g"  $s/html/sref.html.base > $s/html/frzh.html

sed -e "s! > Fog! selected > Fog!g" -e "s!PRODS!<option value=prb_fog.light > Prob: Occurrence  <option value=prb_fog.mid > Prob: medium <option value=prb_fog.dense > Prob: dense !g"  $s/html/sref.html.base > $s/html/fog.html


sed -e "s! > Flight Restriction! selected > Flight Restriction!g" -e "s!PRODS!<option value=prb_LIFR> Probability of LIFR <option value=prb_IFR> Probability of IFR <option value=prb_MVFR> Probability of MVFR <option value=prb_VFR> Probability of VFR !g"  $s/html/sref.html.base > $s/html/fltrestr.html   

#sed -e "s! > Cloud! selected > Cloud!g" -e "s!PRODS!<option value=cldtot > Total cloud mean/spread <option value=cldtop > Cloud top mean/spread <option value=prb_sky1 > Prob: clear sky <option value=prb_sky2 > Prob: scattered sky <option value=prb_sky3 > Prob: broken sky <option value=prb_sky4 > Prob: overcast sky <option value=prb_ctop.gt.3000feet> Prob: cloud top > 3000 ft <option value=prb_ctop.gt.6000feet> Prob: cloud top > 6000 ft <option value=prb_ctop.gt.9000feet> Prob: cloud top > 9000 ft <option value=prb_ctop.gt.15000feet> Prob: cloud top > 15000 ft <option value=prb_ctop.gt.30000feet> Prob: cloud top > 30000 ft!g" $s/html/sref.html.base > $s/html/cloud.html  

sed -e "s! > Cloud! selected > Cloud!g" -e "s!PRODS!<option value=cldtop > Cloud top mean  <option value=prb_sky1 > Prob: clear sky <option value=prb_sky2 > Prob: scattered sky <option value=prb_sky3 > Prob: broken sky <option value=prb_sky4 > Prob: overcast sky !g" $s/html/sref.html.base > $s/html/cloud.html  


sed -e "s! > Precip! selected > Precip!g" -e "s!PRODS!<option value=prb_rain > Prob: rain <option value=prb_snow > Prob: snow <option value=prb_frzr > Prob: Freezing rain !g" $s/html/sref.html.base > $s/html/prcp.html



sftp  wd20bz@emcrzdm << EOF
prompt
cd /home/www/emc/htdocs/mmb/wd20bz/SREF_aviation/web_site/html_$GRID
lcd /lfs/h2/emc/vpppg/noscrub/binbin.zhou/sref/plot_py/web_grib2/html
mput *.html
close
EOF

exit


