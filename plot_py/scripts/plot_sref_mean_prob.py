#!/bin/env/python

import grib2io
import pyproj
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib
#matplotlib.use('Agg')
import io
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as image
from matplotlib.gridspec import GridSpec
import numpy as np
import time,os,sys,multiprocessing
import multiprocessing.pool
import ncepy
from scipy import ndimage
from netCDF4 import Dataset
import cartopy

#--------------Set some classes------------------------#
# Make Python process pools non-daemonic
class NoDaemonProcess(multiprocessing.Process):
  # make 'daemon' attribute always return False
  @property
  def daemon(self):
    return False

  @daemon.setter
  def daemon(self, value):
    pass

class NoDaemonContext(type(multiprocessing.get_context())):
  Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
  def __init__(self, *args, **kwargs):
    kwargs['context'] = NoDaemonContext()
    super(MyPool, self).__init__(*args, **kwargs)


#--------------Define some functions ------------------#

def clear_plotables(ax,keep_ax_lst,fig):
  #### - step to clear off old plottables but leave the map info - ####
  if len(keep_ax_lst) == 0 :
    print("clear_plotables WARNING keep_ax_lst has length 0. Clearing ALL plottables including map info!")
  cur_ax_children = ax.get_children()[:]
  if len(cur_ax_children) > 0:
    for a in cur_ax_children:
      if a not in keep_ax_lst:
       # if the artist isn't part of the initial set up, remove it
        a.remove()

def compress_and_save(filename):
  #### - compress and save the image - ####
#  ram = io.StringIO()
  ram = io.BytesIO()
  plt.savefig(ram, format='png', bbox_inches='tight', dpi=300)
#  plt.savefig(filename, format='png', bbox_inches='tight', dpi=300)
  ram.seek(0)
  im = Image.open(ram)
  im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
  im2.save(filename, format='PNG')

def cmap_t2m():
 # Create colormap for 2-m temperature
 # Modified version of the ncl_t2m colormap from Jacob's ncepy code
    r=np.array([255,128,0,  70, 51, 0,  255,0, 0,  51, 255,255,255,255,255,171,128,128,36,162,255])
    g=np.array([0,  0,  0,  70, 102,162,255,92,128,185,255,214,153,102,0,  0,  0,  68, 36,162,255])
    b=np.array([255,128,128,255,255,255,255,0, 0,  102,0,  112,0,  0,  0,  56, 0,  68, 36,162,255])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=float(i)/(float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_t2m_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_T2M_COLTBL',colorDict)
    return cmap_t2m_coltbl


def cmap_t850():
 # Create colormap for 850-mb equivalent potential temperature
    r=np.array([255,128,0,  70, 51, 0,  0,  0, 51, 255,255,255,255,255,171,128,128,96,201])
    g=np.array([0,  0,  0,  70, 102,162,225,92,153,255,214,153,102,0,  0,  0,  68, 96,201])
    b=np.array([255,128,128,255,255,255,162,0, 102,0,  112,0,  0,  0,  56, 0,  68, 96,201])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=float(i)/(float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_t850_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_T850_COLTBL',colorDict)
    return cmap_t850_coltbl


def cmap_terra():
 # Create colormap for terrain height
 # Emerald green to light green to tan to gold to dark red to brown to light brown to white
    r=np.array([0,  152,212,188,127,119,186])
    g=np.array([128,201,208,148,34, 83, 186])
    b=np.array([64, 152,140,0,  34, 64, 186])
    xsize=np.arange(np.size(r))
    r = r/255.
    g = g/255.
    b = b/255.
    red = []
    green = []
    blue = []
    for i in range(len(xsize)):
        xNorm=float(i)/(float(np.size(r))-1.0)
        red.append([xNorm,r[i],r[i]])
        green.append([xNorm,g[i],g[i]])
        blue.append([xNorm,b[i],b[i]])
    colorDict = {"red":red, "green":green, "blue":blue}
    cmap_terra_coltbl = matplotlib.colors.LinearSegmentedColormap('CMAP_TERRA_COLTBL',colorDict)
    cmap_terra_coltbl.set_over(color='#E0EEE0')
    return cmap_terra_coltbl


def extrema(mat,mode='wrap',window=100):
    # find the indices of local extrema (max only) in the input array.
    mx = ndimage.filters.maximum_filter(mat,size=window,mode=mode)
    # (mat == mx) true if pixel is equal to the local max
    return np.nonzero(mat == mx)

#-------------------------------------------------------#

# Necessary to generate figs when not running an Xserver (e.g. via PBS)
plt.switch_backend('agg')

# Read date/time and forecast hour from command line
ymdh = str(sys.argv[1])
ymd = ymdh[0:8]
year = int(ymdh[0:4])
month = int(ymdh[4:6])
day = int(ymdh[6:8])
hour = int(ymdh[8:10])
cyc = str(hour).zfill(2)
print(year, month, day, hour)

fhr = int(sys.argv[2])
fhrm1 = fhr - 1
fhrm2 = fhr - 2
fhrm6 = fhr - 6
fhrm24 = fhr - 24
fhour = str(fhr).zfill(2)
fhour1 = str(fhrm1).zfill(2)
fhour2 = str(fhrm2).zfill(2)
fhour6 = str(fhrm6).zfill(2)
fhour24 = str(fhrm24).zfill(2)
print('fhour '+fhour)

workdir = '/lfs/h2/emc/ptmp/binbin.zhou/sref/plot_py/' + ymdh + '/'
#workdir = './'

# Define the input files
#data1 = grib2io.open('sref.mean_c3.t03z.pgrb212.grib2.f36')
data1 = grib2io.open('/lfs/h2/emc/ptmp/binbin.zhou/sref/plot_py/'+str(ymdh)+'/sref.mean_c3.t'+cyc+'z.pgrb212.f'+fhour+'.grib2')
#data2 = grib2io.open('sref.prob.t03z.pgrb212.grib2.f36')
data2 = grib2io.open('/lfs/h2/emc/ptmp/binbin.zhou/sref/plot_py/'+str(ymdh)+'/sref.prob_c3.t'+cyc+'z.pgrb212.f'+fhour+'.grib2')


msg = data1[1][0] 	# msg is a Grib2Message object

# Get the lats and lons
lats = []
lons = []
lats_shift = []
lons_shift = []

# Unshifted grid for contours and wind barbs
lat, lon = msg.latlons()
lats.append(lat)
lons.append(lon)

# Shift grid for pcolormesh
lat1 = msg.latitudeFirstGridpoint
lon1 = msg.longitudeFirstGridpoint
nx = msg.nx
ny = msg.ny
dx = msg.gridlengthXDirection
dy = msg.gridlengthYDirection
pj = pyproj.Proj(msg.projparams)
llcrnrx, llcrnry = pj(lon1,lat1)
llcrnrx = llcrnrx - (dx/2.)
llcrnry = llcrnry - (dy/2.)
x = llcrnrx + dx*np.arange(nx)
y = llcrnry + dy*np.arange(ny)
x,y = np.meshgrid(x,y)
lon, lat = pj(x, y, inverse=True)
lats_shift.append(lat)
lons_shift.append(lon)

# Unshifted lat/lon arrays grabbed directly using latlons() method
lat = lats[0]
lon = lons[0]

# Shifted lat/lon arrays for pcolormesh
lat_shift = lats_shift[0]
lon_shift = lons_shift[0]

# Forecast valid date/time
itime = ymdh
vtime = ncepy.ndate(itime,int(fhr))

# Specify plotting domains
#domains=['us','ne','se','mw','ms','nw','sw','zny','zdc','zhu','zau','zoa','sco']
domains=['us','ne','se','mw','ms','nw','sw']
#domains=['us']

###################################################
# Read in all variables and calculate differences #
###################################################
t1a = time.perf_counter()

# Visibility

ceiling = data1.select(shortName='HGT',level='cloud ceiling')[0].data() * 3.28084
visibility = data1.select(shortName='VIS')[0].data()
cloud_top = data1.select(shortName='HGT', level='cloud top')[0].data()
windshear = data1.select(shortName='VWSH', level='0 sigma level')[0].data()

ceiling_500 = data2.select(shortName='HGT', level='cloud ceiling', threshold='prob <152.5')[0].data()
ceiling_1000 = data2.select(shortName='HGT', level='cloud ceiling', threshold='prob <305')[0].data()
ceiling_2000 = data2.select(shortName='HGT', level='cloud ceiling', threshold='prob <610')[0].data()
ceiling_3000 = data2.select(shortName='HGT', level='cloud ceiling', threshold='prob <914.6')[0].data()
ceiling_4500 = data2.select(shortName='HGT', level='cloud ceiling', threshold='prob <1372')[0].data()
ceiling_6000 = data2.select(shortName='HGT', level='cloud ceiling', threshold='prob <1830')[0].data()

visibility_400 = data2.select(shortName='VIS', level='surface', threshold='prob <402')[0].data() 
visibility_800 = data2.select(shortName='VIS', level='surface', threshold='prob <804')[0].data() 
visibility_1600 = data2.select(shortName='VIS', level='surface', threshold='prob <1609')[0].data() 
visibility_3200 = data2.select(shortName='VIS', level='surface', threshold='prob <3218')[0].data() 
visibility_6400 = data2.select(shortName='VIS', level='surface', threshold='prob <4827')[0].data() 
visibility_8000 = data2.select(shortName='VIS', level='surface', threshold='prob <8046')[0].data() 
visibility_9600 = data2.select(shortName='VIS', level='surface', threshold='prob <9654')[0].data() 

LIFR = data2.select(shortName='VIS', level='cloud base', threshold='prob =1')[0].data()
IFR =  data2.select(shortName='VIS', level='cloud base', threshold='prob =2')[0].data()
MVFR = data2.select(shortName='VIS', level='cloud base', threshold='prob =3')[0].data()
VFR =  data2.select(shortName='VIS', level='cloud base', threshold='prob =4')[0].data()

fog_light = data2.select(shortName='CWORK', level='0 m above ground', threshold='prob >0.016')[0].data()
fog_med = data2.select(shortName='CWORK', level='0 m above ground', threshold='prob >0.036')[0].data()
fog_dense = data2.select(shortName='CWORK', level='0 m above ground', threshold='prob >0.103')[0].data()

#reflect_10 = data2.select(shortName='REFC', level='entire atmosphere (considered as a single layer)', threshold='prob >10')[0].data()
#reflect_20 = data2.select(shortName='REFC', level='entire atmosphere (considered as a single layer)', threshold='prob >20')[0].data()
#reflect_30 = data2.select(shortName='REFC', level='entire atmosphere (considered as a single layer)', threshold='prob >30')[0].data()
#reflect_40 = data2.select(shortName='REFC', level='entire atmosphere (considered as a single layer)', threshold='prob >40')[0].data()
#reflect_50 = data2.select(shortName='REFC', level='entire atmosphere (considered as a single layer)', threshold='prob >50')[0].data()

etop_3000 = data2.select(shortName='COVTW', level='entire atmosphere (considered as a single layer)', threshold='prob >1000')[0].data()
etop_9000 = data2.select(shortName='COVTW', level='entire atmosphere (considered as a single layer)', threshold='prob >3000')[0].data()
etop_15000 = data2.select(shortName='COVTW', level='entire atmosphere (considered as a single layer)', threshold='prob >5000')[0].data()
etop_21000 = data2.select(shortName='COVTW', level='entire atmosphere (considered as a single layer)', threshold='prob >7000')[0].data()
etop_30000 = data2.select(shortName='COVTW', level='entire atmosphere (considered as a single layer)', threshold='prob >10000')[0].data()

cloud_clear = data2.select(shortName='TCDC', level='entire atmosphere (considered as a single layer)', threshold='prob >=0 <20')[0].data()
cloud_scatr = data2.select(shortName='TCDC', level='entire atmosphere (considered as a single layer)', threshold='prob >=20 <50')[0].data()
cloud_brokn = data2.select(shortName='TCDC', level='entire atmosphere (considered as a single layer)', threshold='prob >=50 <80')[0].data()
cloud_overc = data2.select(shortName='TCDC', level='entire atmosphere (considered as a single layer)', threshold='prob >=80 <100')[0].data()

llws_20 = data2.select(shortName='VWSH', level='0 sigma level', threshold='prob >20')[0].data()

wind10m_20 = data2.select(shortName='WIND', level='10 m above ground', threshold='prob >12.89')[0].data()
wind10m_40 = data2.select(shortName='WIND', level='10 m above ground', threshold='prob >17.5')[0].data()
wind10m_60 = data2.select(shortName='WIND', level='10 m above ground', threshold='prob >25.78')[0].data()

type_rain = data2.select(shortName='CRAIN', level='surface', threshold='prob =1')[0].data()
type_snow = data2.select(shortName='CSNOW', level='surface', threshold='prob =1')[0].data()
type_frzr = data2.select(shortName='CFRZR', level='surface', threshold='prob =1')[0].data()

icing_FL030 = data2.select(shortName='TIPD', level='900 mb', threshold='prob =1')[0].data()
icing_FL060 = data2.select(shortName='TIPD', level='800 mb', threshold='prob =1')[0].data()
icing_FL090 = data2.select(shortName='TIPD', level='725 mb', threshold='prob =1')[0].data()
icing_FL120 = data2.select(shortName='TIPD', level='650 mb', threshold='prob =1')[0].data()
icing_FL150 = data2.select(shortName='TIPD', level='575 mb', threshold='prob =1')[0].data()
icing_FL180 = data2.select(shortName='TIPD', level='500 mb', threshold='prob =1')[0].data()
icing_FL240 = data2.select(shortName='TIPD', level='400 mb', threshold='prob =1')[0].data()


cat_light_FL180  = data2.select(shortName='TPFI', level='500 mb', threshold='prob =1')[0].data()
cat_med_FL180    = data2.select(shortName='TPFI', level='500 mb', threshold='prob =2')[0].data()
cat_severe_FL180 = data2.select(shortName='TPFI', level='500 mb', threshold='prob =3')[0].data()

cat_light_FL210  = data2.select(shortName='TPFI', level='450 mb', threshold='prob =1')[0].data()
cat_med_FL210    = data2.select(shortName='TPFI', level='450 mb', threshold='prob =2')[0].data()
cat_severe_FL210 = data2.select(shortName='TPFI', level='450 mb', threshold='prob =3')[0].data()

cat_light_FL240  = data2.select(shortName='TPFI', level='400 mb', threshold='prob =1')[0].data()
cat_med_FL240    = data2.select(shortName='TPFI', level='400 mb', threshold='prob =2')[0].data()
cat_severe_FL240 = data2.select(shortName='TPFI', level='400 mb', threshold='prob =3')[0].data()

cat_light_FL270  = data2.select(shortName='TPFI', level='350 mb', threshold='prob =1')[0].data()
cat_med_FL270    = data2.select(shortName='TPFI', level='350 mb', threshold='prob =2')[0].data()
cat_severe_FL270 = data2.select(shortName='TPFI', level='350 mb', threshold='prob =3')[0].data()

cat_light_FL300  = data2.select(shortName='TPFI', level='300 mb', threshold='prob =1')[0].data()
cat_med_FL300    = data2.select(shortName='TPFI', level='300 mb', threshold='prob =2')[0].data()
cat_severe_FL300 = data2.select(shortName='TPFI', level='300 mb', threshold='prob =3')[0].data()

cat_light_FL330  = data2.select(shortName='TPFI', level='275 mb', threshold='prob =1')[0].data()
cat_med_FL330    = data2.select(shortName='TPFI', level='275 mb', threshold='prob =2')[0].data()
cat_severe_FL330 = data2.select(shortName='TPFI', level='275 mb', threshold='prob =3')[0].data()

cat_light_FL360  = data2.select(shortName='TPFI', level='225 mb', threshold='prob =1')[0].data()
cat_med_FL360    = data2.select(shortName='TPFI', level='225 mb', threshold='prob =2')[0].data()
cat_severe_FL360 = data2.select(shortName='TPFI', level='225 mb', threshold='prob =3')[0].data()

cat_light_FL390  = data2.select(shortName='TPFI', level='200 mb', threshold='prob =1')[0].data()
cat_med_FL390    = data2.select(shortName='TPFI', level='200 mb', threshold='prob =2')[0].data()
cat_severe_FL390 = data2.select(shortName='TPFI', level='200 mb', threshold='prob =3')[0].data()

t2a = time.perf_counter()
t3a = round(t2a-t1a, 3)
print(("%.3f seconds to read all messages") % t3a)

# colors for difference plots, only need to define once
difcolors = ['blue','#1874CD','dodgerblue','deepskyblue','turquoise','white','white','#EEEE00','#EEC900','darkorange','orangered','red']
difcolors2 = ['white']
difcolors3 = ['blue','dodgerblue','turquoise','white','white','#EEEE00','darkorange','red']

########################################
#    START PLOTTING FOR EACH DOMAIN    #
########################################

def main():

  # Number of processes must coincide with the number of domains to plot
#  pool = multiprocessing.Pool(len(domains))
  pool = MyPool(len(domains))
  pool.map(plot_all,domains)

def plot_all(domain):

  global dom
  dom = domain
  print(('Working on '+dom))

  #global fig,axes,ax1,keep_ax_lst_1,x,y,xextent,yextent,im,par,transform
  #fig,axes,ax1,keep_ax_lst_1,x,y,xextent,yextent,im,par,transform = create_figure()

  global fig,axes,ax1,ax2,keep_ax_lst_1,keep_ax_lst_2,x,y,xextent,yextent,im,par,transform
  fig,axes,ax1,ax2,keep_ax_lst_1,keep_ax_lst_2,x,y,xextent,yextent,im,par,transform = create_figure()


  # Split plots into 2 sets with multiprocessing
  sets = [1,2]
  pool2 = multiprocessing.Pool(len(sets))
  pool2.map(plot_sets,sets)

def create_figure():

  # Map corners for each domain
  if dom == 'us':
    llcrnrlon = -130.0
    llcrnrlat = 20.0 
    urcrnrlon = -62.0
    urcrnrlat = 50.0
    cen_lat = 35.4
    cen_lon = -97.6
    xextent=-2200000
    yextent=-675000
  elif dom == 'ne':
    llcrnrlon = -86.0
    llcrnrlat = 36.0
    urcrnrlon = -62.0
    urcrnrlat = 50.0
    cen_lat = 41.0
    cen_lon = -74.6
    xextent=-175000
    yextent=-282791
  elif dom == 'se':
    llcrnrlon = -95.0
    llcrnrlat = 22.0
    urcrnrlon = -70
    urcrnrlat = 40.0
    cen_lat = 30.0
    cen_lon = -80.0
    xextent=-12438
    yextent=-448648
  elif dom == 'mw':
    llcrnrlon = -105.0
    llcrnrlat = 36.0
    urcrnrlon = -83.0
    urcrnrlat = 50.0
    cen_lat = 42.0
    cen_lon = -93.0
    xextent=-230258
    yextent=-316762
  elif dom == 'ms':
    llcrnrlon = -107.0
    llcrnrlat = 25.0
    urcrnrlon = -85.0
    urcrnrlat = 40.0
    cen_lat = 33.0
    cen_lon = -95.0
    xextent=-529631
    yextent=-407090
  elif dom == 'nw':
    llcrnrlon = -130.0
    llcrnrlat = 38.0
    urcrnrlon = -105.5
    urcrnrlat = 50.0
    cen_lat = 44.0
    cen_lon = -115.0
    xextent=-540000
    yextent=-333623
  elif dom == 'sw':
    llcrnrlon = -130.0
    llcrnrlat = 30.0
    urcrnrlon = -105.0
    urcrnrlat = 42.0
    cen_lat = 36.0
    cen_lon = -115.0
    xextent=-593059
    yextent=-377213
  elif dom == 'zny':
    llcrnrlon = -82.0     
    llcrnrlat = 38.5
    urcrnrlon = -71.0
    urcrnrlat = 43.5
    cen_lat = 41.0
    cen_lon = -77.0
    xextent=112182
    yextent=-99031
  elif dom == 'zdc':
    llcrnrlon = -83.0     
    llcrnrlat = 33.0
    urcrnrlon = -70.0
    urcrnrlat = 41.0
    cen_lat = 37.0
    cen_lon = -77.0
    xextent=112182
    yextent=-99031
  elif dom == 'zhu':
    llcrnrlon = -105.0 
    llcrnrlat = 25.0
    urcrnrlon = -83.0
    urcrnrlat = 36.0
    cen_lat = 30.0
    cen_lon = -94.0
    xextent=-224751
    yextent=-238851
  elif dom == 'zau':
    llcrnrlon = -94.0 
    llcrnrlat = 40.0
    urcrnrlon = -85.0
    urcrnrlat = 45.0
    cen_lat = 43.0
    cen_lon = -90.0 
    xextent=-224751
    yextent=-238851
  elif dom == 'zoa':
    llcrnrlon = -130.0 
    llcrnrlat = 35.0
    urcrnrlon = -115.0
    urcrnrlat = 41.0
    cen_lat = 37.0
    cen_lon = -125.0
    xextent=-227169
    yextent=-200000
  elif dom == 'sco':
    llcrnrlon = -111.0 
    llcrnrlat = 36.5
    urcrnrlon = -100.0
    urcrnrlat = 42.0
    cen_lat = 38.0
    cen_lon = -106.0
    xextent=-224751
    yextent=-238851

  # create figure and axes instances
  fig = plt.figure(figsize=(4,4))
  gs = GridSpec(4,4,wspace=0.0,hspace=0.0)
  im = image.imread('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python.rrfs/noaa.png')
  par = 1

  # Define where Cartopy maps are located
  cartopy.config['data_dir'] = '/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth'

  back_res='50m'
  back_img='on'

  # set up the map background with cartopy
  if dom == 'us':
    extent = [llcrnrlon-1,urcrnrlon-6,llcrnrlat,urcrnrlat+1]
  else:
    extent = [llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat]

  myproj=ccrs.LambertConformal(central_longitude=cen_lon, central_latitude=cen_lat,
          false_easting=0.0,false_northing=0.0, secant_latitudes=None, 
          standard_parallels=None,globe=None)

# myproj=ccrs.PlateCarree(central_longitude=0.0,globe=None)
# Note PlateCarree seems running time is extremely slow

  ax1 = fig.add_subplot(gs[0:4,0:4], projection=myproj)
  ax2 = fig.add_subplot(gs[0:4,0:4], projection=myproj)
  ax1.set_extent(extent)
  ax2.set_extent(extent)
  axes = [ax1,ax2] 

  fline_wd = 0.2  # line width
  fline_wd_lakes = 0.2  # line width
  falpha = 0.5    # transparency

  # natural_earth
#  land=cfeature.NaturalEarthFeature('physical','land',back_res,
#                    edgecolor='face',facecolor=cfeature.COLORS['land'],
#                    alpha=falpha)
  lakes=cfeature.NaturalEarthFeature('physical','lakes',back_res,
                    edgecolor='black',facecolor='none',
                    linewidth=fline_wd_lakes,alpha=falpha)
  coastlines=cfeature.NaturalEarthFeature('physical','coastline',
                    back_res,edgecolor='black',facecolor='none',
                    linewidth=fline_wd,alpha=falpha)
  states=cfeature.NaturalEarthFeature('cultural','admin_1_states_provinces',
                    back_res,edgecolor='black',facecolor='none',
                    linewidth=fline_wd,alpha=falpha)
  borders=cfeature.NaturalEarthFeature('cultural','admin_0_countries',
                    back_res,edgecolor='black',facecolor='none',
                    linewidth=fline_wd,alpha=falpha)

  # All lat lons are earth relative, so setup the associated projection correct for that data
  transform = ccrs.PlateCarree()

  # high-resolution background images
  if back_img=='on':
    img = plt.imread('/lfs/h2/emc/lam/noscrub/Benjamin.Blake/python/NaturalEarth/raster_files/NE1_50M_SR_W.tif')
    ax1.imshow(img, origin='upper', transform=transform)
    ax2.imshow(img, origin='upper', transform=transform)

#  ax1.add_feature(land)
  ax1.add_feature(lakes)
  ax1.add_feature(states)
#  ax1.add_feature(borders)
  ax1.add_feature(coastlines)

  ax2.add_feature(lakes)
  ax2.add_feature(states)
  ax2.add_feature(coastlines)


  # Map/figure has been set up here, save axes instances for use again later
  keep_ax_lst_1 = ax1.get_children()[:]
  keep_ax_lst_2 = ax1.get_children()[:]

  return fig,axes,ax1,ax2,keep_ax_lst_1,keep_ax_lst_2,x,y,xextent,yextent,im,par,transform
#  return fig,axes,ax1,keep_ax_lst_1,x,y,xextent,yextent,im,par,transform


def plot_sets(set):
# Add print to see if dom is being passed in
  print(('plot_sets dom variable '+dom))

  global fig,axes,ax1,ax2,keep_ax_lst_1,keep_ax_lst_2,x,y,xextent,yextent,im,par,transform

  if set == 1:
    plot_set_1()
  elif set == 2:
    plot_set_2()

def plot_set_1():
  global fig,axes,ax1,keep_ax_lst_1,x,y,xextent,yextent,im,par,transform


#################################
  # Plot ensemble mean
#################################

# 1/4. Ceiling mean

  print(('Working on ceiling mean  for '+dom))

  units = 'feet AGL'
  clevs = [0,1000,3000,6000,10000,15000,20000,25000]
  colorlist = ['deeppink','orange','yellow','lawngreen','cyan','blue','darkmagenta']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,ceiling,norm=norm,transform=transform,cmap=cm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.5,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=4)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=4)
  ax1.text(.5,1.03,'SREF Ceiling  Mean ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save(workdir+dom+'/ceiling.t'+cyc+'z.'+fhour+'.png')

# 2/4 Visibility mean

  print(('Working on ceiling mean  for '+dom))

  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'm'
  clevs = [200,500,1000,2000,4000,6000,10000]
  colorlist = ['deeppink','orange','yellow','lawngreen','cyan','blue','darkmagenta']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,visibility,norm=norm,transform=transform,cmap=cm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.5,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'SREF Visibility Mean ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save(workdir+dom+'/visb.t'+cyc+'z.'+fhour+'.png')

# 3/4 Iwnd shear  mean

  print(('Working on wind shear  mean  for '+dom))

  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'knots/2000 feet'
  clevs = [5,10,15,20,25,30,35]
  colorlist = ['deeppink','orange','yellow','lawngreen','cyan','blue','darkmagenta']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,windshear,norm=norm,transform=transform,cmap=cm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.5,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'SREF Low Level Wind Shear Mean ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save(workdir+dom+'/llws.t'+cyc+'z.'+fhour+'.png')


# 4/4 cloud top mean

  print(('Working on cloud top mean  for '+dom))

  cbar1.remove()
  clear_plotables(ax1,keep_ax_lst_1,fig)

  units = 'feet'
  clevs = [0,1500,3000,6000,9000,12000,15000]
  colorlist = ['deeppink','orange','yellow','lawngreen','cyan','blue','darkmagenta']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  xmin, xmax = ax1.get_xlim()
  ymin, ymax = ax1.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax1.pcolormesh(lon_shift,lat_shift,cloud_top,norm=norm,transform=transform,cmap=cm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')
  cbar1 = plt.colorbar(cs_1,ax=ax1,orientation='horizontal',pad=0.01,shrink=0.5,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=5)
  ax1.text(.5,1.03,'SREF Cloud Top Mean ('+units+') \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax1.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax1.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)

  compress_and_save(workdir+dom+'/cldtop.t'+cyc+'z.'+fhour+'.png')

  plt.clf()


def plot_set_2():
  global fig,axes,ax1,ax2,keep_ax_lst_1,keep_ax_lst_2,x,y,xextent,yextent,im,par,transfor

#################################
  # Plot ensemble probability plots
#################################

# 1. plot ceiling prob

  units = ''
  clevs = [0,10,20,30,40,50,60,70,80,90,100]
  colorlist = ['white','purple','blue','deepskyblue','cyan','yellowgreen','yellow','orange','orangered','deeppink','red']
  cm = matplotlib.colors.ListedColormap(colorlist)
  norm = matplotlib.colors.BoundaryNorm(clevs, cm.N)

  xmin, xmax = ax2.get_xlim()
  ymin, ymax = ax2.get_ylim()
  xmax = int(round(xmax))
  ymax = int(round(ymax))

  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,ceiling_500,norm=norm,transform=transform,cmap=cm)
  cs_1.cmap.set_under('white')
  cs_1.cmap.set_over('white')

  cbar1 = plt.colorbar(cs_1,ax=ax2,orientation='horizontal',pad=0.01,shrink=0.5,ticks=clevs,extend='max')
  cbar1.set_label(units,fontsize=6)
  cbar1.ax.set_xticklabels(clevs)
  cbar1.ax.tick_params(labelsize=5)

  print(('1~1/6: Working on ceiling < 500 feet prob for '+dom))
  ax2.text(.5,1.03,'SREF Probability of Ceiling < 500 feet  \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_ceiling.lt.500ft.t'+cyc+'z.'+fhour+'.png')


  print(('1~2/6: Working on ceiling < 1000 feet prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,ceiling_1000,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Ceiling < 1000 feet  \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_ceiling.lt.1000ft.t'+cyc+'z.'+fhour+'.png')

  print(('1~3/6: Working on ceiling < 2000 feet prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,ceiling_2000,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Ceiling < 2000 feet  \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_ceiling.lt.2000ft.t'+cyc+'z.'+fhour+'.png')

  print(('1~4/6: Working on ceiling < 3000 feet prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,ceiling_3000,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Ceiling < 3000 feet \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_ceiling.lt.3000ft.t'+cyc+'z.'+fhour+'.png')

  print(('1~5/6: Working on ceiling < 4500 feet prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,ceiling_4500,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Ceiling < 4500 feet \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_ceiling.lt.4500ft.t'+cyc+'z.'+fhour+'.png')

  print(('1~6/6: Working on ceiling < 6000 feet prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,ceiling_6000,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Ceiling < 6000 feet  \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_ceiling.lt.6000ft.t'+cyc+'z.'+fhour+'.png')


# 2. plot visibility prob

  print(('2~1/7: Working on visibility < 400 m prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,visibility_400,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Visibility < 1/4 mile  \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_vis.lt.400m.t'+cyc+'z.'+fhour+'.png')

  print(('2~2/7: Working on visibility < 800 m prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,visibility_800,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Visibility < 1/2 mile  \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_vis.lt.800m.t'+cyc+'z.'+fhour+'.png')


  print(('2~3/7: Working on visibility < 1600 m prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,visibility_1600,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Visibility < 1 mile  \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_vis.lt.1600m.t'+cyc+'z.'+fhour+'.png')


  print(('2~4/7: Working on visibility < 3200 m prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,visibility_3200,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Visibility < 2 miles \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_vis.lt.3200m.t'+cyc+'z.'+fhour+'.png')


  print(('2~5/7: Working on visibility < 6400 m prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,visibility_6400,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Visibility < 4 miles \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_vis.lt.6400m.t'+cyc+'z.'+fhour+'.png')

  print(('2~6/7: Working on visibility < 8000 m prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,visibility_8000,norm=norm,transform=transform,cmap=cm)  
  ax2.text(.5,1.03,'SREF Probability of Visibility < 5 miles \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_vis.lt.8000m.t'+cyc+'z.'+fhour+'.png')

  print(('2~7/7: Working on visibility < 9600 m prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,visibility_9600,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Visibility < 6 miles \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_vis.lt.9600m.t'+cyc+'z.'+fhour+'.png')


# 3. Flight Restriction prob

  print(('3~1/4: Working on LIFR prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,LIFR,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of LIFR \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_LIFR.t'+cyc+'z.'+fhour+'.png')

  print(('3~2/4: Working on IFR prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,IFR,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of IFR \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize =6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2')) 
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_IFR.t'+cyc+'z.'+fhour+'.png')

  print(('3~3/4: Working on MVFR prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,MVFR,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of MVFR \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_MVFR.t'+cyc+'z.'+fhour+'.png')

  print(('3~4/4: Working on VFR prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,VFR,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of VFR \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2')) 
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_VFR.t'+cyc+'z.'+fhour+'.png')


# 4. Fog probability
  print(('4~1/3: Working on light fog prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,fog_light,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Light Fog \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_fog.light.t'+cyc+'z.'+fhour+'.png')


  print(('4~2/3: Working on med fog prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,fog_med,norm=norm,transform=transform,cmap=cm)  
  ax2.text(.5,1.03,'SREF Probability of Moderate Fog \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_fog.mid.t'+cyc+'z.'+fhour+'.png')


  print(('4~3/3: Working on dense fog prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,fog_dense,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Dense Fog \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_fog.dense.t'+cyc+'z.'+fhour+'.png')

# 5. Low level wind shear probability

  print(('5~1/1: Working on LLWS prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,llws_20,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Low Level Wind Shear > 20knots \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_llws.gt.20knt.t'+cyc+'z.'+fhour+'.png')

#6 10m Wind speed probability
  
  print(('6~1/3: Working on Wind10m > 20kn for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,wind10m_20,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of 10m Wind Speed > 20knots \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)  
  compress_and_save(workdir+dom+'/prb_jet10.gt.20knt.t'+cyc+'z.'+fhour+'.png')

  print(('6~2/3: Working on Wind10m > 40kn for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,wind10m_40,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of 10m Wind Speed > 40knots \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)  
  compress_and_save(workdir+dom+'/prb_jet10.gt.40knt.t'+cyc+'z.'+fhour+'.png')

  print(('6~3/3: Working on Wind10m > 60kn for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,wind10m_60,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of 10m Wind Speed > 60knots \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)  
  compress_and_save(workdir+dom+'/prb_jet10.gt.60knt.t'+cyc+'z.'+fhour+'.png')

# 7. Cloud sky coverage prob
  print(('7~1/4: Working on Clear Sky prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cloud_clear,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Clear Sky \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_sky1.t'+cyc+'z.'+fhour+'.png')

  print(('7~2/4: Working on Scattered Sky prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cloud_scatr,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Scatted Sky \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_sky2.t'+cyc+'z.'+fhour+'.png')

  print(('7~3/4: Working on Clear Sky prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cloud_brokn,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Broken Sky \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_sky3.t'+cyc+'z.'+fhour+'.png')

  print(('7~4/4: Working on Overcast Sky prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cloud_overc,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Overcast Sky \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_sky4.t'+cyc+'z.'+fhour+'.png')

# 8 Echo top prob
  print(('8~1/5: Working on echo top > 3000 ft prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,etop_3000,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Echo Top > 3000 feet \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_etop.gt.3000feet.t'+cyc+'z.'+fhour+'.png')

  print(('8~2/5: Working on echo top > 9000 ft prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,etop_9000,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Echo Top > 9000 feet \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_etop.gt.9000feet.t'+cyc+'z.'+fhour+'.png')

  print(('8~3/5: Working on echo top > 15000 ft prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,etop_15000,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Echo Top > 15000 feet \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_etop.gt.15000feet.t'+cyc+'z.'+fhour+'.png')

  print(('8~4/5: Working on echo top > 21000 ft prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,etop_21000,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Echo Top > 21000 feet \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_etop.gt.21000feet.t'+cyc+'z.'+fhour+'.png')

  print(('8~5/5: Working on echo top > 30000 ft prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,etop_30000,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Echo Top > 30000 feet \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_etop.gt.30000feet.t'+cyc+'z.'+fhour+'.png')


# 9. Precip type prob

  print(('9~1/3: Working on rain type prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,type_rain,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Rain Type                    \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)  
  compress_and_save(workdir+dom+'/prb_rain.t'+cyc+'z.'+fhour+'.png')

  print(('9~2/3: Working on snow type prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,type_snow,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Snow Type                    \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)  
  compress_and_save(workdir+dom+'/prb_snow.t'+cyc+'z.'+fhour+'.png')

  print(('9~3/3: Working on freezing rain type prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,type_frzr,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Freezing Rain Type                    \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)  
  compress_and_save(workdir+dom+'/prb_frzr.t'+cyc+'z.'+fhour+'.png')

# 11. Icing prob

  print(('11~2/8: Working on icing on FL030 prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,icing_FL030,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Icing at FL030                           \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_icing.at.FL030.t'+cyc+'z.'+fhour+'.png')

  print(('11~3/8: Working on icing on FL060 prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,icing_FL060,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Icing at FL060                           \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_icing.at.FL060.t'+cyc+'z.'+fhour+'.png')

  print(('11~4/8: Working on icing on FL090 prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,icing_FL090,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Icing at FL090                           \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_icing.at.FL090.t'+cyc+'z.'+fhour+'.png')

  print(('11~5/8: Working on icing on FL120 prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,icing_FL120,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Icing at FL120                           \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_icing.at.FL120.t'+cyc+'z.'+fhour+'.png')

  print(('11~6/8: Working on icing on FL150 prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,icing_FL150,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Icing at FL150                           \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_icing.at.FL150.t'+cyc+'z.'+fhour+'.png')

  print(('11~7/8: Working on icing on FL180 prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,icing_FL180,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Icing at FL180                           \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_icing.at.FL180.t'+cyc+'z.'+fhour+'.png')

  print(('11~8/8: Working on icing on FL240 prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,icing_FL240,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of Icing at FL240                           \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_icing.at.FL240.t'+cyc+'z.'+fhour+'.png')

# 12. Light CAT probability

  print(('12~1/9: Working on cat_light  on FL180  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_light_FL180,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Light CAT at FL180                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_1cat8.FL240.t'+cyc+'z.'+fhour+'.png')

  print(('12~2/9: Working on cat_light  on FL210  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_light_FL210,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Light CAT at FL210                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_1cat8.FL240.t'+cyc+'z.'+fhour+'.png')

  print(('12~3/9: Working on cat_light  on FL240  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_light_FL240,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Light CAT at FL240                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_1cat8.FL240.t'+cyc+'z.'+fhour+'.png')

  print(('12~4/9: Working on cat_light  on FL270  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_light_FL270,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Light CAT at FL270                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_1cat8.FL270.t'+cyc+'z.'+fhour+'.png')

  print(('12~5/9: Working on cat_light  on FL300  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_light_FL300,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Light CAT at FL300                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_1cat8.FL300.t'+cyc+'z.'+fhour+'.png')

  print(('12~6/9: Working on cat_light  on FL330  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_light_FL330,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Light CAT at FL330                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_1cat8.FL330.t'+cyc+'z.'+fhour+'.png')

  print(('12~7/9: Working on cat_light  on FL360  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_light_FL360,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Light CAT at FL360                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_1cat8.FL360.t'+cyc+'z.'+fhour+'.png')

  print(('12~8/9: Working on cat_light  on FL390  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_light_FL390,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Light CAT at FL390                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_1cat8.FL390.t'+cyc+'z.'+fhour+'.png')


# 13.  Moderate CAT probability

  print(('13~1/9: Working on cat_med  on FL180  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_med_FL180,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Moderate CAT at FL180                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_2cat8.FL240.t'+cyc+'z.'+fhour+'.png')

  print(('13~2/9: Working on cat_med  on FL210  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_med_FL210,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Moderate CAT at FL210                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_2cat8.FL240.t'+cyc+'z.'+fhour+'.png')

  print(('13~3/9: Working on cat_med  on FL240  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_med_FL240,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Moderate CAT at FL240                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_2cat8.FL240.t'+cyc+'z.'+fhour+'.png')

  print(('13~4/9: Working on cat_med  on FL270  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_med_FL270,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Moderate CAT at FL270                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_2cat8.FL270.t'+cyc+'z.'+fhour+'.png')

  print(('13~5/9: Working on cat_med  on FL300  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_med_FL300,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Moderate CAT at FL300                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_2cat8.FL300.t'+cyc+'z.'+fhour+'.png')

  print(('13~6/9: Working on cat_med  on FL330  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_med_FL330,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Moderate CAT at FL330                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_2cat8.FL330.t'+cyc+'z.'+fhour+'.png')

  print(('13~7/9: Working on cat_med  on FL360  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_med_FL360,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Moderate CAT at FL360                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_2cat8.FL360.t'+cyc+'z.'+fhour+'.png')

  print(('13~8/9: Working on cat_med  on FL390  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_med_FL390,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Moderate CAT at FL390                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_2cat8.FL390.t'+cyc+'z.'+fhour+'.png')

# 14. Severe CAT probability

  print(('14~1/9: Working on cat_severe  on FL180  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_severe_FL180,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Severe CAT at FL180                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')',horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_3cat8.FL240.t'+cyc+'z.'+fhour+'.png')

  print(('14~2/9: Working on cat_severe  on FL210  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_severe_FL210,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Severe CAT at FL210                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_3cat8.FL240.t'+cyc+'z.'+fhour+'.png')

  print(('14~3/9: Working on cat_severe  on FL240  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_severe_FL240,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Severe CAT at FL240                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_3cat8.FL240.t'+cyc+'z.'+fhour+'.png')

  print(('14~4/9: Working on cat_severe  on FL270  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_severe_FL270,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Severe CAT at FL270                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_3cat8.FL270.t'+cyc+'z.'+fhour+'.png')

  print(('14~5/9: Working on cat_severe  on FL300  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_severe_FL300,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Severe CAT at FL300                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_3cat8.FL300.t'+cyc+'z.'+fhour+'.png')

  print(('14~6/9: Working on cat_severe  on FL330  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_severe_FL330,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Severe CAT at FL330                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_3cat8.FL330.t'+cyc+'z.'+fhour+'.png')

  print(('14~7/9: Working on cat_severe  on FL360  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_severe_FL360,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Severe CAT at FL360                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_3cat8.FL360.t'+cyc+'z.'+fhour+'.png')

  print(('14~8/9: Working on cat_severe  on FL390  prob for '+dom))
  cs_1 = ax2.pcolormesh(lon_shift,lat_shift,cat_severe_FL390,norm=norm,transform=transform,cmap=cm)
  ax2.text(.5,1.03,'SREF Probability of of Severe CAT at FL390                     \n initialized: '+itime+' valid: '+vtime + ' (f'+fhour+')'
,horizontalalignment='center',fontsize=6,transform=ax2.transAxes,bbox=dict(facecolor='white',alpha=0.85,boxstyle='square,pad=0.2'))
  ax2.imshow(im,aspect='equal',alpha=0.5,origin='upper',extent=(xmin,xextent,ymin,yextent),zorder=4)
  compress_and_save(workdir+dom+'/prb_3cat8.FL390.t'+cyc+'z.'+fhour+'.png')


#####################################
  plt.clf()
#####################################

  
################################################################################

main()

