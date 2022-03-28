# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:31:43 2022

@author: olive
"""
import os
os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Scripts") # change directory to where the files are located!

import numpy as np

#import h5py
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import rasterio
from rasterio.plot import show
import rioxarray as rxr
import geopandas as gpd
from pyproj import Transformer
from pyproj import CRS
import glob
import tifftools #For later when merging the tiff files
from collections import defaultdict
import imageio
from pathlib import Path
from osgeo import gdal,osr
from matplotlib.animation import FuncAnimation
import time
from PIL import Image
import shutil
import re
import pandas as pd
import nwp750_read_plot
import radar_data_handling
import ClimateGrid_handling
import Other_functions
from nwp750_read_plot import *
from radar_data_handling import *
from ClimateGrid_handling import *
from Other_functions import *
import pygrib
#from skimage.feature import peak_local_max
import csv
import pdb
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Rectangle
import pysteps
import pysteps.verification.spatialscores as pvs
import pysteps.verification.salscores as pvs_sal
import pysteps.feature.tstorm as pvs_tstorm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pickle
#import xagg
import collections
from scipy.interpolate import griddata
import rasterstats
from rasterstats import zonal_stats
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage.filters import uniform_filter
import xarray as xr
from osgeo import osr
from pygeoprocessing import zonal_statistics
from skimage import data, io, filters
from scipy.ndimage.measurements import center_of_mass


###########################################################################
###########################################################################
###########################################################################
os.chdir("C:/Users/olive/Desktop/Speciale/Kode") 
world_map_file = "C:/Users/olive/OneDrive/Desktop/Speciale/Kode/pygrib_functionality/pygrib_functionality/world_map_cut/world_map_background.shp" # file path to a shapefile with outline of Denmark

grid_25="C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI.shp"
grid=gpd.read_file(grid_25)
grid_25_NWP="C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI_NWP.shp"
grid_NWP=gpd.read_file(grid_25_NWP)
coor_grid="C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/DMI_grid_coor1.shp"
coor_grid_NWP="C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_NWP_coor.shp"
grid_25_coor=gpd.read_file(coor_grid)
grid_25_coor_NWP=gpd.read_file(coor_grid_NWP)
grid['ycoor']=grid_25_coor['ycoord']
grid['xcoor']=grid_25_coor['xcoord']
grid_NWP['ycoor']=grid_25_coor_NWP['ycoord']
grid_NWP['xcoor']=grid_25_coor_NWP['xcoord']

    

threshold_value=0.5
dmi_stere_crs = CRS("+proj=stere +ellps=WGS84 +lat_0=56 +lon_0=10.5666 +lat_ts=56") # raw data CRS projection
plotting_crs = 'epsg:4326' # the CRS projection you want to plot the data in

#FILES
###############################################################################
#NWP
#extract_tar('./NWP750/Data0708/dk7502021070810.tar.gz')
coordinates_nwp=(7,15,54.5,58)
Myfiles_nwp=[i for i in glob.glob("./NWP750/2021072612/*") if len(i)<24]
extracted_nwp_coor=Output_rain_NWP(Myfiles_nwp,threshold_value) 
extracted_nwp=open_not_plot(Myfiles_nwp,threshold_value,world_map_file,coordinates_nwp)

lons_nwp=extracted_nwp_coor[1]['lons']
lats_nwp=extracted_nwp_coor[1]['lats']

           
#tif_path_nwp="./Tiff_files/"+Myfiles_nwp[1][-3:]+'_rain1'+'.tif'
#data_array_to_raster_NWP(extracted_nwp[1],lons_nwp,lats_nwp,tif_path_nwp)

    ###############################################################################
#Radar
#Radar coordinates
x_radar_coords = np.arange(-421364.8 - 500, 569635.2 + 500, 500) # need to add and subtract 500 because np.arange does not include end points
y_radar_coords = np.arange(468631 + 500, -394369 - 500, -500)

#####Limit radar
radar_lons, radar_lats = project_raster_coords(x_radar_coords, y_radar_coords, dmi_stere_crs, dmi_stere_crs)
radar_lons=radar_lons[412:1500,494:1175]
radar_lats=radar_lats[412:1500,494:1175]

#movefiles("./Radar/Data0708/nowcast.dk.com.*/interpolated.*.h5","./Radar/Data0708/Interpolations") #Move interpolated files for all folders to store them in one place
Myfiles_radar=[i for i in glob.glob("./Radar/Data/Interpolations/*")]
Myfiles_radar_hr=np.array_split(Myfiles_radar[1-len(Myfiles_radar):],len(Myfiles_radar[1-len(Myfiles_radar):])/60) #The first is removed because the belongs to previous timeframe
start=time.time()
Radar_agg=aggregate_data(Myfiles_radar_hr,threshold_value)
print(time.time()-start)

tif_path_radar = Myfiles_radar_hr[0][0][-15:-5] + "_rain_radar" + ".tif" # file path for a tif file that will be generated
data_to_raster_RADAR(Radar_agg[0],radar_lons,radar_lats,tif_path_radar)
start=time.time()
zs_radar_tst=zonal_stats("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI.shp",tif_path_radar, stats="sum")
print(time.time()-start)
data_array_to_raster(Radar_agg[0],tif_path_radar)
start=time.time()
zs_radar_tst1=zonal_stats("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI.shp",tif_path_radar, stats="sum")
print(time.time()-start)

#raster_radar=rasterio.open(tif_path_radar,masked=False)
#raster_nwp=rasterio.open(tif_path_nwp,masked=False)

#for i in range(0,len(extracted_nwp)):
#    tif_path_nwp="./Tiff_files/"+Myfiles_nwp[i+1][-3:]+'_NWPtst'+'.tif'
#    data_to_raster_NWP(extracted_nwp[i+1],lons_nwp,lats_nwp,tif_path_nwp)
#start=time.time()
#zs_nwp_tst=(zonal_stats("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI_NWP.shp",tif_path_nwp, stats="sum"))
#print(time.time()-start)

#fig,ax=plt.subplots(1,1,figsize=(10,10))
#grid.plot(ax=ax,facecolor="none",edgecolor="yellow")
#show(raster_radar,ax=ax,title="radar")
#plt.show()



zs_radar,zs_nwp=produce_zonalstat(Radar_agg,radar_lons,radar_lats,Myfiles_radar_hr,extracted_nwp,lons_nwp,lats_nwp,Myfiles_nwp,grid)

radar_2d,nwp_2d=zone_to_2d(zs_radar,zs_nwp)
  



#plot_together(radar_2d[7],nwp_2d[7],grid_NWP['xcoor'].values.reshape(125,157),grid_NWP['ycoor'].values.reshape(125,157),grid_NWP['xcoor'].values.reshape(125,157),grid_NWP['ycoor'].values.reshape(125,157),world_map_file,"test_radar","test_nwp",1,1)


#np.savetxt("./25grid/Nwp+Radar/radar_2d_2607.txt",np.array(radar_2d).reshape(np.array(radar_2d).shape[0],-1))
#np.savetxt("./25grid/Nwp+radar/nwp_2d_2607.txt",np.array(nwp_2d).reshape(np.array(nwp_2d).shape[0],-1))

#Retrieve:
#load_nwp=np.loadtxt("./25grid/Nwp+Radar/nwp_2d_tst.txt")
#load_radar=np.loadtxt("./25grid/Nwp+radar/radar_2d_tst.txt")
#orig_nwp=load_nwp.reshape(load_nwp.shape[0],load_nwp.shape[1] // 157, 157)
#orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1] // 157, 157)


intensity_scale=np.logspace(0.1,1.6,10)
spatial_scale=np.logspace(2.5,1,10)

plot_spatial_threshold_matrix(intensity_scale,spatial_scale,zs_radar,zs_nwp)


#Quantiles

#fss_qt=Fractional_skillscore(zs_radar,zs_nwp,20,percentile=0.95)



#Confusion matrix
cm_day,ETS,PSS,FBI=confusion_mat(radar_2d,nwp_2d)

#SAL

loc_dict={'minref':10,'maxref':100,'minsize':2,'mindis':5,'minmax': 15}
amp=[]
struc=[]
loca=[]
for i in range(0,len(radar_2d)):
    #print(pvs_sal.sal(nwp_2d[i].transpose(1,0),radar_2d[i].transpose(1,0),thr_factor=None,thr_quantile=0.95,tstorm_kwargs=loc_dict))
    l1=pvs_sal._sal_l1_param(nwp_2d[i].transpose(1,0),radar_2d[i].transpose(1,0))
    l2=pvs_sal._sal_l2_param(nwp_2d[i].transpose(1,0),radar_2d[i].transpose(1,0),thr_factor=None,thr_quantile=None,tstorm_kwargs=loc_dict)
    amp.append(pvs_sal.sal_amplitude(nwp_2d[i].transpose(1,0),radar_2d[i].transpose(1,0)))
    struc.append(pvs_sal.sal_structure(nwp_2d[i].transpose(1,0),radar_2d[i].transpose(1,0),thr_factor=None,thr_quantile=None,tstorm_kwargs=loc_dict))
    loca.append(l1+l2)
    #print(l1,l2)

#pvs_sal._sal_detect_objects(nwp_2d[0].transpose(1,0),thr_factor=None,thr_quantile=None,tstorm_kwargs=loc_dict)
#pvs_sal._sal_detect_objects(radar_2d[0].transpose(1,0),thr_factor=None,thr_quantile=None,tstorm_kwargs=loc_dict)


plt.figure()
cm=plt.cm.get_cmap('copper')
plt.axhline(0,0,color="black",alpha=0.5)
plt.axvline(0,0,color="black",alpha=0.5)
plt.axhline(np.median(amp),0,color="red",alpha=0.5)
plt.axvline(np.median(struc),0,color="red",alpha=0.5)
sc=plt.scatter(struc,amp,c=loca,vmin=0,vmax=2,cmap=cm,zorder=2)
for i in range(0,9):
    plt.text(struc[i],amp[i],str(i))
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xlabel("Structure")
plt.ylabel("Amplitude")
cbar=plt.colorbar(sc)
cbar.ax.set_ylabel('Location', rotation=90)
plt.title("26th of July 2021")
plt.show()



plt.imshow(radar_2d[0].transpose(1,0))
plt.imshow(nwp_2d[0].transpose(1,0))
###############################################################################
##Rain gauge
#Myfiles_raingauge=[i for i in glob.glob("./Rain_gauge/Data/2021/07/26/20210726_*.txt.gz") if int(i[-11:-7])>1200 and int(i[-11:-7])<2200]
#df_raingauge=[open_gzip(i) for i in Myfiles_raingauge]
#rainobs,rainobs_lons,rainobs_lats=raingauge_obs(Myfiles_raingauge)



###############################################################################
###############################################################################

#Radar and NWP750
#for i in range(7,len(Radar_agg)):
#   plot_together(Radar_agg[i],extracted_nwp[i+1], radar_lons, radar_lats, extracted_nwp_coor[0]['lons'],extracted_nwp_coor[0]['lats'], world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[i][-1:])[-11:-9],str(Myfiles_radar_hr[i][-1:])[-13:-11],str(Myfiles_radar_hr[i][-1:])[-17:-13],str(Myfiles_radar_hr[i][-1:])[-9:-7]),
#                  "NWP \n %s/%s/%s - %s:00 + %s UTC"%(Myfiles_nwp[0][-8:-6],Myfiles_nwp[0][-10:-8],Myfiles_nwp[0][-14:-10],Myfiles_nwp[0][-6:-4],str(1+i)),Myfiles_radar_hr[i],"./Pics0807/%s.png")

#make_gif("./Pics0807/*.png",'./Pics0807/png_to_gif_hr.gif',1)


##Radar and raingauge
#for i in range(0,len(Radar_agg)):
#    plot_together(Radar_agg[i],rainobs[i], radar_lons, radar_lats, rainobs_lons,rainobs_lats, world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[i][-1:])[-11:-9],str(Myfiles_radar_hr[i][-1:])[-13:-11],str(Myfiles_radar_hr[i][-1:])[-17:-13],str(Myfiles_radar_hr[i][-1:])[-9:-7]),
#                  "Obs \n %s/%s/%s - %s:%s UTC"%(Myfiles_raingauge[0][-14:-12],Myfiles_raingauge[0][-14:-16],Myfiles_raingauge[0][-20:-16],Myfiles_raingauge[i][-11:-9],Myfiles_raingauge[i][-9:-7]),Myfiles_radar_hr[i],"./Pics/RadarRaingauge/%s.png")
#
#make_gif("./Pics/RadarRaingauge/*.png",'./Pics/RadarRaingauge/png_to_gif.gif',1)


#Is there a bias in terms of total volum of precipitation?
        

#total_precipitation(Radar_agg,extracted_nwp)



###############################################################################
###############################################################################
#Verification
#grid,precip_nwp,precip_radar=verification_area(np.nan_to_num(Radar_agg[0]),radar_lons,radar_lats,extracted_nwp_coor[1],2,15,2)       


#FSS
nwp_newdom=[[] for _ in range(len(Radar_agg))]
radar_newdom=[[] for _ in range(len(Radar_agg))]
for i in range(0,len(Radar_agg)):
    nwp_newdom[i],radar_newdom[i]=spatial_agg(extracted_nwp_coor[i+1],Radar_agg[i],radar_lats,radar_lons)
    print(i)
 
    
nwp_newdom,radar_newdom=spatial_agg(extracted_nwp_coor[1],Radar_agg[0],radar_lats,radar_lons)
file_name="7.5 common domain"

#open_file=open(file_name1,"wb")
#pickle.dump([nwp_tst,radar_tst],open_file)
#open_file.close()

open_file=open(file_name,"rb")
loaded_list=pickle.load(open_file)
open_file.close()

open_file1=open(file_name1,"rb")
loaded_list1=pickle.load(open_file1)
open_file1.close()

    
#plot squares over verification areas
plot_together(loaded_list[1][0][0],loaded_list[0][0][0], loaded_list[1][0][2], loaded_list[1][0][1], loaded_list[0][0][2],loaded_list[0][0][1], world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[0][-1:])[-11:-9],str(Myfiles_radar_hr[0][-1:])[-13:-11],str(Myfiles_radar_hr[0][-1:])[-17:-13],str(Myfiles_radar_hr[0][-1:])[-9:-7]),
                  "NWP \n %s/%s/%s - %s:00 + %s UTC"%(Myfiles_nwp[0][-8:-6],Myfiles_nwp[0][-10:-8],Myfiles_nwp[0][-14:-10],Myfiles_nwp[0][-6:-4],str(1)),Myfiles_radar_hr[0],"./Pics0807/%s.png")


FSS_random=np.repeat(np.sum((radar_grid_2d*1)>0)/np.size(radar_grid_2d),250)
FSS_uniform=0.5+FSS_random/2
plt.plot(avg_fss)
plt.plot(FSS_random)
plt.plot(FSS_uniform)
plt.xlabel("Horizontal Scale (Pixels)")
plt.ylabel("Fractional Skill Score (FSS)")
plt.legend(["0.75km","FSS random","FSS uniform"])
plt.show()
       

pvs.fss(loaded_list[0][0][0],loaded_list[1][0][0],np.quantile(np.nan_to_num(np.append(loaded_list[1][0][0],loaded_list[0][0][0])),0.95),400)

#Confusion matrix
grid,precip_nwp,precip_radar=verification_alldomain(np.nan_to_num(Radar_agg[0]),radar_lons,radar_lats,extracted_nwp_coor[1],2,30)       
cf=binary_to_confusion(precip_radar,precip_nwp)


#PI index
grid_nwp,grid_radar,precip_nwp_pi,precip_radar_pi=verification_area(np.nan_to_num(Radar_agg[1]),radar_lons,radar_lats,extracted_nwp_coor[2],2,30,2)       


binary_to_confusion(precip_radar_pi,precip_nwp_pi)
PI=precip_radar_tst[0][4][15,15]/((1/(2*15+1)**2-1)*np.sum(precip_radar_tst[0][4]-precip_radar_tst[0][4][15,15]))


#plot squares over verification areas
#plot_together(Radar_agg[0],extracted_nwp[1], radar_lons, radar_lats, extracted_nwp_coor[0]['lons'],extracted_nwp_coor[0]['lats'], world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[0][-1:])[-11:-9],str(Myfiles_radar_hr[0][-1:])[-13:-11],str(Myfiles_radar_hr[0][-1:])[-17:-13],str(Myfiles_radar_hr[0][-1:])[-9:-7]),
#                  "NWP \n %s/%s/%s - %s:00 + %s UTC"%(Myfiles_nwp[0][-8:-6],Myfiles_nwp[0][-10:-8],Myfiles_nwp[0][-14:-10],Myfiles_nwp[0][-6:-4],str(1+i)),Myfiles_radar_hr[0],"./Pics0807/%s.png")


#sq=squares_for_plot(grid_radar)
#plot_w_squares(Radar_agg[0],extracted_nwp[1], radar_lons, radar_lats, extracted_nwp_coor[0]['lons'],extracted_nwp_coor[0]['lats'], world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[0][-1:])[-11:-9],str(Myfiles_radar_hr[0][-1:])[-13:-11],str(Myfiles_radar_hr[0][-1:])[-17:-13],str(Myfiles_radar_hr[0][-1:])[-9:-7]),"NWP \n %s/%s/%s - %s:00 + %s UTC"%(Myfiles_nwp[0][-8:-6],Myfiles_nwp[0][-10:-8],Myfiles_nwp[0][-14:-10],Myfiles_nwp[0][-6:-4],str(1+i)),df_squares)
for i in range(0,len(Radar_agg)):
   grid_nwp_sq,grid_radar_sq,precip_nwp_sq,precip_radar_sq=verification_area(np.nan_to_num(Radar_agg[i]),radar_lons,radar_lats,extracted_nwp_coor[i+1],2,30,2)       
   sq=squares_for_plot(grid_radar_sq)
   plot_w_squares(Radar_agg[i],extracted_nwp[i+1], radar_lons, radar_lats, extracted_nwp_coor[0]['lons'],extracted_nwp_coor[0]['lats'], world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[0][-1:])[-11:-9],str(Myfiles_radar_hr[0][-1:])[-13:-11],str(Myfiles_radar_hr[0][-1:])[-17:-13],str(Myfiles_radar_hr[0][-1:])[-9:-7]),"NWP \n %s/%s/%s - %s:00 + %s UTC"%(Myfiles_nwp[0][-8:-6],Myfiles_nwp[0][-10:-8],Myfiles_nwp[0][-14:-10],Myfiles_nwp[0][-6:-4],str(1+i)),sq)
