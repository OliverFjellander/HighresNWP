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
import pysteps.verification.probscores as pvs_prob
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
start_start=time.time()

###########################################################################
###########################################################################
###########################################################################
os.chdir("C:/Users/olive/Desktop/Speciale/Kode") 
world_map_file = "C:/Users/olive/OneDrive/Desktop/Speciale/Kode/pygrib_functionality/pygrib_functionality/world_map_cut/world_map_background.shp" # file path to a shapefile with outline of Denmark


threshold_value=0.5
dmi_stere_crs = CRS("+proj=stere +ellps=WGS84 +lat_0=56 +lon_0=10.5666 +lat_ts=56") # raw data CRS projection
plotting_crs = 'epsg:4326' # the CRS projection you want to plot the data in
nwp_crs = 'epsg:25832' # the CRS projection you want to plot the data in


#FILES
###############################################################################
#2.5km common grid
#grid_NWP=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI_NWP.shp")

#coor_grid="C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/DMI_grid_coor1.shp"
#coor_grid_NWP="C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_NWP_coor.shp"

#grid_25_coor=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/DMI_grid_coor1.shp")
#grid_25_coor_NWP=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_NWP_coor.shp")
#grid_NWP['ycoor']=grid_25_coor_NWP['ycoord']
#grid_NWP['xcoor']=grid_25_coor_NWP['xcoord']

#grid_small=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_northernzealand.shp")
#grid_small_coor=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/coor_nothernzealand.shp")
#grid_small_nwp=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_northernzealand_nwp.shp")
#grid_small_nwp['ycoor']=grid_small_coor['ycoord']
#grid_small_nwp['xcoor']=grid_small_coor['xcoord']

#grid_coor=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_op.shp") #In original rotated lat/lon
#grid=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/25gridnwp.shp") #In original rotated lat/lon

#grid_radar=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/25gridradar.shp")
#date_time=str(2021072612)
#date_time=str(2021080708)
month="05"

file_radar=glob.glob("./Radar/2021-%s/*"%month)
file_short=[file_radar[s][-15:-7] for s in range(0,len(file_radar))]

remove_dup=[item for item, count in collections.Counter(file_short).items() if count > 1439]
date_times=[]
for i in range(0,len(remove_dup)):
    for j in range(0,24,1):
        if len(str(j))==1:
            hour=str(0)+str(j)
        else:
            hour=str(j)
        date_times.append(remove_dup[i]+hour)

#NWP
#extract_tar('./NWP750/Data0708/dk7502021070810.tar.gz')
# coordinates_nwp=(7,15,54.5,58)
# Myfiles_nwp=[i for i in glob.glob("./NWP750/%s/*"%date_time) if len(i)<24]
# extracted_nwp_coor=Output_rain_NWP(Myfiles_nwp,threshold_value) 
# extracted_nwp=open_not_plot(Myfiles_nwp,threshold_value,world_map_file,coordinates_nwp)

# lons_nwp=extracted_nwp_coor[1]['lons']
# lats_nwp=extracted_nwp_coor[1]['lats']

#data_to_raster_NWP(np.repeat(1,np.size(lons_nwp)).reshape(630,700),lons_nwp,lats_nwp,"./nwp_area.tif")        
   

###############################################################################
#Radar
#Radar coordinates
x_radar_coords = np.arange(-421364.8 - 500, 569635.2 + 500, 500) # need to add and subtract 500 because np.arange does not include end points
y_radar_coords = np.arange(468631 + 500, -394369 - 500, -500)

#####Limit radar
radar_lons, radar_lats = project_raster_coords(x_radar_coords, y_radar_coords, dmi_stere_crs, dmi_stere_crs)
radar_lons_4326,radar_lats_4326=project_raster_coords(x_radar_coords, y_radar_coords, dmi_stere_crs, plotting_crs)

#data_to_raster_RADAR(np.repeat(1,np.size(radar_lats_4326)).reshape(1728,1984),radar_lons_4326,radar_lats_4326,"./radar_area.tif")

radar_lons=radar_lons[412:1500,494:1175]
radar_lats=radar_lats[412:1500,494:1175]
radar_lons_4326=radar_lons_4326[412:1500,494:1175]
radar_lats_4326=radar_lats_4326[412:1500,494:1175]


#movefiles("./Radar/Data0708/nowcast.dk.com.*/interpolated.*.h5","./Radar/Data0708/Interpolations") #Move interpolated files for all folders to store them in one place
#Myfiles_radar=[i for i in glob.glob("./Radar/Data%s/Interpolations/*"%date_time[-6:-2])]
Myfiles_radar=[i for i in glob.glob("./Radar/2021-%s/*.%s*"%(month,date_times[0]))]
Myfiles_radar_hr=np.array_split(Myfiles_radar[1-len(Myfiles_radar):],len(Myfiles_radar[1-len(Myfiles_radar):])/60) #The first is removed because the belongs to previous timeframe
print("before radar aggregation takes:", time.time()-start_start)
start=time.time()
Radar_agg=aggregate_data(Myfiles_radar_hr,threshold_value)
print("Radar aggregations takes:", time.time()-start)

#start_radar=time.time()
#radar_tst=aggregate_data_tst(Myfiles_radar_hr,threshold_value)
#print(time.time()-start_radar)

#tif_path_radar = Myfiles_radar_hr[0][0][-15:-5] + "_rain_radar" + ".tif" # file path for a tif file that will be generated
#data_to_raster_RADAR(Radar_agg[0],radar_lons,radar_lats,tif_path_radar)

save_name=str(Myfiles_radar_hr[0][0])[-15:-5]
#raster_radar=rasterio.open(tif_path_radar,masked=False)

#for i in range(0,len(extracted_nwp)):
#    tif_path_nwp="./Tiff_files/"+Myfiles_nwp[i+1][-3:]+'_NWPtst'+'.tif'
#    data_to_raster_NWP(extracted_nwp[i+1],lons_nwp,lats_nwp,tif_path_nwp)

#fig,ax=plt.subplots(1,1,figsize=(10,10))
#grid.plot(ax=ax,facecolor="none",edgecolor="yellow")
#show(raster_radar,ax=ax,title="radar")
#plt.show()

###############################################################################
##Rain gauge
# Myfiles_raingauge=[i for i in glob.glob("./Rain_gauge/Data/2021/%s/%s/%s_*.txt.gz"%(date_time[-6:-4],date_time[-4:-2],date_time[:-2])) if int(i[-11:-7])>int(date_time[-2:])*100 and int(i[-11:-7])<(int(date_time[-2:])*100+1000)]
# df_raingauge=[open_gzip(i) for i in Myfiles_raingauge]
# #rainobs,rainobs_lons,rainobs_lats=raingauge_obs(Myfiles_raingauge)
# rainobs,rainobs_lons, rainobs_lats = raingauge_obs_orig(Myfiles_raingauge)

# zs_raingauge=produce_zonalstat_rg(rainobs,rainobs_lons,rainobs_lats,Myfiles_raingauge)
# rg_2d=zone_to_2d_raingauge(zs_raingauge)

# #data_to_raster_rg(rainobs[0], rainobs_lons, rainobs_lats, "test_raingauge09.tif")

# raingauge_plot(rainobs[0],rainobs_lons,rainobs_lats,world_map_file,"test",Myfiles_raingauge)


# raingauge_plot(rg_2d[0],grid_NWP['xcoor'].values.reshape(125,157),grid_NWP['ycoor'].values.reshape(125,157),world_map_file,"test",Myfiles_raingauge)

# ###############################################################################
###############################################################################

#Radar and NWP750
# for i in range(0,len(Radar_agg)):
#    plot_together(Radar_agg[i],extracted_nwp[i+1], radar_lons_4326, radar_lats_4326, extracted_nwp_coor[0]['lons'],extracted_nwp_coor[0]['lats'], world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[i][-1:])[-11:-9],str(Myfiles_radar_hr[i][-1:])[-13:-11],str(Myfiles_radar_hr[i][-1:])[-17:-13],str(Myfiles_radar_hr[i][-1:])[-9:-7]),
#                   "NWP \n %s/%s/%s - %s:00 + %s UTC"%(Myfiles_nwp[0][-8:-6],Myfiles_nwp[0][-10:-8],Myfiles_nwp[0][-14:-10],Myfiles_nwp[0][-6:-4],str(1+i)),Myfiles_radar_hr[i],"./Pics0807/%s.png")

# #make_gif("./Pics%s/*.png"%save_name[-6:-2],'./Pics%s/png_to_gif_hr.gif'%date_time[-6:-2],1)


# #Radar and raingauge
# for i in range(7,8):
#     plot_together(Radar_agg[i],rainobs[i], radar_lons, radar_lats, rainobs_lons,rainobs_lats, world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[i][-1:])[-11:-9],str(Myfiles_radar_hr[i][-1:])[-13:-11],str(Myfiles_radar_hr[i][-1:])[-17:-13],str(Myfiles_radar_hr[i][-1:])[-9:-7]),
#                   "Obs \n %s/%s/%s - %s:%s UTC"%(Myfiles_raingauge[0][-14:-12],Myfiles_raingauge[0][-14:-16],Myfiles_raingauge[0][-20:-16],Myfiles_raingauge[i][-11:-9],Myfiles_raingauge[i][-9:-7]),Myfiles_radar_hr[i],"./Pics/RadarRaingauge/%s.png")

#make_gif("./Pics/RadarRaingauge/*.png",'./Pics/RadarRaingauge/png_to_gif.gif',1)


##################################################################################
################################################################################

#Is there a bias in terms of total volum of precipitation?
        

#total_precipitation(Radar_agg,extracted_nwp)



###############################################################################
###############################################################################
#Verification

start_zonal=time.time()
#zs_nwp=produce_zonalstat(extracted_nwp,lons_nwp,lats_nwp,Myfiles_nwp, "NWP")
zs_radar=produce_zonalstat(Radar_agg,radar_lons,radar_lats,Myfiles_radar_hr,"radar")
print("Zonal stat takes:", time.time()-start_zonal)

radar_2d=zone_to_2d(zs_radar)


#plot_together(radar_2d[7],nwp_2d[7],grid_NWP['xcoor'].values.reshape(125,157),grid_NWP['ycoor'].values.reshape(125,157),grid_NWP['xcoor'].values.reshape(125,157),grid_NWP['ycoor'].values.reshape(125,157),world_map_file,"test_radar","test_nwp",1,1)


np.savetxt("./25grid/Nwp+Radar/radar_2d_%s.txt"%(save_name[:10]),np.array(radar_2d).reshape(np.array(radar_2d).shape[0],-1))
#np.savetxt("./25grid/Nwp+radar/nwp_2d_%s.txt"%(save_name[:10]),np.array(nwp_2d).reshape(np.array(nwp_2d).shape[0],-1))
print("Time in total:", time.time()-start_start)
#Retrieve:
#load_nwp=np.loadtxt("./25grid/Nwp+Radar/nwp_2d_2607.txt")
#load_radar=np.loadtxt("./25grid/Nwp+radar/radar_2d_2607.txt")
#orig_nwp=load_nwp.reshape(load_nwp.shape[0],load_nwp.shape[1] // 157, 157)
#orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1] // 157, 157)
#radar_plot(orig_radar[0],np.array(grid_25_coor_NWP['xcoord']).reshape(125,157),np.array(grid_25_coor_NWP['ycoord']).reshape(125,157),world_map_file,"Radar \n 26/07/2021 - 16:00 UTC",1)
#radar_plot(Radar_agg[0],radar_lons_4326,radar_lats_4326,world_map_file,"Radar \n 26/07/2021 - 16:00 UTC",1)


#intensity_scale=np.logspace(0.1,1.6,9)
#spatial_scale=np.logspace(2.5,1,10)
#plot_spatial_threshold_matrix(intensity_scale,spatial_scale,zs_radar,zs_nwp)


#Quantiles

#fss_qt=Fractional_skillscore(zs_radar,zs_nwp,297,percentile=0.95,plot=True)
#fss=Fractional_skillscore(zs_radar,zs_nwp,296,intensity=10,plot=True)

#Confusion matrix
cm_day,ETS,PSS,FBI,TPR,TS,FPR=confusion_mat(rg_2d,nwp_2d)
#pvs_prob.ROC_curve(zs_nwp,zs_raingauge,1)

#SAL

struc,amp,loca=produce_SAL(radar_2d,nwp_2d)
print("Structure: %s"%struc)
print("Amplitude: %s"%amp)
print("Location: %s"%loca)

plot_SAL(struc,amp,loca,save_name)


#########################TEST
loc_dict={'minref':3.0,'maxref':100,'minsize':1,'mindis':3,'minmax': 4,'mindiff':2}
pvs_sal._sal_l2_param(nwp_2d[4].transpose(1,0),radar_2d[4].transpose(1,0),thr_factor=None,thr_quantile=None,tstorm_kwargs=loc_dict)

l2_tst=[]
for i in range(0,len(radar_2d)):
    max_dis_tst=np.sqrt(((radar_2d[i].transpose(1,0).shape[0]) ** 2) + ((radar_2d[i].transpose(1,0).shape[1]) ** 2))
    obs_r=pvs_sal._sal_weighted_distance(nwp_2d[i].transpose(1,0),thr_factor=None,thr_quantile=None,tstorm_kwargs=loc_dict)/(np.nanmean(nwp_2d[i]))
    forc_r=pvs_sal._sal_weighted_distance(radar_2d[i].transpose(1,0),thr_factor=None,thr_quantile=None,tstorm_kwargs=loc_dict)/(np.nanmean(radar_2d[i]))
    print(2 * ((abs(obs_r - forc_r)) / max_dis_tst))
    l2_tst.append(2 * ((abs(obs_r - forc_r)) / max_dis_tst))
    
l_sum=[np.nansum((x,y)) for x,y in zip(l1,l2_tst)]
l_sum=[np.nan if x==0 else x for x in l_sum]

plot_SAL(struc,amp,loca,save_name)
##############################

print("In total:", time.time()-start_start)

fig,ax=plt.subplots(3,3,sharex=True,sharey=True,figsize=(10,10))
fig.tight_layout(pad=1.0)
fig.legend(["Radar","NWP"],loc="upper right")
k=0
for i in range(0,3):
    for j in range(0,3):
        ax[i,j].imshow(radar_2d[k].transpose(1,0),cmap='winter',alpha=0.5)
        ax[i,j].imshow(nwp_2d[k].transpose(1,0),cmap='autumn',alpha=0.5)
        ax[i,j].title.set_text("Lead time: %i"%k)
        k=k+1
        fig.text(5,5,"legend",bbox={'facecolor':'white'})
        plt.legend(["Radar","NWP"],loc="upper right")
        plt.show()
    