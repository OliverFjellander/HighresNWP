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
#start_start=time.time()

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
grid_tst=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/25grid_25832.shp")
grid_25832=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/Centroids_25832.shp")
coor_4326=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/coor_4326.shp")
grid_4326=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/25grid_4326.shp")

date_time=str(2021072612)
#date_time=str(2021080708)



#Is there a bias in terms of total volum of precipitation?
        
#total_precipitation(Radar_agg,extracted_nwp)



###############################################################################
###############################################################################
#Subjective Verification

lons_25=np.array(coor_4326['xcoor']).reshape((156,184))
lats_25=np.array(coor_4326['ycoor']).reshape((156,184))

#Retrieve:
cloudburst_days=np.genfromtxt("./cloudburstdays.txt",dtype='str')
severe_days=np.genfromtxt("./severedays.txt",dtype='str')

def precipitation_days(days_found):
    dates=[]
    for i in range(0,len(days_found)):
        dates.append(int(days_found[i,0][2:4]+days_found[i,0][5:7]+days_found[i,0][8:10]+days_found[i,1][:2]))
    dates=np.unique(dates)

    nwp_files=glob.glob("./25grid/NWP750/*")
    nwp_int=[]
    for i in range(0,len(nwp_files)):
        nwp_int.append(int(nwp_files[i][-12:-4]))

    final_dates=[]
    for i in range(0,len(dates)):
        temp=dates[i]-nwp_int
        temp[temp<=0]=1000
        if any(temp[temp<7]):
            final_dates.append(nwp_int[np.argmin(temp)])
    return final_dates

cloudburst_dates=precipitation_days(cloudburst_days)
severe_dates=precipitation_days(severe_days)

def produce_gifs_samegrid(dates):
    os.chdir("C:/Users/olive/Desktop/Speciale/Kode") 
    for i in range(0,len(np.unique(dates))):
        date=str(np.unique(dates)[i])
        try:
            load_radar=np.loadtxt("./25grid/Radar/radar_20%s.txt"%date[:-2])
        except FileNotFoundError:
            continue

        load_NEA=np.loadtxt("./25grid/NEA/nea_2d_%s.txt"%date)
        load_nwp=np.loadtxt("./25grid/NWP750/nwp_2d_%s.txt"%date)
        load_radar=np.loadtxt("./25grid/Radar/radar_20%s.txt"%date[:-2])

        #orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1] // 184, 184)
        orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1])
        orig_nwp=load_nwp.reshape(load_nwp.shape[0],load_nwp.shape[1] // 184, 184)
        #orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1] // 184, 184)
        orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1])
        
        intensity_scale=[1,2.5,5,10,25]
        spatial_scale=[1,2.5,5,10,25,50,100,200]
        plot_spatial_threshold_matrix(intensity_scale,spatial_scale,orig_radar[12:18],orig_nea)

produce_gifs_samegrid(cloudburst_dates)

#intensity_scale=np.logspace(0.1,1.6,9)
#spatial_scale=np.logspace(2.5,1,10)
plot_spatial_threshold_matrix(intensity_scale,spatial_scale,zs_radar,zs_nwp)

#Quantiles

#fss_qt=Fractional_skillscore(zs_radar,zs_nwp,297,percentile=0.95,plot=True)
#fss=Fractional_skillscore(zs_radar,zs_nwp,296,intensity=10,plot=True)

#Confusion matrix
cm_day,ETS,PSS,FBI,TPR,TS,FPR,hit,miss,tot=confusion_mat(rg_2d,nwp_2d)

plt.plot(hit)
plt.plot(miss)

plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1))
plt.scatter(FPR,TPR)
plt.xlim([0,1])
plt.ylim([0,1])

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
    