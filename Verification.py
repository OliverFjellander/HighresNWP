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
from matplotlib.ticker import MultipleLocator, ScalarFormatter
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
extreme_days=np.genfromtxt("./extremerain.txt",dtype='str')
rainy_days=np.genfromtxt("./rainydays.txt",dtype='str')

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
extreme_dates=precipitation_days(extreme_days)
rainy_dates=precipitation_days(rainy_days)
rainy_dates=[x for x in rainy_dates if x not in extreme_dates]

def produce_FSS_matrix(dates,model,threshold):
    os.chdir("C:/Users/olive/Desktop/Speciale/Kode") 
    fss_collect=[[] for _ in range(len(np.unique(dates)))]
    random_collect=[[] for _ in range(len(np.unique(dates)))]
    for i in range(0,len(np.unique(dates))):
        date=str(np.unique(dates)[i])
        try:
            load_radar=np.loadtxt("./25grid/Radar/radar_20%s.txt"%date[:-2])
        except FileNotFoundError:
            continue

        load_NEA=np.loadtxt("./25grid/NEA/nea_2d_%s.txt"%date)
        load_nwp=np.loadtxt("./25grid/NWP750/nwp_2d_%s.txt"%date)
        load_radar=np.loadtxt("./25grid/Radar/radar_20%s.txt"%date[:-2])
        #print(date)
        
        #orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1] // 184, 184)
        orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1])
        #orig_nwp=load_nwp.reshape(load_nwp.shape[0],load_nwp.shape[1] // 184, 184)
        orig_nwp=load_nwp.reshape(load_nwp.shape[0],load_nwp.shape[1])
        #orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1] // 184, 184)
        orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1])
        spatial_scale=[2.5,5,10,20,40,80,160,320]
        if threshold=="Precipitation":
            intensity_scale=[0.5,2.5,4.5,6.5,8.5,10.5,12.5,14.5]
        elif threshold=="Extreme Precipitation":
            intensity_scale=[2.5,5,7.5,10,12.5,15,17.5,20]
        elif threshold=="Cloudbursts":
            intensity_scale=[2.5,5,7.5,10,12.5,15,17.5,20]
        elif threshold=="Severe Rain":
            intensity_scale=[2.5,5,7.5,10,12.5,15,17.5,20]
        else:
            return("Threshold needs to be either Precipitation or Extreme Precipitation")
        if model=="NEA":
            fss_collect[i],random_collect[i]=plot_spatial_threshold_matrix(intensity_scale,spatial_scale,orig_radar[int(date[-2:]):int(date[-2:])+6],orig_nea,orig_nwp)
        elif model=="DK750":
            fss_collect[i],random_collect[i]=plot_spatial_threshold_matrix(intensity_scale,spatial_scale,orig_radar[int(date[-2:]):int(date[-2:])+6],orig_nwp,orig_nea)
        else:
            return("Model needs to be either NEA or DK750")
    fss_collect=[x for x in fss_collect if x!=[]]
    random_collect=[x for x in random_collect if x!=[]]
    fss_array=np.stack(fss_collect, axis=-1) 
    #fss_min=np.nanmin(fss_array,axis=-1)
    #fss_max=np.nanmax(fss_array,axis=-1)
    fss_array=np.nanmean(fss_array,axis=-1)
    random_array=[np.mean(k) for k in zip(*random_collect)]
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    cmap = colors.ListedColormap(["darkred","firebrick","orangered","orange","gold","yellow","greenyellow","limegreen","forestgreen","green"])
    boundaries = [0.0, 0.1,0.2,0.3, 0.4,0.5, 0.6,0.7, 0.8, 0.9, 1.0]
    norm=colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.set_yscale('log')
    plt.pcolor(intensity_scale, spatial_scale, fss_array, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    #ax.grid(True, axis='both', linestyle='-', color='k')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Fractional Skill Score (FSS)', rotation=90)
    for i in range(0,len(intensity_scale)):
        for j in range(0,len(spatial_scale)):
            if fss_array[j,i]>0.1:    
                if fss_array[j,i]>=(0.5+random_array[i]/2):
                    ax.text(intensity_scale[i], spatial_scale[j],'{:0.2f}'.format((fss_array[j,i])), ha='center', va='center',size="small",weight="bold")
                else:
                    ax.text(intensity_scale[i], spatial_scale[j],'{:0.2f}'.format((fss_array[j,i])), ha='center', va='center',size="small")
            else:
                pass
    plt.xticks(intensity_scale)
    plt.yticks(spatial_scale)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_tick_params(which='minor', width=0) 
    ax.get_yaxis().set_tick_params(which='minor', size=0)
    ax.get_yaxis().set_tick_params(which='minor', width=0) 
    plt.xlabel("Intensity [mm/hr]")
    plt.ylabel("Scale [km]")
    plt.title(model+" "+threshold)
    plt.show()

produce_FSS_matrix(cloudburst_dates,"NEA","Cloudbursts")
produce_FSS_matrix(cloudburst_dates,"DK750","Cloudbursts")
produce_FSS_matrix(extreme_dates,"NEA","Extreme Precipitation")
produce_FSS_matrix(extreme_dates,"DK750","Extreme Precipitation")
produce_FSS_matrix(rainy_dates,"NEA","Precipitation")
produce_FSS_matrix(rainy_dates,"DK750","Precipitation")
produce_FSS_matrix(severe_dates,"NEA","Severe Rain")
produce_FSS_matrix(severe_dates,"DK750","Severe Rain")

#Quantiles
fss_qt,fss_qt_nea,random_qt=Fractional_skillscore(orig_radar[6:12],orig_nwp,orig_nea,366,date,percentile=0.999,plot=True)
2.5*bisect_left(fss_qt,0.5+(random_qt[0]/2))

def produce_fss_percentiles(dates,percentile,plot=True):
    fss_nwp=[[] for _ in range(len(np.unique(dates)))]
    fss_nea=[[] for _ in range(len(np.unique(dates)))]
    scalemin_nwp=[]
    scalemin_nea=[]
    full_day=Counter([str(i)[:-2] for i in np.unique(dates)])
    for i in range(0,len(np.unique(dates))):
        date=str(np.unique(dates)[i])
        if date[:-2]==str(np.unique(dates)[i-1])[:-2]:
            pass
        else:
            
            try:
                load_radar=np.loadtxt("./25grid/Radar/radar_20%s.txt"%date[:-2])
            except FileNotFoundError:
                continue
        
            load_NEA=np.loadtxt("./25grid/NEA/nea_2d_%s.txt"%date)[0:6]
            load_nwp=np.loadtxt("./25grid/NWP750/nwp_2d_%s.txt"%date)[0:6]
            load_radar=np.loadtxt("./25grid/Radar/radar_20%s.txt"%date[:-2])
        
            if full_day[date[:-2]]>1:
                radar_list=[]
                orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1])
                radar_concat=orig_radar[int(date[-2:]):int(date[-2:])+6]
                for times in range(0,full_day[date[:-2]]):
                    date_new=str(np.unique(dates)[i+1])
                    load_NEA=np.concatenate((load_NEA,np.loadtxt("./25grid/NEA/nea_2d_%s.txt"%date_new)[0:6]),axis=0)
                    load_nwp=np.concatenate((load_nwp,np.loadtxt("./25grid/NWP750/nwp_2d_%s.txt"%date_new)[0:6]),axis=0)
                
                    radar_list.append(date_new[-2:])
                
                orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1])
                orig_nwp=load_nwp.reshape(load_nwp.shape[0],load_nwp.shape[1])
    
                for k in range(0,len(radar_list)):    
                    radar_concat=np.concatenate((radar_concat,orig_radar[int(radar_list[k]):int(radar_list[k])+6]),axis=0)
        
                fss_qt,fss_qt_nea,random_qt=Fractional_skillscore(radar_concat,orig_nwp,orig_nea,366,date,percentile=percentile,plot=plot)
                scalemin_nwp.append(2.5*bisect_left(fss_qt,0.5+(random_qt[0]/2)))
                scalemin_nea.append(2.5*bisect_left(fss_qt_nea,0.5+(random_qt[0]/2)))
            else:
            
                #print(date)
        
                #orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1] // 184, 184)
                orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1])
                #orig_nwp=load_nwp.reshape(load_nwp.shape[0],load_nwp.shape[1] // 184, 184)
                orig_nwp=load_nwp.reshape(load_nwp.shape[0],load_nwp.shape[1])
                #orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1] // 184, 184)
                orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1])
                fss_qt,fss_qt_nea,random_qt=Fractional_skillscore(orig_radar[int(date[-2:]):int(date[-2:])+6],orig_nwp,orig_nea,366,date,percentile=percentile,plot=plot)
                scalemin_nwp.append(2.5*bisect_left(fss_qt,0.5+(random_qt[0]/2)))
                scalemin_nea.append(2.5*bisect_left(fss_qt_nea,0.5+(random_qt[0]/2)))
    return np.array(scalemin_nwp),np.array(scalemin_nea)

scalemin_nwp=[]
scalemin_nea=[]
for i in np.arange(0.99,0.99999,0.0005):
    temp_nwp,temp_nea=produce_fss_percentiles(extreme_dates,i,plot=True)
    temp_nwp[temp_nwp>=914]=np.nan
    temp_nea[temp_nea>=914]=np.nan
    scalemin_nwp.append(temp_nwp)
    scalemin_nea.append(temp_nea)
    print(i)

nwp_meanscale=[np.nanmean(k) for k in scalemin_nwp]
nea_meanscale=[np.nanmean(k) for k in scalemin_nea]

plt.plot(np.arange(0.99,0.99999,0.0005),nwp_meanscale)
plt.plot(np.arange(0.99,0.99999,0.0005),nea_meanscale)

#Confusion matrix
#cm_day,ETS,PSS,FBI,TPR,TS,FPR,hit,miss,tot=confusion_mat(rg_2d,nwp_2d)

#plt.plot(hit)
#plt.plot(miss)

#plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1))
#plt.scatter(FPR,TPR)
#plt.xlim([0,1])
#plt.ylim([0,1])

#SAL
def SAL_output(dates,model,threshold):
    os.chdir("C:/Users/olive/Desktop/Speciale/Kode") 
    struc=[[] for _ in range(len(np.unique(dates)))]
    amp=[[] for _ in range(len(np.unique(dates)))]
    loca=[[] for _ in range(len(np.unique(dates)))]
    for i in range(0,len(np.unique(dates))):
        date=str(np.unique(dates)[i])
        try:
            load_radar=np.loadtxt("./25grid/Radar/radar_20%s.txt"%date[:-2])
        except FileNotFoundError:
            continue

        load_NEA=np.loadtxt("./25grid/NEA/nea_2d_%s.txt"%date)
        load_nwp=np.loadtxt("./25grid/NWP750/nwp_2d_%s.txt"%date)
        load_radar=np.loadtxt("./25grid/Radar/radar_20%s.txt"%date[:-2])

        orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1] // 184, 184)
        #orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1])
        orig_nwp=load_nwp.reshape(load_nwp.shape[0],load_nwp.shape[1] // 184, 184)
        orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1] // 184, 184)
        if model=="NEA":
            struc[i],amp[i],loca[i]=produce_SAL(orig_radar[int(date[-2:]):int(date[-2:])+6],orig_nea)
        elif model=="DK750":
            struc[i],amp[i],loca[i]=produce_SAL(orig_radar[int(date[-2:]):int(date[-2:])+6],orig_nwp)
    struc=[x for x in struc if x!=[]]
    amp=[x for x in amp if x!=[]]
    loca=[x for x in loca if x!=[]]
        #print("Structure: %s"%struc)
        #print("Amplitude: %s"%amp)
        #print("Location: %s"%loca)
    title=model+" "+threshold
    plot_SAL(struc,amp,loca,title)
        
SAL_output(extreme_dates,"NEA","cloudbursts")
SAL_output(extreme_dates,"DK750","cloudbursts")
    

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
    