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

import geopandas as gpd
import glob

import time

import pandas as pd
import nwp750_read_plot
import radar_data_handling
import ClimateGrid_handling
import Other_functions
from nwp750_read_plot import *
from radar_data_handling import *
from ClimateGrid_handling import *
from Other_functions import *

import CRS
import gzip
import pickle
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

#NWP
date_time=str(2021072612)
coordinates_nwp=(7,15,54.5,58)
coor_file=[i for i in glob.glob("./NWP750/%s/*"%date_time) if len(i)<24]
extracted_nwp_coor=Output_rain_NWP(coor_file,threshold_value) 
lons_nwp=extracted_nwp_coor[1]['lons']
lats_nwp=extracted_nwp_coor[1]['lats']



Files_NEA=[i[-12:-4] for i in glob.glob("./25grid/NEA/*")] #what is available?

all_nwp=[i for i in glob.glob("./NWP750/pickled_data/pickled_data/*")]
#all_nwp=[i for i in glob.glob("./NWP750/pickled_data_4/*")]
Myfiles_nwp=[]
for i in range(0,len(all_nwp)):
    if all_nwp[i][-15:-7] in Files_NEA:
        Myfiles_nwp.append(all_nwp[i])
    else:
        pass

f=gzip.open(Myfiles_nwp[2],'rb')
df=pickle.load(f,encoding='bytes')
f.close()

for i in range(0,len(Myfiles_nwp)):
    start=time.time()
    file=Myfiles_nwp[i]
    save_name=Myfiles_nwp[i][-15:-7]
    f=gzip.open(file,'rb')
    df=pickle.load(f,encoding='bytes')
    f.close()
    df=df.transpose(2,0,1)
    start_zonal=time.time()
    zs=produce_zonalstat(df,lons_nwp,lats_nwp,file, "NWP")
    #print("Zonal stat takes:", time.time()-start_zonal)
    nwp_2d=zone_to_2d(zs)
    np.savetxt("./25grid/NWP750/nwp_2d_%s.txt"%(save_name[:10]),np.array(nwp_2d).reshape(np.array(nwp_2d).shape[0],-1))
    print("time for number %s:"%i,time.time()-start)
#Retrieve:
    
load_nwp=np.loadtxt("./25grid/NWP750/nwp_2d_%s.txt"%(save_name[:10]))
orig_nwp=load_nwp.reshape(load_nwp.shape[0],load_nwp.shape[1] // 184, 184)
plt.imshow(orig_nwp[4])

coor_4326=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/coor_4326.shp")
for i in range(0,len(orig_nwp)): 
    nwp_plot(orig_nwp[i], np.array(coor_4326['xcoor']).reshape((156,184)), np.array(coor_4326['ycoor']).reshape((156,184)),"test", world_map_file, 1, 1)


#orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1] // 157, 157)
#radar_plot(orig_radar[0],np.array(grid_25_coor_NWP['xcoord']).reshape(125,157),np.array(grid_25_coor_NWP['ycoord']).reshape(125,157),world_map_file,"Radar \n 26/07/2021 - 16:00 UTC",1)
#radar_plot(Radar_agg[0],radar_lons_4326,radar_lats_4326,world_map_file,"Radar \n 26/07/2021 - 16:00 UTC",1)

