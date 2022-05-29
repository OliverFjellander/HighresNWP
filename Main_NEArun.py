# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:31:43 2022

@author: olive
"""
import os
os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Scripts") # change directory to where the files are located!

import numpy as np

#import h5py
from pyproj import CRS
import glob
import gzip

import nwp750_read_plot
import radar_data_handling
import ClimateGrid_handling
import Other_functions
from nwp750_read_plot import *
from radar_data_handling import *
from ClimateGrid_handling import *
from Other_functions import *
import pygrib

import pickle

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

#NWP
#extract_tar('./NWP750/dk7502021070810.tar.gz')
#extract_tar('./NWP750/dk7502021050101.pkl.gz')
coordinates_nwp=(7,15,54.5,58)
Myfiles_nwp=[i for i in glob.glob("./NWP2500/*")]

NEA_file="./pygrib_functionality/pygrib_functionality/sNEA2104302103"
tst_nea=pygrib.open(NEA_file)
nea_param=tst_nea.read()
extracted_nea=read_parameter_info(nea_param,58)
extracted_nea['values']=remove_values_below(extracted_nea["values"],0.5)
#nea_lats=extracted_nea['lats']
#nea_lons=extracted_nea['lons']
#np.savetxt("./nea.csv",np.transpose(np.vstack((np.array((nea_lons.flatten())),np.array(nea_lats.flatten()),np.array(np.repeat(1,np.size(nea_lons)))))),delimiter=",")

nea_lats=extracted_nea['lats'][255:455,535:692]
nea_lons=extracted_nea['lons'][255:455,535:692]

for i in range(35,len(Myfiles_nwp)):
    save_name=str(Myfiles_nwp[i])[-21:-13]
    
    f=gzip.open(Myfiles_nwp[i],'rb')
    loaded=pickle.load(f,encoding='bytes')
    f.close()

    df=loaded.transpose(2,0,1)
    
    zs=produce_zonalstat(df,nea_lons,nea_lats,Myfiles_nwp[i],"NWP25")

    nwp_2d=zone_to_2d(zs)
    np.savetxt("./25grid/NEA/nea_2d_%s.txt"%(save_name[:10]),np.array(nwp_2d).reshape(np.array(nwp_2d).shape[0],-1))


load_nwp=np.loadtxt("./25grid/NEA/nea_2d_%s.txt"%(save_name[:10]))
orig_nwp=load_nwp.reshape(load_nwp.shape[0],load_nwp.shape[1] // 184, 184)


load_radar=np.loadtxt("./25grid/Radar/radar_20210507.txt")
orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1] // 184, 184)

#plt.imshow(df[0])
#plot_together(orig_radar[13], orig_nwp[6], np.array(grid_4326['xcoor']).reshape((156,184)), np.array(grid_4326['ycoor']).reshape((156,184)), np.array(grid_4326['xcoor']).reshape((156,184)), np.array(grid_4326['ycoor']).reshape((156,184)), world_map_file, "radar", "nwp", 1,1)


#data_to_raster_NWP(np.repeat(1,np.size(lons_nwp)).reshape(630,700),lons_nwp,lats_nwp,"./nwp_area.tif")        
#np.savetxt("./high_res.csv",np.transpose(np.vstack((np.array((lons_nwp.flatten())),np.array(lats_nwp.flatten()),np.array(np.repeat(1,np.size(lons_nwp)))))),delimiter=",")
   


    