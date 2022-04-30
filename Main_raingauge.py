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

###########################################################################
###########################################################################
###########################################################################
os.chdir("C:/Users/olive/Desktop/Speciale/Kode") 
world_map_file = "C:/Users/olive/OneDrive/Desktop/Speciale/Kode/pygrib_functionality/pygrib_functionality/world_map_cut/world_map_background.shp" # file path to a shapefile with outline of Denmark
grid=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/coor_4326.shp") 
grid_x=np.array(grid['xcoor']).reshape(156,184)
grid_y=np.array(grid['ycoor']).reshape(156,184)
threshold_value=0.5
dmi_stere_crs = CRS("+proj=stere +ellps=WGS84 +lat_0=56 +lon_0=10.5666 +lat_ts=56") # raw data CRS projection
plotting_crs = 'epsg:4326' # the CRS projection you want to plot the data in
nwp_crs = 'epsg:25832' # the CRS projection you want to plot the data in


st=time.time()
rainobs,rainobs_lons,rainobs_lats=raingauge_obs(raingauge_days[0:1])
print(time.time()-st)
st1=time.time()
rainobs,rainobs_lons,rainobs_lats=raingauge_obs_orig(raingauge_days[0:1])
print(time.time()-st1)

###############################################################################
month=["05","06","07","08","09","10"]

date_times=[]
for k in range(0,len(month)):
    file_raingauge=glob.glob("./Rain_gauge/Data/2021/%s/*"%month[k])
    days=len(file_raingauge)
    for number in range(0,days):
        start_start=time.time()
        
        raingauge_days=glob.glob(str(file_raingauge[number])+"/*")
        df_raingauge=[open_gzip(i) for i in raingauge_days]
        #rainobs,rainobs_lons,rainobs_lats=raingauge_obs(raingauge_days)

        
        zs_raingauge=produce_zonalstat_rg(rainobs,rainobs_lons,rainobs_lats,Myfiles_raingauge)
        rg_2d=zone_to_2d_raingauge(zs_raingauge)
        
        save_name=str(file_raingauge[number])[-5:]


###############################################################################
###############################################################################
#aggregate
        np.savetxt("./25grid/raingauge/raingauge_%s_%s.txt"%(save_name[:-3],save_name[-2:]),np.array(rg_2d).reshape(np.array(rg_2d).shape[0],-1))
        print("Time in total:", time.time()-start_start)
        print(number)


#For testing
#load_radar=np.loadtxt("./25grid/Radar/radar_%s.txt"%save_name[:8])
#load_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1] // 184,184)
#radar_plot(Radar_agg[0],radar_lons_4326,radar_lats_4326,world_map_file,"test",1)
#radar_plot(load_radar[0],grid_x,grid_y,world_map_file,"test",1)
