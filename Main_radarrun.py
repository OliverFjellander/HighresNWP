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


###############################################################################
#Radar
#Radar coordinates
x_radar_coords = np.arange(-421364.8 - 500, 569635.2 + 500, 500) # need to add and subtract 500 because np.arange does not include end points
y_radar_coords = np.arange(468631 + 500, -394369 - 500, -500)

#####Limit radar
radar_lons, radar_lats = project_raster_coords(x_radar_coords, y_radar_coords, dmi_stere_crs, dmi_stere_crs)
radar_lons_4326,radar_lats_4326=project_raster_coords(x_radar_coords, y_radar_coords, dmi_stere_crs, plotting_crs)

radar_lons=radar_lons[412:1500,494:1175]
radar_lats=radar_lats[412:1500,494:1175]
radar_lons_4326=radar_lons_4326[412:1500,494:1175]
radar_lats_4326=radar_lats_4326[412:1500,494:1175]
#FILES
###############################################################################
month="10"

file_radar=glob.glob("./Radar/2021-%s/*"%month)
file_short=[file_radar[s][-15:-7] for s in range(0,len(file_radar))]

date_times=[item for item, count in collections.Counter(file_short).items() if count > 1439]
# date_times=[]
# for i in range(0,len(remove_dup)):
#     for j in range(0,24,1):
#         if len(str(j))==1:
#             hour=str(0)+str(j)
#         else:
#             hour=str(j)
#         date_times.append(remove_dup[i]+hour)
    
for i in range(0,len(date_times)):
    start_start=time.time()
    

    Myfiles_radar=[i for i in glob.glob("./Radar/2021-%s/*.%s*"%(month,date_times[i]))]
    Myfiles_radar_hr=np.array_split(Myfiles_radar,24) #The first is removed because the belongs to previous timeframe
    Radar_agg=aggregate_data(Myfiles_radar_hr,threshold_value)

    save_name=str(Myfiles_radar_hr[0][0])[-15:-5]

###############################################################################
###############################################################################
#aggregate

    zs_radar=produce_zonalstat(Radar_agg,radar_lons,radar_lats,Myfiles_radar,"radar")

    radar_2d=zone_to_2d(zs_radar)

    np.savetxt("./25grid/Radar/radar_%s.txt"%(save_name[:8]),np.array(radar_2d).reshape(np.array(radar_2d).shape[0],-1))
    print("Time in total:", time.time()-start_start)
    print(i)


#For testing
#load_radar=np.loadtxt("./25grid/Radar/radar_%s.txt"%save_name[:8])
#load_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1] // 184,184)
#radar_plot(Radar_agg[0],radar_lons_4326,radar_lats_4326,world_map_file,"test",1)
#radar_plot(load_radar[0],grid_x,grid_y,world_map_file,"test",1)
