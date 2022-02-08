# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:31:43 2022

@author: olive
"""
import os
os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Scripts") # change directory to where the files are located!

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import colors
import rasterio
import rioxarray as rxr
import geopandas as gpd
from pyproj import Transformer
from pyproj import CRS
import glob
import tifftools #For later when merging the tiff files
from collections import defaultdict
import imageio
from pathlib import Path
from osgeo import gdal
from matplotlib.animation import FuncAnimation
import time
from PIL import Image
import shutil
import re
import pandas as pd
import nwp750_read_plot
import radar_data_handling
import ClimateGrid_handling
from nwp750_read_plot import *
from radar_data_handling import *
from ClimateGrid_handling import *
import pygrib

###########################################################################
###########################################################################
###########################################################################
os.chdir("C:/Users/olive/Desktop/Speciale/Kode") 
world_map_file = "C:/Users/olive/OneDrive/Desktop/Speciale/Kode/pygrib_functionality/pygrib_functionality/world_map_cut/world_map_background.shp" # file path to a shapefile with outline of Denmark

threshold_value=0.5

#FILES
###############################################################################
#NWP
coordinates_nwp=(7,15,54.5,58)
Myfiles_nwp=[i for i in glob.glob("./NWP750/2021072612/*") if len(i)<24]
extracted_nwp_coor=Output_rain_NWP(Myfiles_nwp,threshold_value)
extracted_nwp=open_not_plot(Myfiles_nwp,threshold_value,world_map_file,coordinates_nwp)

###############################################################################
#Radar
#movefiles("./Radar/Data/nowcast.dk.com.*/interpolated.*.h5","./Radar/Data/Interpolations") #Move interpolated files for all folders to store them in one place
Myfiles_radar=[i for i in glob.glob("./Radar/Data/Interpolations/*")]
Myfiles_radar_hr=np.array_split(Myfiles_radar[1-len(Myfiles_radar):],len(Myfiles_radar[1-len(Myfiles_radar):])/60) #The first is removed because the belongs to previous timeframe
Radar_agg=aggregate_data(Myfiles_radar_hr,threshold_value)


#Radar coordinates
x_radar_coords = np.arange(-421364.8 - 500, 569635.2 + 500, 500) # need to add and subtract 500 because np.arange does not include end points
y_radar_coords = np.arange(468631 + 500, -394369 - 500, -500)

dmi_stere_crs = CRS("+proj=stere +ellps=WGS84 +lat_0=56 +lon_0=10.5666 +lat_ts=56") # raw data CRS projection
plotting_crs = 'epsg:4326' # the CRS projection you want to plot the data in

radar_lons, radar_lats = project_raster_coords(x_radar_coords, y_radar_coords, dmi_stere_crs, plotting_crs)
###############################################################################
#Rain gauge
Myfiles_raingauge=[i for i in glob.glob("./Rain_gauge/Data/2021/07/26/20210726_*.txt.gz") if int(i[-11:-7])>1200 and int(i[-11:-7])<2200]
df_raingauge=[open_gzip(i) for i in Myfiles_raingauge]
rainobs,rainobs_lons,rainobs_lats=raingauge_obs(Myfiles_raingauge)



###############################################################################
###############################################################################


#Radar and NWP750
for i in range(0,len(Radar_agg)):
    plot_together(Radar_agg[i],extracted_nwp[i+1], radar_lons, radar_lats, extracted_nwp_coor[0]['lons'],extracted_nwp_coor[0]['lats'], world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[i][-1:])[-11:-9],str(Myfiles_radar_hr[i][-1:])[-13:-11],str(Myfiles_radar_hr[i][-1:])[-17:-13],str(Myfiles_radar_hr[i][-1:])[-9:-7]),
                  "NWP \n %s/%s/%s - %s:00 + %s UTC"%(Myfiles_nwp[0][-8:-6],Myfiles_nwp[0][-10:-8],Myfiles_nwp[0][-14:-10],Myfiles_nwp[0][-6:-4],str(1+i)),Myfiles_radar_hr[i],"./Pics/%s.png")

make_gif("./Pics/*.png",'./Pics/png_to_gif_hr.gif',1)


#Radar and raingauge
for i in range(0,len(Radar_agg)):
    plot_together(Radar_agg[i],rainobs[i], radar_lons, radar_lats, rainobs_lons,rainobs_lats, world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[i][-1:])[-11:-9],str(Myfiles_radar_hr[i][-1:])[-13:-11],str(Myfiles_radar_hr[i][-1:])[-17:-13],str(Myfiles_radar_hr[i][-1:])[-9:-7]),
                  "Obs \n %s/%s/%s - %s:%s UTC"%(Myfiles_raingauge[0][-14:-12],Myfiles_raingauge[0][-14:-16],Myfiles_raingauge[0][-20:-16],Myfiles_raingauge[i][-11:-9],Myfiles_raingauge[i][-9:-7]),Myfiles_radar_hr[i],"./Pics/RadarRaingauge/%s.png")

make_gif("./Pics/RadarRaingauge/*.png",'./Pics/RadarRaingauge/png_to_gif.gif',1)






