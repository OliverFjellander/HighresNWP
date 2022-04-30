import os
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
from osgeo import gdal
from osgeo import osr
from pygeoprocessing import zonal_statistics
#from skimage.feature import peak_local_max
# Function that reads the raw data from an HDF5 file

            
##############################################

def movefiles(input_path,output_path):
    file_move=[i for i in glob.glob(input_path)]
    for f in file_move:
        source=os.getcwd()+f
        destination=output_path
        filename=os.path.basename(os.getcwd()+f)
        dest=os.path.join(destination,filename)
        shutil.move(source,dest)
    return print("Done")


def read_raw_radardata(file_path):   
    
    with h5py.File(file_path, 'r') as file:
        raw_data = file["dataset1"]["data1"]["data"]
        raw_data_array = raw_data[()]
    
    raw_data_array = raw_data_array.astype(np.float)
    
    return(raw_data_array)


# Function that converts raw data to dBZ values
def raw_radardata_to_dbz(raw_data_array):
    raw_data_array[raw_data_array == 255] = np.nan # values of 255 in raw data are actually NaN
    zero_values = raw_data_array == 0 # store where zero values are located in the grid, as these are changed by the transformation below

    gain = 0.5
    offset = -32
    dbz_data = offset + gain * raw_data_array # convert to dBZ
    
    dbz_data[zero_values] = np.nan # insert NaN where there originally where zeros (if zeros are needed rather than NaN, change np.nan to 0 here)
    
    return(dbz_data)

# Function that converts dBZ values to rainfall rates with constant Marshall-Palmer
def dbz_to_R_marshallpalmer(dbz_data):
    z_data = np.power(10, dbz_data/10)

    a = 200 # DMI recommended value
    b = 1.6 # DMI recommended value
    rain_data = np.power(z_data/a, 1/b)

    #rain_data_nozeros = rain_data
    #rain_data_nozeros[zero_values] = np.nan
    
    return(rain_data)



def data_to_raster_RADAR(data_array,lon,lat,path):
    xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]
    nrows,ncols = np.shape(data_array)
    xres = (xmax-xmin)/float(ncols)
    yres = (ymax-ymin)/float(nrows)
    geotransform=(xmin,xres,0,ymax,0, -yres)   

    output_raster = gdal.GetDriverByName('GTiff').Create(path,ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
    srs = osr.SpatialReference()                 # Establish its coordinate encoding
    srs.ImportFromProj4('+proj=stere +ellps=WGS84 +lat_0=56 +lon_0=10.5666 +lat_ts=56')                     # This one specifies WGS84 lat long.
                                             
    output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system
    output_raster.GetRasterBand(1).WriteArray(data_array)   # Writes my array to the raster
    output_raster.FlushCache()

def produce_zonalstat_radar(data,lons,lats,files):
    tif_path ="./Tiff_files/"+files[0][-15:-5] + "_radar" + ".tif" # file path for a tif file that will be generated
    data_to_raster_RADAR(data,lons,lats,tif_path)
    zs=zonal_statistics((tif_path,1),"C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/25gridradar.shp",ignore_nodata=False,polygons_might_overlap=False)
    return zs

    
# Function that transforms a set of coordinates
# Input a set of coordinates, the original CRS and desired output CRS
def project_raster_coords(x_coords, y_coords, orig_crs, dest_crs):
    xx, yy = np.meshgrid(x_coords, y_coords)
    transformer = Transformer.from_crs(orig_crs, dest_crs, always_xy = True)
    new_coords = transformer.transform(xx, yy)
    x_new, y_new = new_coords[0], new_coords[1]
    return(x_new, y_new)

def remove_values_below(surface_field, threshold):
    surface_field[surface_field <= threshold] = np.nan
    return(surface_field)


def aggregate_data(Data,threshold_value): #Aggregate data for every hour
    #rain_total=np.zeros((len(Data),1728,1984),float)
    rain_total=np.zeros((len(Data),1500-412,1175-494),float)
    for i in range(0,len(Data)):
        for j in range(0,len(Data[i])):
            raw=read_raw_radardata(Data[i][j])
            dbz=raw_radardata_to_dbz(raw)
            dbz_smalldom=dbz[412:1500,494:1175]
            #rain=dbz_to_R_marshallpalmer(dbz)/60
            rain=dbz_to_R_marshallpalmer(dbz_smalldom)/60#Divide by 60 to get correct intensity
            rain_total[i]=rain_total[i]+pd.DataFrame(rain).fillna(0)
        rain_total[i]=remove_values_below(rain_total[i],threshold_value)
        #radar_plot(rain_total[i],radar_lons,radar_lats,world_map_file,"%s/%s/%s - %s:00 UTC"%(Data[i][0][-9:-7],Data[i][0][-11:-9],Data[i][0][-15:-11],Data[i][j][-7:-5]),Data[i][j])
    return rain_total

# Function that makes a simple plot of the radar data on top of a map of Denmark
def radar_plot(rain_array, lons, lats, world_map_file, plot_title,file_input):
    
    world_map = gpd.read_file(world_map_file)
    
    # create custom color map
    cmap = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
    #boundaries = [0, .5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
    boundaries = [0, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    # Make plot
    world_map.plot(facecolor="lightgrey")
    plt.pcolor(lons, lats, rain_array, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Rainfall intensity [mm/h]', rotation=90)
    plt.xlim(7, 15)
    plt.ylim(54.5, 58)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(plot_title)
    #save_name="./Radar/Data/Pics/%s_hr.png"%file_input[-15:-3]
    #if os.path.isfile(save_name):
    #    os.remove(save_name)
    #plt.savefig(save_name,bbox_inches='tight')
    #plt.close()
    

    
def make_gif(path_to_png,path_placing_gif,duration):
    frames=[]
    imgs=glob.glob(path_to_png)
    for i in imgs:
        new_frame=Image.open(i)
        frames.append(new_frame)
    imageio.mimsave(path_placing_gif,frames,'GIF',duration=duration)
    



########################
########################
########################


#os.chdir("C:/Users/olive/Desktop/Speciale/Kode") # change directory to where the files are located!

        
#movefiles("./Radar/Data/nowcast.dk.com.*/interpolated.*.h5","./Radar/Data/Interpolations") #Move interpolated files for all folders to store them in one place
    

#file_path = './Radar/Data/nowcast.dk.com.202107261200/interpolated.dk.com.202107261150.h5' # example file
#tif_path_rain = file_path[:-3] + "_rain" + ".tif" # file path for a tif file that will be generated

# this is how you access the where, how, what attributes of raw file
#coordinates_info = list(test_data["where"].attrs.items())

#raw_data = read_raw_radardata(file_path) # read raw data
#dbz_data = raw_radardata_to_dbz(raw_data) # convert to dBZ values
#rain_data = dbz_to_R_marshallpalmer(dbz_data) # convert to rainfall intensity

# make numpy arrays with coordinates
x_radar_coords = np.arange(-421364.8 - 500, 569635.2 + 500, 500) # need to add and subtract 500 because np.arange does not include end points
y_radar_coords = np.arange(468631 + 500, -394369 - 500, -500)

dmi_stere_crs = CRS("+proj=stere +ellps=WGS84 +lat_0=56 +lon_0=10.5666 +lat_ts=56") # raw data CRS projection
plotting_crs = 'epsg:4326' # the CRS projection you want to plot the data in

# Project the raw data into the target CRS
radar_lons, radar_lats = project_raster_coords(x_radar_coords, y_radar_coords, dmi_stere_crs, plotting_crs)

# Make a simple plot
world_map_file = "C:/Users/olive/OneDrive/Desktop/Speciale/Kode/pygrib_functionality/pygrib_functionality/world_map_cut/world_map_background.shp" # file path to a shapefile with outline of Denmark

#start_time=time.time()
#radar_plot(rain_data, radar_lons, radar_lats, world_map_file, "%s/%s/%s - %s:%s UTC"%(file_path[-9:-7],file_path[-11:-9],file_path[-15:-11],file_path[-7:-5],file_path[-5:-3]),file_path)
#print(time.time()-start_time)

#######################################################
########################################################
#######################################################

#Myfiles=[i for i in glob.glob("./Radar/Data/Interpolations/*")] #There should only be complete hours

#Myfiles_hr=np.array_split(Myfiles[1-len(Myfiles):],len(Myfiles[1-len(Myfiles):])/60) #The first is removed because the belongs to previous timeframe

#agg_list=aggregate_data(Myfiles_hr,0.5)


#make_gif("./Radar/Data/Pics/*_hr.png",'./Radar/Data/Pics/png_to_gif_hr.gif',1)

