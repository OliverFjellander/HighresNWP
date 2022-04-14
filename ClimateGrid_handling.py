# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:25:07 2022

@author: olive
"""
#import os
#os.chdir("C:/Users/olive/OneDrive/Desktop/Speciale/Kode") 

import os
import gzip
import pandas as pd
import numpy as np
from pyproj import Transformer
from pyproj import CRS
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from matplotlib.colors import LinearSegmentedColormap
import tarfile
import glob
from PIL import Image
import imageio
from pygeoprocessing import zonal_statistics
from osgeo import gdal
from osgeo import osr
import rasterio
import rioxarray as rxr


##############################################################################
##############################################################################
##############################################################################
########################## FUNCTIONS #########################################


def extract_tar(file_to_open):
    file_import=file_to_open
    file_tar=tarfile.open(file_import)
    file_tar.extractall(os.getcwd())
    file_tar.close()
    return ""

def raingauge_obs(Data):    
    df=np.zeros((len(Data),354,328))
    for i in range(0,len(Data)):
        file=open_gzip(Data[i])
        rainobs_crs=CRS("epsg:25832")
        plotting_crs=CRS("epsg:4326")
        tst_transformer=Transformer.from_crs(rainobs_crs, plotting_crs, always_xy = True)
        new_coords = tst_transformer.transform(np.array(file.iloc[:,2]), np.array(file.iloc[:,1]))
        file['E coor new'],file['N coor new']=new_coords[0],new_coords[1]
        precip_lons,precip_lats=project_raingauge_coords(file.iloc[:,1],file.iloc[:,2],rainobs_crs,plotting_crs)
        df[i]=rain_obs_restructure(file)
    return df,precip_lons,precip_lats

def raingauge_obs_orig(Data):    
    df=np.zeros((len(Data),354,328))
    for i in range(0,len(Data)):
        file=open_gzip(Data[i])
        rainobs_crs=CRS("epsg:25832")
        tst_transformer=Transformer.from_crs(rainobs_crs, rainobs_crs, always_xy = True)
        new_coords = tst_transformer.transform(np.array(file.iloc[:,2]), np.array(file.iloc[:,1]))
        file['E coor new'],file['N coor new']=new_coords[0],new_coords[1]
        precip_lons,precip_lats=project_raingauge_coords(file.iloc[:,1],file.iloc[:,2],rainobs_crs,rainobs_crs)
        df[i]=rain_obs_restructure(file)
    return df,precip_lons,precip_lats

def open_gzip(file_to_open):
    with gzip.open(file_to_open,'rt') as fin:        
        file=[]
        for line in fin:        
            file.append(line)
    file=pd.DataFrame(file)
    data=pd.DataFrame()
    data['GridID']=[float(file[0][i].split()[0]) for i in range(0,len(file.index))]
    data['E coor']=[float(file[0][i].split()[1]) for i in range(0,len(file.index))]
    data['N coor']=[float(file[0][i].split()[2]) for i in range(0,len(file.index))]
    data['rain']=[float(file[0][i].split()[3]) for i in range(0,len(file.index))]
    return data

def project_raingauge_coords(x_coords, y_coords, orig_crs, dest_crs):
    xx, yy = np.meshgrid(np.unique(x_coords), np.unique(y_coords))
    transformer = Transformer.from_crs(orig_crs, dest_crs, always_xy = True)
    new_coords = transformer.transform(xx, yy)
    x_new, y_new = new_coords[0], new_coords[1]
    return(x_new, y_new)

def rain_obs_restructure(data):
    rain_obs=pd.DataFrame(np.zeros((np.shape(np.unique(data.iloc[:,2]))[0],(np.shape(np.unique(data.iloc[:,1]))[0]))))
    df_lon,df_lat=np.meshgrid(np.unique(data.iloc[:,1]),np.unique(data.iloc[:,2])) #uses this to be able to compare the numbers in the datafram
    #start=time.time()
    for i in range(0,np.shape(df_lon)[0]):
        #print(i,np.shape(df_lon)[0])
        #start_small=time.time()
        for j in range(0, np.shape(df_lon)[1]):
            if any(data.index[((data['N coor']==df_lat[:,0][i]) & (data['E coor']==df_lon[0,:][j]))])==True:
                indice_input=data.index[((data['N coor']==df_lat[:,0][i]) & (data['E coor']==df_lon[0,:][j]))][0]
                rain_obs.iloc[i,j]=data.iloc[indice_input,:]['rain']
            else:
              pass
        #print("Time for indice input:",(time.time()-start_small))
    #print("time used:",(time.time()-start))
    rain_obs[rain_obs==0]=np.nan
    return rain_obs
    
def raingauge_plot(rain_array, lons, lats, world_map_file, plot_title,file_input):
    
    world_map = gpd.read_file(world_map_file)
    
    # create custom color map
    
    #nodes=[0,0.5,1,2,4,8,16,32,50,75,100]
    #colors_nodes=["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"]
    #cmap=LinearSegmentedColormap.from_list("",list(zip(nodes,colors_nodes)))
    cmap = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
    boundaries = [0, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100]
    #max_rain=rain_array.max(numeric_only=True).max()
    #boundaries=[0,1,np.round(max_rain*0.05),np.round(max_rain*0.10),np.round(max_rain*0.15),np.round(max_rain*0.2),np.round(max_rain*0.25),np.round(max_rain*0.375),np.round(max_rain*0.5),np.round(max_rain*0.75),np.round(max_rain)]
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
    #save_name="./Pics/%s.png"%file_input[0][-20:-7]
    #if os.path.isfile(save_name):
    #    os.remove(save_name)
    #plt.savefig(save_name,bbox_inches='tight')
    #plt.close()
    
# def data_to_raster_rg(data_array,lon,lat,path):
#     xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]
#     nrows,ncols = np.shape(data_array)
#     xres = (xmax-xmin)/float(ncols)
#     yres = (ymax-ymin)/float(nrows)
#     geotransform=(xmin,xres,0,ymax,0, -yres)   

#     output_raster = gdal.GetDriverByName('GTiff').Create(path,ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
#     output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
#     srs = osr.SpatialReference()                 # Establish its coordinate encoding
#     srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                             
#     output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system
#     output_raster.GetRasterBand(1).WriteArray(np.flip(np.array(data_array),axis=0))   # Writes my array to the raster
#     output_raster.FlushCache()

# Make a raster that the data can be inserted into. Produces a tiff file.
def data_array_to_raster_rg(data_array, tif_path):
    #transform = rasterio.transform.from_origin(-422114.8, 469381, 500, 500) # define coordinates for the DMI grid
    transform = rasterio.transform.from_origin(441500, 6049500, 1000, -1000) # define coordinates for the DMI grid
    proj4string_dmistere = "epsg:25832"  # The raw data's projection
    #transform = rasterio.transform.from_origin(7.582964884675271,58.32806110474353,0.008519049771461227, 0.004481469249843848) # define coordinates for the DMI grid
    #proj4string_dmistere = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs " # The raw data's projection

    # Produce raster with rasterio package
    with rasterio.open(tif_path, 'w', driver='GTiff',
                       height = data_array.shape[0], width = data_array.shape[1],
                       count=1, dtype=str(data_array.dtype),
                       crs=proj4string_dmistere,
                       transform=transform) as file:
        file.write(data_array, 1)
    
    raster_array = rxr.open_rasterio('./{}'.format(tif_path),tif_path).squeeze() # load data
    
    with rxr.open_rasterio(tif_path) as file:
        raster_array = file.squeeze()
    
    #raster_array.close()
    #os.remove('./data_array.tif')
    
    return(raster_array)

def produce_zonalstat_rg(data,lons,lats,filename):
    zs=[]

    for i in range(0,len(data)):
        tif_path="./Tiff_files/"+filename[i][-20:-9]+'_raingauge'+'.tif'
        data_array_to_raster_rg(data[i],tif_path)

        #raster_radar=rasterio.open(tif_path_radar,masked=False)
        #raster_nwp=rasterio.open(tif_path_nwp,masked=False)

        zs.append(zonal_statistics((tif_path,1),"C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI_raingauge.shp",ignore_nodata=False,polygons_might_overlap=False))
        #print(time.time()-start) 
    return zs

def zonalstat_rg(data_zonalstat):
    data_ts=np.empty((0,len(data_zonalstat)),np.float64)
    for i in range(0,len(data_zonalstat)):
        if np.isnan(data_zonalstat[i]['sum']):
            data_ts=np.append(data_ts,0.0)
        else:
            data_ts=np.append(data_ts,data_zonalstat[i]['sum']/data_zonalstat[i]['count'])
    return data_ts

def zone_to_2d_raingauge(z_stat):
    grid_2d=[[] for _ in range(len(z_stat))]
    for i in range(0,len(z_stat)):
        grid_2d[i]=np.reshape(zonalstat_rg(z_stat[i]),(125,157))
        grid_2d[i][grid_2d[i]==0]=np.nan
        grid_2d[i][grid_2d[i]==0]=np.nan
    return grid_2d
    

    
def make_gif(path_to_png,path_placing_gif,duration):
    frames=[]
    imgs=glob.glob(path_to_png)
    for i in imgs:
        new_frame=Image.open(i)
        frames.append(new_frame)
    imageio.mimsave(path_placing_gif,frames,'GIF',duration=duration)

##############################################################################
##############################################################################
##############################################################################


#os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Rain_gauge") # change directory to where the files are located!

#df=open_gzip('./Precip2011/2011/01/01/20110101_0100.txt.gz')


#file_import='.Data/Precip2021.tar.gz'
#extract_tar(file_import) #Only need to do this once

#file_gz="./Data/2021/07/26/20210726_1500.txt.gz"

#df=open_gzip(file_gz)

#rainobs_crs=CRS("epsg:25832")
#plotting_crs=CRS("epsg:4326")
#tst_transformer=Transformer.from_crs(rainobs_crs, plotting_crs, always_xy = True)
#new_coords = tst_transformer.transform(np.array(df.iloc[:,2]), np.array(df.iloc[:,1]))
#df['E coor new'],df['N coor new']=new_coords[0],new_coords[1]


#precip_lons,precip_lats=project_raster_coords(np.unique(df.iloc[:,1]),np.unique(df.iloc[:,2]),rainobs_crs,plotting_crs)


#rainobs=rain_obs_restructure(df)
      


#Try making the above to a list comprehension
#indice_input2=[]
#indice_input2=[rain_obs_new2.iloc[i,j]=df.iloc[df.index[((df['N coor']==tst_lat[:,0][i]) & (df['E coor']==tst_lon[0,:][j]))][0],:]['rain'] for i in range(0,np.shape(tst_lon)[0]) for j in range(0, np.shape(tst_lon)[1]) if any(df.index[((df['N coor']==tst_lat[:,0][i]) & (df['E coor']==tst_lon[0,:][j]))])==True]  
#indice_input3=[]
#start2=time.time()
#indice_input3=[df.index[((df['N coor']==tst_lat[:,0][i]) & (df['E coor']==tst_lon[0,:][j]))][0] for i in range(0,np.shape(tst_lon)[0]) for j in range(0, np.shape(tst_lon)[1]) if any(df.index[((df['N coor']==tst_lat[:,0][i]) & (df['E coor']==tst_lon[0,:][j]))])==True ]  
#elapsed_time_lc=(time.time()-start2)



#Myfiles_raingauge=[i for i in glob.glob("./Data/2021/07/26/20210726_*.txt.gz") if int(i[-11:-7])>1200 and int(i[-11:-7])<2200]

world_map_file = "C:/Users/olive/OneDrive/Desktop/Speciale/Kode/pygrib_functionality/pygrib_functionality/world_map_cut/world_map_background.shp" # file path to a shapefile with outline of Denmark

#df_raingauge=[open_gzip(i) for i in Myfiles_raingauge]

    
#rainobs,rainobs_lons,rainobs_lats=raingauge_obs(Myfiles_raingauge)

#for i in range(0,len(rainobs)):    
#    raingauge_plot(rainobs[i], precip_lons, precip_lats, world_map_file, "%s/%s/%s - %s"%(Myfiles_raingauge[i][-14:-12],Myfiles_raingauge[i][-16:-14],Myfiles_raingauge[i][-20:-16],Myfiles_raingauge[i][-11:-7]),Myfiles_raingauge[i])

#make_gif("./Pics/*.png","./Pics/png_to_gif.gif",1)
