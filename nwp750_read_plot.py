# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 08:41:10 2022

@author: olive
"""

import pygrib
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import pandas as pd
from osgeo import gdal,osr
from pyproj import Transformer


def read_parameter_info(parameter_list, param_number):
    ''' Takes a number for the parameter of interest, 
        returns various information incl. gridded values '''
    parameter_list[param_number].dataDate
    parameter_list[param_number].parameterName
    parameter_list[param_number].indicatorOfTypeOfLevel
    parameter_list[param_number].typeOfLevel
    parameter_list[param_number].latitudes
    parameter_list[param_number].longitudes
    parameter_list[param_number].values
    
    parameter_list[param_number].Ni
    parameter_list[param_number].Nj
    
    
    lats = parameter_list[param_number].latitudes.reshape(parameter_list[param_number].Nj, parameter_list[param_number].Ni)
    lons = parameter_list[param_number].longitudes.reshape(parameter_list[param_number].Nj, parameter_list[param_number].Ni)
    grid_values = parameter_list[param_number].values.reshape(parameter_list[param_number].Nj, parameter_list[param_number].Ni)
    
    
    param_output = {"date": parameter_list[param_number].dataDate,
                    "grib_number": parameter_list[param_number].parameterName,
                    "indicator_type": parameter_list[param_number].indicatorOfTypeOfLevel,
                    "type_of_level": parameter_list[param_number].indicatorOfTypeOfLevel,
                    "lats": lats,
                    "lons": lons,
                    "values": grid_values}
        
    return(param_output)

def remove_values_below(surface_field, threshold):
    surface_field[surface_field <= threshold] = np.nan
    return(surface_field)


def Output_rain_NWP(file_path,removal_threshold):
    #extracted_field=np.zeros((len(file_path),630,700))
    extracted_field={}
    for i in range(0,len(file_path)):
        pygrib_file = pygrib.open(file_path[i])
        parameter_list=pygrib_file.read()
        extracted_field[i]=read_parameter_info(parameter_list,58)
        extracted_field[i]['values']=remove_values_below(extracted_field[i]["values"],removal_threshold)
    return extracted_field

def project_raster_coords_NWP(x_coords, y_coords, orig_crs, dest_crs):
    xx, yy = x_coords, y_coords
    transformer = Transformer.from_crs(orig_crs, dest_crs, always_xy = True)
    new_coords = transformer.transform(xx, yy)
    x_new, y_new = new_coords[0], new_coords[1]
    return(x_new, y_new)

def nwp_plot(rain_array, lons, lats, plot_title,file,coordinates,file_input):
    ''' Function that plots gridded NWP data'''
    world_map = gpd.read_file(file)
    
    # create custom color map
    cmap = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
    #boundaries = [0, .5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
    boundaries = [0, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    
    world_map.plot(facecolor="lightgrey")
    plt.pcolor(lons, lats, rain_array, shading = "auto", alpha=0.8, cmap=cmap, norm=norm)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Rainfall intensity [mm/h]', rotation=90)
    plt.xlim(6,16) #7,15 longitudes for Denmark (for a zoomed plot)
    plt.ylim(53,59) #54,5 , 58 lattitudes for Denmark (for a zoomed plot)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(plot_title)
    #save_name="./Pics/%s.png"%file_input[-3:]
    #if os.path.isfile(save_name):
    #    os.remove(save_name)
    #plt.savefig(save_name,bbox_inches='tight')
    #plt.close()
    

def unaccumulate(Data,removal_threshold,world_file,coordinates):
    previous_accum=pd.DataFrame(np.zeros((630,700)))
    rain={}
    for i in range(1,len(Data)):
        open_file = pygrib.open(Data[i])
        parameters=open_file.read()
        attributes=read_parameter_info(parameters,58)
        #threshold_new=1
        values_plot=pd.DataFrame(attributes['values']).fillna(0)-previous_accum
        values_plot=values_plot.replace(0,np.nan)
        values_plot=remove_values_below(values_plot,removal_threshold)
        previous_accum=pd.DataFrame(attributes['values']).fillna(0)
        rain[i]=values_plot
    return rain 


def data_to_raster_NWP(data_array,lon,lat,path):
    xmin,ymin,xmax,ymax = [lon.min(),lat.min(),lon.max(),lat.max()]
    nrows,ncols = np.shape(data_array)
    xres = (xmax-xmin)/float(ncols)
    yres = (ymax-ymin)/float(nrows)
    geotransform=(xmin,xres,0,ymax,0, -yres)   

    output_raster = gdal.GetDriverByName('GTiff').Create(path,ncols, nrows, 1 ,gdal.GDT_Float32)  # Open the file
    output_raster.SetGeoTransform(geotransform)  # Specify its coordinates
    srs = osr.SpatialReference()                 # Establish its coordinate encoding
    srs.ImportFromEPSG(4326)                     # This one specifies WGS84 lat long.
                                             
    output_raster.SetProjection( srs.ExportToWkt() )   # Exports the coordinate system
    output_raster.GetRasterBand(1).WriteArray(np.flip(np.array(data_array),axis=0))   # Writes my array to the raster
    output_raster.FlushCache()


#############################################################################
#############################################################################



