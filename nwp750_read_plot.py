# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 08:41:10 2022

@author: olive
"""

import tarfile
import os
import gzip
import pygrib
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors
import glob
import time
import pandas as pd
from PIL import Image
import imageio


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
    #plt.xlim(7,15) # longitudes for Denmark (for a zoomed plot)
    #plt.ylim(54.5,58) # lattitudes for Denmark (for a zoomed plot)
    plt.xlim(coordinates[0],coordinates[1])
    plt.ylim(coordinates[2],coordinates[3])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(plot_title)
    save_name="./Pics/%s.png"%file_input[-3:]
    if os.path.isfile(save_name):
        os.remove(save_name)
    plt.savefig(save_name,bbox_inches='tight')
    #plt.close()
    

def open_n_plot(Data,removal_threshold,world_file,coordinates):
    previous_accum=pd.DataFrame(np.zeros((630,700)))
    for i in range(1,len(Data)):
        start=time.time()
        open_file = pygrib.open(Data[i])
        parameters=open_file.read()
        attributes=read_parameter_info(parameters,58)
        #threshold_new=1
        attributes['values']=remove_values_below(attributes['values'],removal_threshold)
        values_plot=pd.DataFrame(attributes['values']).fillna(0)-previous_accum
        values_plot=values_plot.replace(0,np.nan)
        values_plot=remove_values_below(values_plot,removal_threshold)
        #nwp_plot(values_plot,attributes['lons'],attributes['lats'],"Rainfall field %2d - %s:00 UTC"%(attributes['date'],int(Data[i][-6:-4])+int(Data[i][-1:])),world_map_file,coordinates,Data[i])
        previous_accum=pd.DataFrame(attributes['values']).fillna(0)
        print(time.time()-start)
    return values_plot

def open_not_plot(Data,removal_threshold,world_file,coordinates):
    previous_accum=pd.DataFrame(np.zeros((630,700)))
    rain={}
    for i in range(1,len(Data)):
        open_file = pygrib.open(Data[i])
        parameters=open_file.read()
        attributes=read_parameter_info(parameters,58)
        #threshold_new=1
        attributes['values']=remove_values_below(attributes['values'],removal_threshold)
        values_plot=pd.DataFrame(attributes['values']).fillna(0)-previous_accum
        values_plot=values_plot.replace(0,np.nan)
        values_plot=remove_values_below(values_plot,removal_threshold)
        previous_accum=pd.DataFrame(attributes['values']).fillna(0)
        rain[i]=values_plot
    return rain 



#############################################################################
#############################################################################


#os.chdir("C:/Users/olive/Desktop/Speciale/Kode/NWP750") # change directory to where the files are located!


#file_grib="./2021072612/00115"
#file_grib="./2021072612/002"
#pygrib_file = pygrib.open(file_grib)

#parameter_list = pygrib_file.read()


#extracted_field = read_parameter_info(parameter_list, 58) #either 13 or 29 if file is not an exact hour 
#np.nanmax(extracted_field['values'])

world_map_file = "C:/Users/olive/OneDrive/Desktop/Speciale/Kode/pygrib_functionality/pygrib_functionality/world_map_cut/world_map_background.shp" # file path to a shapefile with outline of Denmark



#removal_threshold = 0.5 # when plotting, define this threshold to remove values below the threshold from the figure
#extracted_field["values"] = remove_values_below(extracted_field["values"], removal_threshold)

# make a plot!
#coordinates=(7,15,54.5,58)
#nwp_plot(extracted_field['values'], extracted_field['lons'], extracted_field['lats'], "Rainfall field %2d - %s"%(extracted_field['date'],file_grib[-4:len(file_grib)]),world_map_file,coordinates,file_grib)
#nwp_plot(extracted_field['values'], lons, lats, "Rainfall field %2d"%(extracted_field['date']),world_map_file2)


#Myfiles=[i for i in glob.glob("./2021072612/*") if len(i)<17]


    
#open_n_plot(Myfiles,0.5,world_map_file,coordinates)


#frames=[]
#imgs=glob.glob("./Pics/*.png")
    

#for i in imgs:
#    new_frame=Image.open(i)
#    frames.append(new_frame)
    


#imageio.mimsave('./Pics/png_to_gif.gif',frames,'GIF',duration=6)



