# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 09:14:17 2022

@author: oliver
"""
import pygrib
import numpy as np
import geopandas as gpd
import os
import pandas as pd
import glob
import pickle
import gzip
import shutil
import matplotlib.pyplot as plt

os.chdir("C:/Users/olive/Desktop/Speciale/Kode") # change directory to where the files are located!
########################functions###############################

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


def open_multiple(Data,removal_threshold):
    previous_accum=np.empty((630,700))
    rain=[[] for _ in range(len(Data)-1)]
    for i in range(0,len(Data)-1):
        open_file = pygrib.open(Data[i+1])
        parameters=open_file.read()
        attributes=read_parameter_info(parameters,58)
        attributes['values']=remove_values_below(attributes['values'],removal_threshold)
        values_plot=pd.DataFrame(attributes['values']).fillna(0)-previous_accum
        values_plot=values_plot.replace(0,np.nan)
        values_plot=remove_values_below(values_plot,removal_threshold)
        previous_accum=np.nan_to_num(np.array(attributes['values']))
        rain[i]=values_plot
    return rain 
###################################################################

threshold_value=0.5 #For removal
Myfiles_nwp=[i for i in glob.glob("./NWP750/2021072612/*") if len(i)<24]
extracted_nwp=open_multiple(Myfiles_nwp,threshold_value)

file_name="NWP750_%s" % (Myfiles_nwp[1][-14:-4])
pickle.dump(extracted_nwp, gzip.open(file_name, 'wb'))



#TESTING
file_name_tst="./NWP750/sNEA21050100_00_07.pkl.gz"
    
f=gzip.open(file_name_tst,'rb')
loaded_tst=pickle.load(f,encoding='bytes')
f.close()

loaded_tst=remove_values_below(loaded_tst,0.5)
tst=loaded_tst.transpose(2,0,1)


    





