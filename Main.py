# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:31:43 2022

@author: olive
"""
import os
os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Scripts") # change directory to where the files are located!

import numpy as np

#import h5py
import geopandas as gpd

from pyproj import CRS
import glob
import time

import nwp750_read_plot
import radar_data_handling
import ClimateGrid_handling
import Other_functions
from nwp750_read_plot import *
from radar_data_handling import *
from plotting_functions import *
from ClimateGrid_handling import *
from Other_functions import *

import pickle
import gzip
start_start=time.time()

###########################################################################
###########################################################################
###########################################################################
os.chdir("C:/Users/olive/Desktop/Speciale/Kode") 
world_map_file = "C:/Users/olive/OneDrive/Desktop/Speciale/Kode/pygrib_functionality/pygrib_functionality/world_map_cut/world_map_background.shp" # file path to a shapefile with outline of Denmark

threshold_value=0.5
dmi_stere_crs = CRS("+proj=stere +ellps=WGS84 +lat_0=56 +lon_0=10.5666 +lat_ts=56") # raw data CRS projection
plotting_crs = 'epsg:4326' # the CRS projection you want to plot the data in

#FILES TO BE USED
###############################################################################
###############################################################################
#2.5km common grid
coor_4326=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/coor_4326.shp")
grid_4326=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/25grid_4326.shp")
date_time=str(2021072612)
#date_time=str(2021080708)
####################################################
#Retrieve txt.files produced by indentify_cloudbursts_in_stationdata:
extreme_days = np.genfromtxt("./extremerain.txt", dtype='str')
cloudburst_days=np.genfromtxt("./cloudburstdays.txt",dtype='str')
severe_days=np.genfromtxt("./severedays.txt",dtype='str')
rainy_days=np.genfromtxt("./rainydays.txt",dtype='str')

#Get extreme precipition events
#Extreme dates is the only that has been used in the report. However, the remaining has also been investigated
cloudburst_dates=precipitation_events(cloudburst_days)
extreme_dates=precipitation_events(extreme_days)
severe_dates=precipitation_events(severe_days)
rainy_dates=precipitation_events(rainy_days)
rainy_dates = [x for x in rainy_dates if x not in extreme_dates] 


####################################################
#NWP
#extract_tar('./NWP750/dk7502021070810.tar.gz')
coordinates_nwp=(7,15,54.5,58)
Myfiles_nwp=[i for i in glob.glob("./NWP750/%s/*"%date_time) if len(i)<24]
extracted_nwp_coor=Output_rain_NWP(Myfiles_nwp,threshold_value) 
extracted_nwp=unaccumulate(Myfiles_nwp,threshold_value,world_map_file,coordinates_nwp)

lons_nwp=extracted_nwp_coor[1]['lons']
lats_nwp=extracted_nwp_coor[1]['lats']


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



#movefiles("./Radar/Data0708/nowcast.dk.com.*/interpolated.*.h5","./Radar/Data0708/Interpolations") #Move interpolated files for all folders to store them in one place


#PLOTS TO BE USED IN THE REPORT
##############################################################################
####################Plot the 15 radar events#####################
#Plot Radar events:
lons_25=np.array(coor_4326['xcoor']).reshape((156,184))
lats_25=np.array(coor_4326['ycoor']).reshape((156,184))
radar_list=[]
title_list=[]
hours=[15,12,17,16,14,15,14,18,14,13,20,14,12,14,14]
dates=["20210516","20210517","20210528","20210708","20210725","20210726","20210731","20210806","20210809","20210810","20210815","20210816","20210817","20210911","20210915"]
for date in range(0,len(dates)):
    Myfiles_radar=[i for i in glob.glob("./Radar/2021-%s/*.%s*"%(dates[date][4:6],dates[date]))]
    Myfiles_radar_hr=np.array_split(Myfiles_radar,24) #The first is removed because the belongs to previous timeframe
    Radar_agg=aggregate_data(Myfiles_radar_hr)
    Radar_agg=remove_values_below(Radar_agg, threshold_value)
    radar_list.append(Radar_agg[hours[date]])
    title_list.append("Event %s \n%s/%s - %s:00 UTC"%(date+1,dates[date][4:6],dates[date][6:8],hours[date]+1))
    print(date)
    
radar_plot_all_events(radar_list,radar_lons_4326,radar_lats_4326,world_map_file,title_list)        


##############################################################################
####################Producing illustrations to the report#####################
    
    
Files_NEA=[i[-12:-4] for i in glob.glob("./25grid/NEA/*")] #what is available?
all_nwp=[i for i in glob.glob("./NWP750/pickled_data/pickled_data/*")]
Myfiles_nwp=[]
for i in range(0,len(all_nwp)):
    if all_nwp[i][-15:-7] in Files_NEA and all_nwp[i][-15:-7] in severe_dates:
        Myfiles_nwp.append(all_nwp[i])
    else:
        pass
    
date=Myfiles_nwp[16][-15:-7] # #11 is a good example as well as 16
date1=Myfiles_nwp[16][-15:-7]
        
file=Myfiles_nwp[16]
file1=Myfiles_nwp[16]
f=gzip.open(file,'rb')
f1=gzip.open(file1,'rb')
df_nwp=pickle.load(f,encoding='bytes')
df_nwp1=pickle.load(f1,encoding='bytes')
f.close()
f1.close()
df_nwp=remove_values_below(df_nwp,0.5)
df_nwp1=remove_values_below(df_nwp1,0.5)
df_nwp=df_nwp.transpose(2,0,1)  
df_nwp1=df_nwp1.transpose(2,0,1) 


Myfiles_radar=[i for i in glob.glob("./Radar/2021-%s/*.20%s*"%(date[2:4],date[:-2]))]
Myfiles_radar_hr=np.array_split(Myfiles_radar,24) #The first is removed because the belongs to previous timeframe
Radar_agg=aggregate_data(Myfiles_radar_hr)
Radar_agg=remove_values_below(Radar_agg, 0.5)


load_NEA=np.loadtxt("./25grid/NEA/nea_2d_%s.txt"%date)
load_NEA1=np.loadtxt("./25grid/NEA/nea_2d_%s.txt"%date1)

orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1] // 184, 184)
orig_nea1=load_NEA1.reshape(load_NEA1.shape[0],load_NEA1.shape[1] // 184, 184)

radar_format="Radar500 \n %s/%s/%s - %s:00 UTC"%(str('20')+date[:2],date[2:4],date[4:6],str(int(date[-2:])+2))
radar_format1="%s/%s/%s - %s:00 UTC"%(str('20')+date[:2],date[2:4],date[4:6],str(int(date[-2:])+4))
radar_format2="%s/%s/%s - %s:00 UTC"%(str('20')+date1[:2],date1[2:4],date1[4:6],str(int(date[-2:])+6))
nwp_format="DK750 \n %s/%s/%s - %s:00 + %s UTC"%(str('20')+date[:2],date[2:4],date[4:6],date[-2:],str(2))
nwp_format1="%s/%s/%s - %s:00 + %s UTC"%(str('20')+date[:2],date[2:4],date[4:6],date[-2:],str(4))
nwp_format2="%s/%s/%s - %s:00 + %s UTC"%(str('20')+date1[:2],date1[2:4],date1[4:6],date[-2:],str(6))
nea_format="NEA2500 \n %s/%s/%s - %s:00 + %s UTC"%(str('20')+date[:2],date[2:4],date[4:6],date[-2:],str(2))    
nea_format1="%s/%s/%s - %s:00 + %s UTC"%(str('20')+date[:2],date[2:4],date[4:6],date[-2:],str(4))    
nea_format2="%s/%s/%s - %s:00 + %s UTC"%(str('20')+date1[:2],date1[2:4],date1[4:6],date[-2:],str(6))    
plot_3x3(Radar_agg[int(date[-2:])+1],Radar_agg[int(date[-2:])+3],Radar_agg[int(date[-2:])+5],df_nwp[1],df_nwp[3],df_nwp[5],orig_nea[1],orig_nea[3],orig_nea[5],radar_lons_4326, radar_lats_4326,lons_nwp,lats_nwp,lons_25, lats_25,world_map_file,radar_format,radar_format1,radar_format2, nwp_format,nwp_format1,nwp_format2, nea_format,nea_format1,nea_format2, str(date),str(3),"cloudburst")


###############################################################################
###############################################################################
#Verification

###############################################################################
# Subjective Verification

produce_gifs(severe_dates,"severerain")
produce_gifs(rainy_dates,"precipitation")


###############################################
#Objective verification

# FSS
produce_FSS_matrix(extreme_dates, "NEA", "Extreme Precipitation") #Or rainy_dates and "Precipitation"
produce_FSS_matrix(extreme_dates, "DK750", "Extreme Precipitation")

temp_dk750, temp_nea = produce_fss_percentiles(extreme_dates, 0.99, plot=True)
scalemin_plot(temp_dk750,temp_nea,extreme_dates)

# FSS BOXPLOT
plot_fss_boxplot(extreme_dates)

# BOXPLOT LEAD TIMES
plot_fss_leadtime_boxplot(extreme_dates)

# SAL
struc, amp, loca = SAL_output(extreme_dates, "DK750", "Extreme rainfall") #OR NEA
plot_SAL_l(struc,amp,loca,"DK750") #or NEA

#To be used for making a table of SAL for the events 
struc = [np.nanmean(x) for x in struc]
amp = [np.nanmean(x) for x in amp]
loca = [np.nanmean(x) for x in loca]

#Finding SAL OBJECTS
#Objects. Also used is 21081006 at 9/3. 21081518 at 20/2. 21091512 at 12/0 (+1 for actual lead time)
objects(str(21072612),3,lons_25,lats_25)