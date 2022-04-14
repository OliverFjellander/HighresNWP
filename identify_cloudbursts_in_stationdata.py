#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:58:26 2022

@author: jope
"""


# aim of this script is to load hourly precip data from rain gauge stations,
# identify dates and station locations for all days with cloudbursts,
# where a cloudburst here is defined as >15 mm/hour (rather than 15mm/30min)
import time
start=time.time()
import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

# I've tried getting background maps with these packages, but it wasn't as nice as "c"ontextily"
#import cartopy.crs as ccrs
#from cartopy.io.img_tiles import OSM

import contextily as cx
import geopandas as gpd
from shapely.geometry import Point



#########################################################################################################
####### Identify days where rainfall has exceeded XX millimeters over a time window of YY hours #########
#########################################################################################################


os.chdir("C:/Users/olive/Desktop/Speciale/Kode") 

# data with hourly rain gauge observations (DMI's new rain gauge network going back to 2011)
filepath = "./Rain_gauge/NotInterpolated/view_basis_hourly_dk_precipitation_sum.csv"

stationdata = pd.read_csv(filepath, sep=",")

stationdata_2021=stationdata[stationdata['timeobs'].str.contains("2021")]

# function that finds days where a rain gauge has measured more than XX mm in a time window of YY hours.
# If longterm_saturation = True, then there is a filter that removes periods where the large amounts of 
# rain are caused by a one or a couple of big events. This is to ensure that I only identify periods where
# there has been long-term rainfall that may cause saturation excess floods, which requires rainfall over several days
# and not just one or two big events within a month

def locate_pluvial_extremes_days_per_station(stationdata_df, rain_threshold, time_window, longterm_saturation=False):
    
    unique_station_ids = stationdata_df["statid"].unique()
    
    # allocate space in lists for station id-numbers and days/months 
    statid_list = []
    timeobs_list = []
       
    
    for station in unique_station_ids:
        # check if accumulated precipitation exceeds the user-specified threshold
        temp_df = stationdata_df[stationdata_df["statid"] == station]
        temp_df["rolling_sum"] = temp_df["precipitation_sum"].rolling(time_window).sum()
        if longterm_saturation==True:
            temp_df["rolling_24h_sum"] = temp_df["precipitation_sum"].rolling(24).sum()
            temp_df["rolling_dailymax"] = temp_df["rolling_24h_sum"].rolling(time_window).max()
            temp_df["maxsum_ratio"] = temp_df["rolling_dailymax"]/temp_df["rolling_sum"]
            
        temp_cloudbursts = temp_df[temp_df["rolling_sum"] >= rain_threshold]
        
        if longterm_saturation==True:
            temp_cloudbursts = temp_df[(temp_df["rolling_sum"] >= rain_threshold) * (temp_df["maxsum_ratio"] < 0.167)]
        
        temp_cloudburst_days = temp_cloudbursts["timeobs"].str[:10]
        temp_cloudburst_days = temp_cloudburst_days.drop_duplicates()
        
        # add station id numbers and day/month information to the list
        statid_list.extend([station]*len(temp_cloudburst_days))
        timeobs_list.extend(temp_cloudburst_days)
    
    # setup pretty dataframe
    cloudburst_station_day = pd.DataFrame({'timeobs': timeobs_list,
                                           'statid': statid_list})
    # convert to proper time formats in pandas and sort the data according to time
    cloudburst_station_day["timeobs"] = pd.to_datetime(cloudburst_station_day["timeobs"], utc=True)
    cloudburst_station_day = cloudburst_station_day.sort_values(by=["timeobs"]).reset_index(drop=True)
    
    return(cloudburst_station_day)


# days with cloudbursts. Here defined as 15mm / 1 hour, since the smallest time steps in the data are 1 hour,
# and I thus can't check for the standard cloudburst definition (15mm / 30 minutes)
cloudburst_days_2011_2021 = locate_pluvial_extremes_days_per_station(stationdata_2021, 15, 1)
# days with "severe rain" (Danish: "kraftig regn") with DMI's standard definition of 24mm over 6 hours.
severerain_days_2011_2021 = locate_pluvial_extremes_days_per_station(stationdata_2021, 24, 6)

# "saturation" periods are here defined as 150 mm over 30 days, with the filter for single event impacts on.
saturation_days_2011_2021 = locate_pluvial_extremes_days_per_station(stationdata_2021, 150, 24*30, longterm_saturation=True)
saturation_days_2011_2021["timeobs"] = pd.to_datetime(saturation_days_2011_2021["timeobs"].astype(str).str[:7], utc=True)
saturation_days_2011_2021 = saturation_days_2011_2021.drop_duplicates().reset_index(drop=True)


##########################################################################
############### Add station metadata to the data frames ##################
##########################################################################


# include metadata on rain gauge station name, lat and long coordinates
metadata_path = "./Rain_gauge/NotInterpolated/statcat_metadata.csv"
metadata = pd.read_csv(metadata_path, sep=",")


def add_station_metadata(cloudburst_days_df, metadata_df):
    # find unique station ids
    unique_station_ids = np.sort(cloudburst_days_df["statid"].unique())
    # convert station ids to string format
    unique_station_ids_metadataformat = unique_station_ids.astype(str)
    # only select first four numbers in station id and convert those to an integer (the metadata station ids are only the first 4 numbers)
    unique_station_ids_metadataformat = [int(s[:4]) for s in unique_station_ids_metadataformat]
    
    # which station entries in the metadata file are part of the set of rain gauge station ids
    metadata_stations = metadata_df[np.in1d(metadata_df["statid"], unique_station_ids)]
    
    # convert station "start_time" to pandas datetime
    metadata_stations["start_time"] = pd.to_datetime(metadata_stations["start_time"], utc=True)
    # stations that have an end_time gets those times converted to pandas datetime
    metadata_stations["end_time"][metadata_stations["end_time"] != "infinity"] = pd.to_datetime(metadata_stations["end_time"][metadata_stations["end_time"] != "infinity"], utc=True)
    # stations that don't have an end_time defined (instead they have value "infinity") gets now as their end_time value
    metadata_stations["end_time"][metadata_stations["end_time"] == "infinity"] = pd.Timestamp.now(tz="utc")
    
    # allocate space in dataframe for three new metadata variables
    cloudburst_days_df["station_name"] = np.nan
    cloudburst_days_df["lat"] = np.nan
    cloudburst_days_df["long"] = np.nan
    
    
    for i in range(len(cloudburst_days_df)):
        # a temporary dataframe with stations that have measured a cloudburst on a given day (gets overwritten every iteration)
        temp_station_metadata = metadata_stations[metadata_stations["statid"] == cloudburst_days_df["statid"][i]]
        # if a station has multiple entries in the metadata (e.g. if it has been moved to another location), find out which metadata entry that is valid for the specific cloudburst day
        if len(temp_station_metadata)>1:
            temp_starttime = cloudburst_days_df["timeobs"][i] > temp_station_metadata["start_time"]
            temp_endtime = cloudburst_days_df["timeobs"][i] < temp_station_metadata["end_time"]
            temp_station_metadata = temp_station_metadata[temp_starttime * temp_endtime]
        
        # add the three metadata variables to the original dataframe
        cloudburst_days_df["station_name"][i] = temp_station_metadata["name"].reset_index(drop=True)[0]
        cloudburst_days_df["lat"][i] = temp_station_metadata["lat"].reset_index(drop=True)[0]
        cloudburst_days_df["long"][i] = temp_station_metadata["long"].reset_index(drop=True)[0]
    
    return(cloudburst_days_df)


cloudburst_days_2011_2021 = add_station_metadata(cloudburst_days_2011_2021, metadata)
severerain_days_2011_2021 = add_station_metadata(severerain_days_2011_2021, metadata)
saturation_days_2011_2021 = add_station_metadata(saturation_days_2011_2021, metadata)



##########################################################################
############### print csv-files with dates for different events ##########
##########################################################################


def print_file_with_nice_dates(df, filepath, monthstep=False):
    df_printfile = df.copy()
    if monthstep==False:
        df_printfile["timeobs"] = df_printfile["timeobs"].dt.strftime("%Y-%m-%d")
    else:
        df_printfile["timeobs"] = df_printfile["timeobs"].dt.strftime("%Y-%m")
        
    df_printfile.to_csv(filepath, sep=",", header=True, index=False)


print_file_with_nice_dates(cloudburst_days_2011_2021, "./cloudburst_days_2011_2021.csv")
print_file_with_nice_dates(severerain_days_2011_2021, "./severerain_days_2011_2021.csv")
print_file_with_nice_dates(saturation_days_2011_2021, "./saturation_days_2011_2021.csv", monthstep=True)




##########################################################################
###### make pdf-files with plots of events for each identified day #######
##########################################################################


# Load background map from an online repository with the package "contextily"
west, south, east, north = (8, 54, 15.3, 58)

dk_img, dk_ext = cx.bounds2img(west, south, east, north,
                               ll=True,
                               source=cx.providers.Stamen.Toner,
                               zoom=11)

fig, ax = plt.subplots(1, figsize=(10,10))
ax.imshow(dk_img, extent=dk_ext)

   
    
# turn the standard pandas dataframes into geodataframes for later map-plotting   
def make_geodataframe_with_geometry(df):
    geometry = [Point(xy) for xy in zip(df["long"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry = geometry)
    gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs(epsg=3857)
    return(gdf)

cloudburst_days_2011_2021 = make_geodataframe_with_geometry(cloudburst_days_2011_2021)
severerain_days_2011_2021 = make_geodataframe_with_geometry(severerain_days_2011_2021)
saturation_days_2011_2021 = make_geodataframe_with_geometry(saturation_days_2011_2021)


### common pdf file with plots of both cloudbursts and severe rain
sorted_cloudburst_days = sorted(cloudburst_days_2011_2021["timeobs"].unique())
sorted_severerain_days = sorted(severerain_days_2011_2021["timeobs"].unique())

#list of days with cloudbursts and/or severe rain
sorted_days = sorted(list(set(sorted_severerain_days + sorted_cloudburst_days)))

# Plot cloudbursts + severe rain in same figure:
with matplotlib.backends.backend_pdf.PdfPages("./Cloudbursts_Severerain_days_2011_2021.pdf") as pdf:
    
    for i in range(len(sorted_days)):
    #for i in range(5):    
        temp_cloudbursts = cloudburst_days_2011_2021[cloudburst_days_2011_2021["timeobs"]==sorted_days[i]]
        
        temp_severerain = severerain_days_2011_2021[severerain_days_2011_2021["timeobs"]==sorted_days[i]]


        fig, ax = plt.subplots(1, figsize=(10,10))
        ax.imshow(dk_img, extent=dk_ext)
        temp_severerain.plot(ax=ax, color="blue", label="Severe rain (> 24mm/6h)",marker='x')
        temp_cloudbursts.plot(ax=ax, color="red", label="Cloudburst (> 15mm/1h)",marker='+')
        ax.legend()
        ax.set_title(sorted_days[i].strftime("%d-%b-%Y"))
        plt.close()
        pdf.savefig(fig, dpi=50)
        
        print(i)
     
    
    
    
# plot saturation rain months
with matplotlib.backends.backend_pdf.PdfPages("./longperiodsofrain_2011_2021.pdf") as pdf:
    
    sorted_saturation_days = sorted(saturation_days_2011_2021["timeobs"].unique())
    
    for i in range(len(sorted_saturation_days)):
    #for i in range(5):
        temp_saturation = saturation_days_2011_2021[saturation_days_2011_2021["timeobs"]==sorted_saturation_days[i]]

        fig, ax = plt.subplots(1, figsize=(10,10))
        ax.imshow(dk_img, extent=dk_ext)
        temp_saturation.plot(ax=ax, color="red", label="Long wet period (> 150mm/30 days) w/o cloudbursts")
        ax.set_title(temp_saturation["timeobs"].iloc[0].strftime("%b-%Y"))
        plt.close()
        pdf.savefig(fig, dpi=50)
        
        print(i)


print("Time gone:",time.time()-start)


#histogram

plt.hist((severerain_days_2011_2021['timeobs'],cloudburst_days_2011_2021['timeobs'],saturation_days_2011_2021['timeobs']),stacked=True,bins=190) #There are 190 days between
plt.legend(["Cloudburst","Severerain","Saturation"],loc='best')
plt.xlabel('Date')
plt.ylabel('Events')
plt.show()


