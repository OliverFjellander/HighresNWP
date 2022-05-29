# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:33:43 2022

@author: olive
"""
import os
os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Scripts")

import nwp750_read_plot
import radar_data_handling
import ClimateGrid_handling
from nwp750_read_plot import *
from radar_data_handling import *
from ClimateGrid_handling import *
import numpy as np
import glob
import Image
import imageio

from pygeoprocessing import zonal_statistics


os.chdir("C:/Users/olive/Desktop/Speciale/Kode")


####################################

def precipitation_events(days_found):
    dates = []
    for i in range(0, len(days_found)):
        dates.append(int(days_found[i, 0][2:4]+days_found[i, 0]
                     [5:7]+days_found[i, 0][8:10]+days_found[i, 1][:2]))
    dates = np.unique(dates)

    nwp_files = glob.glob("./25grid/NWP750/*")
    nwp_int = []
    for i in range(0, len(nwp_files)):
        nwp_int.append(int(nwp_files[i][-12:-4]))

    final_dates = []
    for i in range(0, len(dates)):
        temp = dates[i]-nwp_int
        temp[temp <= 0] = 1000
        if any(temp[temp < 7]):
            final_dates.append(nwp_int[np.argmin(temp)])
    return final_dates



def produce_zonalstat(data,lons,lats,files,datatype):
    zs=[]
    for i in range(0,len(data)):
        if datatype=="NWP":    
            #tif_path="./Tiff_files/"+files[i+1][-3:]+'_nwp'+'.tif'
            tif_path="./Tiff_files/"+files[-15:-9]+"0"+str(i+1)+'_nwp'+'.tif'
            data_to_raster_NWP(data[i],lons,lats,tif_path)
            #data_to_raster_NWP(data[i+1],lons,lats,tif_path)
            zs.append(zonal_statistics((tif_path,1),"C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/25grid_4326.shp",ignore_nodata=False,polygons_might_overlap=False))
        elif datatype=="radar":
            tif_path ="./Tiff_files/"+files[i][0][-15:-5] + "_radar" + ".tif" # file path for a tif file that will be generated
            data_to_raster_RADAR(data[i],lons,lats,tif_path)
            zs.append(zonal_statistics((tif_path,1),"C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/25gridradar.shp",ignore_nodata=False,polygons_might_overlap=False))
        elif datatype=="raingauge":
            tif_path="./Tiff_files/"+files[i][-20:-9]+'_raingauge'+'.tif'
            data_array_to_raster_rg(data[i],tif_path)
            zs.append(zonal_statistics((tif_path,1),"C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/25gridraingauge.shp",ignore_nodata=False,polygons_might_overlap=False))
        elif datatype=="NWP25":
            tif_path="./Tiff_files/"+files[-19:-15]+"0"+str(i)+'_nwp25'+'.tif'
            data_to_raster_NWP(data[i],lons,lats,tif_path)
            zs.append(zonal_statistics((tif_path,1),"C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/25grid_4326.shp",ignore_nodata=False,polygons_might_overlap=False))

        else:
            print("Error. Type needs to be either NWP,radar or raingauge")
            break
    return zs

def extract_zonalstat(data_zonalstat):
    data_ts=np.empty((0,len(data_zonalstat)),np.float64)
    for i in range(0,len(data_zonalstat)):
        if np.isnan(data_zonalstat[i]['sum']):
            data_ts=np.append(data_ts,0.0)
        elif data_zonalstat[i]['count']==0:
            data_ts=np.append(data_ts,0.0)
        else:
            data_ts=np.append(data_ts,data_zonalstat[i]['sum']/data_zonalstat[i]['count'])
    return data_ts

def zone_to_2d(z_stat):
    grid_2d=[[] for _ in range(len(z_stat))]

    for i in range(0,len(z_stat)):
        grid_2d[i]=np.reshape(extract_zonalstat(z_stat[i]),(156,184))
        #grid_2d[i]=np.reshape(extract_zonalstat(z_stat[i]),(49,40))
        grid_2d[i][grid_2d[i]==0]=np.nan
    return grid_2d

def produce_gifs(dates,preciptype):
    start=time.time()
    os.chdir("C:/Users/olive/Desktop/Speciale/Kode")
    lons_25=np.array(coor_4326['xcoor']).reshape((156,184))
    lats_25=np.array(coor_4326['ycoor']).reshape((156,184))

    
    
    Files_NEA=[i[-12:-4] for i in glob.glob("./25grid/NEA/*")] #what is available?
    all_nwp=[i for i in glob.glob("./NWP750/pickled_data/pickled_data/*")]
    Myfiles_nwp=[]
    for i in range(0,len(all_nwp)):
        if all_nwp[i][-15:-7] in Files_NEA and all_nwp[i][-15:-7] in dates:
            Myfiles_nwp.append(all_nwp[i])
        else:
            pass
    for i in range(0,len(Myfiles_nwp)):
        date=Myfiles_nwp[i][-15:-7]
        
        file=Myfiles_nwp[i]
        save_name=Myfiles_nwp[i][-15:-7]
        f=gzip.open(file,'rb')
        df_nwp=pickle.load(f,encoding='bytes')
        f.close()
        df_nwp=remove_values_below(df_nwp,0.5)
        df_nwp=df_nwp.transpose(2,0,1)  


        Myfiles_radar=[i for i in glob.glob("./Radar/2021-%s/*.20%s*"%(date[2:4],date[:-2]))]
        if len(Myfiles_radar)<1439:
            continue
        else:
            Myfiles_radar_hr=np.array_split(Myfiles_radar,24) #The first is removed because the belongs to previous timeframe
            Radar_agg=aggregate_data(Myfiles_radar_hr)
            Radar_agg=remove_values_below(Radar_agg, 0.5)
        
        
        
        
        load_NEA=np.loadtxt("./25grid/NEA/nea_2d_%s.txt"%date)
        orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1] // 184, 184)

        for timestep in range(0,len(orig_nea)): 
            if (int(date[-2:])+timestep)>23:
                continue
            else:
                radar_format="Radar fields \n %s/%s/%s - %s:00 UTC"%(str('20')+date[:2],date[2:4],date[4:6],str(int(date[-2:])+timestep+1))
                nwp_format="DK750 \n %s/%s/%s - %s:00 + %s UTC"%(str('20')+date[:2],date[2:4],date[4:6],date[-2:],str(timestep+1))
                nea_format="NEA \n %s/%s/%s - %s:00 + %s UTC"%(str('20')+date[:2],date[2:4],date[4:6],date[-2:],str(timestep+1))    
                plot_all(Radar_agg[int(date[-2:])+timestep],df_nwp[timestep],orig_nea[timestep],radar_lons_4326, radar_lats_4326,lons_nwp,lats_nwp,lons_25, lats_25,world_map_file,radar_format, nwp_format, nea_format, str(date),str(timestep),preciptype)
        print("Time for %s:"%i,time.time()-start)
        make_gif("./Pics/%s/%s/*.png"%(preciptype,date[2:6]),'./Pics/%s/gifs/%s.gif'%(preciptype,date),0.4)
        


def make_gif(path_to_png,path_placing_gif,duration):
    frames=[]
    imgs=glob.glob(path_to_png)
    for i in imgs:
        new_frame=Image.open(i)
        frames.append(new_frame)
    imageio.mimsave(path_placing_gif,frames,'GIF',duration=duration)
    

# Calculate radar percentage covered of domain"
radar_percentage = np.loadtxt("./25grid/Radar/radar_20%s.txt" % "210915")
percentage = []
for i in range(6, 18):
    percentage.append(
        np.sum(radar_percentage[i] > 0)/len(radar_percentage[i])*100)
print(np.mean(percentage))