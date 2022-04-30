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
import pdb
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Rectangle
import collections
from rasterstats import zonal_stats
import pysteps.verification.spatialscores as pvs
import pysteps.verification.salscores as pvs_sal
#from skimage.feature import peak_local_max
from pygeoprocessing import zonal_statistics

os.chdir("C:/Users/olive/Desktop/Speciale/Kode")

#Finding missing data in radar data
# def datesearch(month):
#     m=[31,28,31,30,31,30,31,31,30,31,30,31] #days in a month
#     date_search=[]
#     for i in np.arange(1,m[int(month)+1]+1,1):
#         for j in np.arange(0,24,1):
#             #for k in np.arange(0,60,1):
#                 if len(str(i))<=1:
#                     i=str(0)+str(i)
#                 else:
#                     i=str(i)
#                 if len(str(j))<=1:
#                     j=str(0)+str(j)
#                 else:
#                     j=str(j)
#                 #if len(str(k))<=1:
#                 #    k=str(0)+str(k)
#                 #else:
#                 #    k=str(k)
#                 #if len(str(month))<=1:
#                 #    mo=str(0)+str(month)
#                 #else:
#                 #    mo=str(month)
#                 date_search.append(str(2021)+month+i+j)
#     return date_search


####################################

def plot_together(rain_array,NWP_data, radar_lons, radar_lats, nwp_lons, nwp_lats, world_map_file, plot_title_radar,plot_title_nwp,save_name,save_path):
    
    world_map = gpd.read_file(world_map_file)
    
    # create custom color map
    cmap = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
    #boundaries = [0, .5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
    boundaries = [0, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    
    # Make plot
    #world_map.plot(facecolor="lightgrey")
    fig,axs=plt.subplots(1,2,sharex=True,sharey=True,gridspec_kw={"width_ratios":[1,1.09]},figsize=(15,15))
    fig.text(0.5,0.04,"Longitude",ha='center')
    fig.text(0.04,0.5,"Latitude",va='center',rotation='vertical')
    #fig.tight_layout(pad=3.0)
    world_map.plot(facecolor="lightgrey",ax=axs[0])
    pcm=axs[0].pcolor(radar_lons, radar_lats, rain_array, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[0].set_xlim(7,15)
    axs[0].set_ylim(54.5,58)
    axs[0].title.set_text(plot_title_radar)
    world_map.plot(facecolor="lightgrey",ax=axs[1])
    pcm=axs[1].pcolor(nwp_lons, nwp_lats, NWP_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[1].set_xlim(7,15)
    axs[1].set_ylim(54.5,58)
    axs[1].title.set_text(plot_title_nwp)
    plt.tick_params('y',labelleft=False)
    #axs[1].set_yticks([])
    cbar = fig.colorbar(pcm,ax=axs[1],fraction=0.040, pad=0.04)
    cbar.ax.set_ylabel('Rainfall intensity [mm/h]', rotation=90)
    plt.subplots_adjust(wspace=0.05,hspace=0)
    plt.show()
    #save_name=save_path%str(save_name[-1:])[-17:-7]
    #if os.path.isfile(save_name):
    #    os.remove(save_name)
    #plt.savefig(save_name,bbox_inches='tight')
    #plt.close()

def plot_all_tst(radar_data,radar_data1,radar_data2,nwp_data,nwp_data1,nwp_data2,nea_data,nea_data1,nea_data2, radar_lons, radar_lats, nwp_lons, nwp_lats, nea_lons, nea_lats, world_map_file, plot_title_radar,plot_title_radar1,plot_title_radar2,plot_title_nwp,plot_title_nwp1,plot_title_nwp2,plot_title_nea,plot_title_nea1,plot_title_nea2,save_name,timestep,preciptype):
    
    world_map = gpd.read_file(world_map_file)
    
    # create custom color map
    cmap = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
    #boundaries = [0, .5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
    boundaries = [0, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    
    # Make plot
    #world_map.plot(facecolor="lightgrey")
    fig,axs=plt.subplots(3,3,sharex=True,sharey=True,figsize=(10,10),gridspec_kw={"width_ratios":[1,1,1]},constrained_layout=True)
    fig.text(0.5,0.1,"Longitude",ha='center')
    fig.text(0.06,0.5,"Latitude",va='center',rotation='vertical')
    #fig.tight_layout(pad=3.0)
    world_map.plot(facecolor="lightgrey",ax=axs[0,0])
    pcm=axs[0,0].pcolor(radar_lons, radar_lats, radar_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[0,0].set_xlim(7,15)
    axs[0,0].set_ylim(54.5,58)
    world_map.plot(facecolor="lightgrey",ax=axs[1,0])
    pcm=axs[1,0].pcolor(radar_lons, radar_lats, radar_data1, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[1,0].set_xlim(7,15)
    axs[1,0].set_ylim(54.5,58)
    world_map.plot(facecolor="lightgrey",ax=axs[2,0])
    pcm=axs[2,0].pcolor(radar_lons, radar_lats, radar_data2, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[2,0].set_xlim(7,15)
    axs[2,0].set_ylim(54.5,58)
    axs[0,0].title.set_text(plot_title_radar)
    axs[1,0].title.set_text(plot_title_radar1)
    axs[2,0].title.set_text(plot_title_radar2)
    world_map.plot(facecolor="lightgrey",ax=axs[0,1])
    pcm=axs[0,1].pcolor(nwp_lons, nwp_lats, nwp_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[0,1].set_xlim(7,15)
    axs[0,1].set_ylim(54.5,58)
    world_map.plot(facecolor="lightgrey",ax=axs[1,1])
    pcm=axs[1,1].pcolor(nwp_lons, nwp_lats, nwp_data1, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[1,1].set_xlim(7,15)
    axs[1,1].set_ylim(54.5,58)
    world_map.plot(facecolor="lightgrey",ax=axs[2,1])
    pcm=axs[2,1].pcolor(nwp_lons, nwp_lats, nwp_data2, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[2,1].set_xlim(7,15)
    axs[2,1].set_ylim(54.5,58)
    axs[0,1].title.set_text(plot_title_nwp)
    axs[1,1].title.set_text(plot_title_nwp1)
    axs[2,1].title.set_text(plot_title_nwp2)
    axs[0,1].tick_params(axis='both',labelleft=False,left=False)
    axs[1,1].tick_params(axis='both',labelleft=False,left=False)
    axs[2,1].tick_params('y',labelleft=False,left=False)
    #axs[1].set_yticks([])
    world_map.plot(facecolor="lightgrey",ax=axs[0,2])
    pcm=axs[0,2].pcolor(nea_lons, nea_lats, nea_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[0,2].set_xlim(7,15)
    axs[0,2].set_ylim(54.5,58)
    axs[0,2].title.set_text(plot_title_nea)
    axs[1,2].title.set_text(plot_title_nea1)
    axs[2,2].title.set_text(plot_title_nea2)
    axs[0,2].tick_params(axis='both',labelleft=False,left=False)
    world_map.plot(facecolor="lightgrey",ax=axs[1,2])
    pcm=axs[1,2].pcolor(nea_lons, nea_lats, nea_data1, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[1,2].set_xlim(7,15)
    axs[1,2].set_ylim(54.5,58)
    axs[1,2].tick_params(axis='both',labelleft=False,left=False)
    world_map.plot(facecolor="lightgrey",ax=axs[2,2])
    pcm=axs[2,2].pcolor(nea_lons, nea_lats, nea_data2, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[2,2].set_xlim(7,15)
    axs[2,2].set_ylim(54.5,58)
    axs[2,2].tick_params('y',labelleft=False,left=False)
    #cbar = fig.colorbar(pcm,ax=axs[:,2],fraction=0.040, pad=0.04)
    cax=plt.axes([0.92,0.15,0.020,0.7])
    cbar = fig.colorbar(pcm,ax=axs[2,:],cax=cax,shrink=1.0)
    cbar.ax.set_ylabel('Rainfall intensity [mm/h]')#, rotation=90)
    plt.subplots_adjust(wspace=0.05,hspace=0.0)
    plt.show()
    

def plot_all(radar_data,nwp_data,nea_data, radar_lons, radar_lats, nwp_lons, nwp_lats, nea_lons, nea_lats, world_map_file, plot_title_radar,plot_title_nwp,plot_title_nea,save_name,timestep,preciptype):
    
    world_map = gpd.read_file(world_map_file)
    
    # create custom color map
    cmap = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
    #boundaries = [0, .5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
    boundaries = [0, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    
    # Make plot
    #world_map.plot(facecolor="lightgrey")
    fig,axs=plt.subplots(1,3,sharex=True,sharey=True,gridspec_kw={"width_ratios":[1,1,1.09]},figsize=(16,8))
    fig.text(0.5,0.2,"Longitude",ha='center')
    fig.text(0.06,0.5,"Latitude",va='center',rotation='vertical')
    #fig.tight_layout(pad=3.0)
    world_map.plot(facecolor="lightgrey",ax=axs[0])
    pcm=axs[0].pcolor(radar_lons, radar_lats, radar_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[0].set_xlim(7,15)
    axs[0].set_ylim(54.5,58)

    axs[0].title.set_text(plot_title_radar)
    world_map.plot(facecolor="lightgrey",ax=axs[1])
    pcm=axs[1].pcolor(nwp_lons, nwp_lats, nwp_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[1].set_xlim(7,15)
    axs[1].set_ylim(54.5,58)

    axs[1].title.set_text(plot_title_nwp)
    axs[1].tick_params('y',labelleft=False,left=False)
    #axs[1].set_yticks([])
    world_map.plot(facecolor="lightgrey",ax=axs[2])
    pcm=axs[2].pcolor(nea_lons, nea_lats, nea_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[2].set_xlim(7,15)
    axs[2].set_ylim(54.5,58)
    axs[2].title.set_text(plot_title_nea)
    axs[2].tick_params('y',labelleft=False,left=False)
    cbar = fig.colorbar(pcm,ax=axs[2],fraction=0.040, pad=0.04)
    cbar.ax.set_ylabel('Rainfall intensity [mm/h]', rotation=90)
    plt.subplots_adjust(wspace=0.05,hspace=0)
    plt.show()
    #current=os.getcwd()
    #os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Pics/%s"%preciptype) 
    #newpath=r'./%s/'%save_name[-6:-2]
    #if not os.path.exists(newpath):
    #    os.makedirs(newpath)
    #os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Pics/%s/%s"%(preciptype,newpath[-6:-1]))
    #save_name=os.getcwd()+"/"+str(save_name[-2:])+"_"+str(timestep)
    #if os.path.isfile(save_name):
    #    os.remove(save_name)
    #plt.savefig(save_name,bbox_inches='tight')
    #plt.close()
    #os.chdir(current)  



def total_precipitation(data,data_nwp):
    total=0
    total_nwp=0
    for i in range(0,len(data)):
        total=total+np.nansum(data[i])
        total_nwp=total_nwp+np.nansum(data_nwp[i+1])
    return total,total_nwp


def saveFSS(FSS,savepath=os.getcwd()):    
    workingdir=os.getcwd()
    os.chdir(savepath)
    np.savetxt("FSS.csv",FSS,delimiter=",",fmt="%s")
    with open("FSS.csv","w") as f:
        writer=csv.writer(f)
        writer.writerows(FSS)
    os.chdir(workingdir)



def confusion_mat(twod_radar,twod_nwp):
    #Confusion matrix [1,1]=TP (hits), [0,0]=TN (correct nulls), [0,1]=FP (false alarm) , [1,0]=FN (miss)
    cm_final=[np.array([[0,0],[0,0]],dtype=int) for _ in range(len(twod_radar))]
    ETS=[[] for _ in range(len(twod_radar))]
    PSS=[[] for _ in range(len(twod_radar))]
    FBI=[[] for _ in range(len(twod_radar))]
    TPR=[[] for _ in range(len(twod_radar))]
    TS=[[] for _ in range(len(twod_radar))]
    FPR=[[] for _ in range(len(twod_radar))]
    hits=[[] for _ in range(len(twod_radar))]
    misses=[[] for _ in range(len(twod_radar))]
    total=[[] for _ in range(len(twod_radar))]
    for i in range(len(twod_radar)):
        for j in range(len(twod_radar[i])):
            cm=confusion_matrix((twod_radar[i][j]>0)*1,(twod_nwp[i][j]>0)*1)
            if np.shape(cm)==(1,1):
               cm_final[i][0,0]=cm_final[i][0,0]+cm[0,0]
            else:    
                cm_final[i]=cm_final[i]+cm
        a=cm_final[i][1,1]
        b=cm_final[i][0,1]
        c=cm_final[i][1,0]
        d=cm_final[i][0,0]
        a_ref=((a+b)*(a+c))/(a+b+c+d)
        ETS[i]=(a-a_ref)/((a-a_ref)+b+c)
        PSS[i]=((a*d)-(b*c))/((a+c)*(b+d))
        FBI[i]=(a+b)/(c+d)
        TPR[i]=a/(a+c)
        TS[i]=a/(a+c+b)
        FPR[i]=b/(b+d)
        hits[i]=a
        misses[i]=c
        total[i]=a+b+c+d
    return cm_final,ETS,PSS,FBI,TPR,TS,FPR,hits,misses,total

    
# def produce_zonalstat(data_radar,radar_lons,radar_lats,files_radar,data_nwp,nwp_lon,nwp_lat,files_nwp):
#     zs_radar=[]
#     zs_nwp=[]
#     for i in range(0,len(data_radar)):
#         tif_path_nwp="./Tiff_files/"+files_nwp[i+1][-3:]+'_nwp'+'.tif'
#         data_to_raster_NWP(data_nwp[i+1],nwp_lon,nwp_lat,tif_path_nwp)
        
#         tif_path_radar ="./Tiff_files/"+files_radar[i][0][-15:-5] + "_radar" + ".tif" # file path for a tif file that will be generated
#         data_to_raster_RADAR(data_radar[i],radar_lons,radar_lats,tif_path_radar)
#         print("done")
#         #raster_radar=rasterio.open(tif_path_radar,masked=False)
#         #raster_nwp=rasterio.open(tif_path_nwp,masked=False)
        
#         #start=time.time()
#         zs_radar.append(zonal_statistics((tif_path_radar,1),"C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI.shp",ignore_nodata=False,polygons_might_overlap=False))
#         #zs_radar.append(zonal_statistics((tif_path_radar,1),"C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_northernzealand_DMI.shp",ignore_nodata=False,polygons_might_overlap=False))
#         #print(time.time()-start)
#         zs_nwp.append(zonal_statistics((tif_path_nwp,1),"C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI_NWP.shp",ignore_nodata=False,polygons_might_overlap=False))
#         #zs_nwp.append(zonal_statistics((tif_path_nwp,1),"C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_northernzealand_nwp_DMI.shp",ignore_nodata=False,polygons_might_overlap=False))
#         #print(time.time()-start) 
#     return zs_radar,zs_nwp

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
        #print("done")
        #raster=rasterio.open(tif_path,masked=False)
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

# def zone_to_2d(z_stat_radar,z_stat_nwp):
#     radar_grid_2d=[[] for _ in range(len(z_stat_nwp))]
#     nwp_grid_2d=[[] for _ in range(len(z_stat_radar))]
#     for i in range(0,len(z_stat_radar)):
#         radar_grid_2d[i]=np.reshape(extract_zonalstat(z_stat_radar[i],25),(125,157))
#         nwp_grid_2d[i]=np.reshape(extract_zonalstat(z_stat_nwp[i],10),(125,157))
#         #radar_grid_2d[i]=np.reshape(extract_zonalstat(z_stat_radar[i]),(49,40))
#         #nwp_grid_2d[i]=np.reshape(extract_zonalstat(z_stat_nwp[i]),(49,40))
#         radar_grid_2d[i][radar_grid_2d[i]==0]=np.nan
#         nwp_grid_2d[i][nwp_grid_2d[i]==0]=np.nan
#     return radar_grid_2d,nwp_grid_2d


def zone_to_2d(z_stat):
    grid_2d=[[] for _ in range(len(z_stat))]

    for i in range(0,len(z_stat)):
        grid_2d[i]=np.reshape(extract_zonalstat(z_stat[i]),(156,184))
        #grid_2d[i]=np.reshape(extract_zonalstat(z_stat[i]),(49,40))
        grid_2d[i][grid_2d[i]==0]=np.nan
    return grid_2d

def precipitation_days(days_found):
    dates=[]
    for i in range(0,len(days_found)):
        dates.append(int(days_found[i,0][2:4]+days_found[i,0][5:7]+days_found[i,0][8:10]+days_found[i,1][:2]))
    dates=np.unique(dates)

    nwp_files=glob.glob("./25grid/NWP750/*")
    nwp_int=[]
    for i in range(0,len(nwp_files)):
        nwp_int.append(int(nwp_files[i][-12:-4]))

    final_dates=[]
    for i in range(0,len(dates)):
        temp=dates[i]-nwp_int
        temp[temp<=0]=1000
        if any(temp[temp<7]):
            final_dates.append(str(nwp_int[np.argmin(temp)]))
    return final_dates


def Fractional_skillscore(zs_radar,zs_nwp,domain,intensity=0,percentile=0,plot=False):
    avg_fss=[[] for _ in range(len(zs_radar))]
    fssrandom=0
    threshold_radar=[]
    qt_nwp=[[] for _ in range(len(zs_nwp))]
    for i in range(0,len(zs_radar)):
        #start=time.time()
        radar_grid_2d=np.reshape(extract_zonalstat(zs_radar[i]),(156,184))
        nwp_grid_2d=np.reshape(extract_zonalstat(zs_nwp[i]),(156,184))
        
        #IF QUANTILE IS DEFINED
        if percentile!=0.0:
            
            threshold=np.quantile(radar_grid_2d,percentile)
            qt=(nwp_grid_2d>np.quantile(np.nan_to_num(nwp_grid_2d),percentile))*1*(threshold)
            threshold_radar.append(threshold_radar)
            fssrandom+=np.sum((radar_grid_2d>0)*1)
            qt_nwp[i]=qt
            
            if plot==False:
                avg_fss[i]=(pvs.fss(qt,radar_grid_2d,threshold,domain))
            else:
                for scale in range(0,domain):
                    avg_fss[i].append(pvs.fss(qt,radar_grid_2d,threshold,scale))
        
        elif intensity!=0.0:
            nwp_grid_2d[nwp_grid_2d==0]=np.nan
            radar_grid_2d[radar_grid_2d==0]=np.nan
        
            fssrandom+=np.sum((radar_grid_2d>0)*1)
        
            #print(time.time()-start)
            if plot==False:
                avg_fss[i]=(pvs.fss(nwp_grid_2d,radar_grid_2d,intensity,domain))
            else: 
                for scale in range(0,domain):
                    avg_fss[i].append(pvs.fss(nwp_grid_2d,radar_grid_2d,intensity,scale))
        else:
            print("Error. Intensity or percentile needs to be defined")
            break
        
    if plot==False:
        return avg_fss
    else:
        fssrandom=np.repeat(fssrandom/(np.size(radar_grid_2d)*len(zs_radar)),domain)
    
        FSS_mean=[np.mean(k) for k in zip(*avg_fss)]
        t_dist=2.306
        #ci_low=[np.mean(k)-((np.std(k)/np.sqrt(len(avg_fss)))*t_dist) for k in zip(*avg_fss)]
        #ci_high=[np.mean(k)+((np.std(k)/np.sqrt(len(avg_fss)))*t_dist) for k in zip(*avg_fss)]
        low=[np.nanmin(k) for k in zip(*avg_fss)]
        high=[np.nanmax(k) for k in zip(*avg_fss)]
        plt.figure()
        plt.fill_between(np.arange(0,domain,1)*2.5,low,high,alpha=0.5,color="grey")
        plt.plot(np.arange(0,domain,1)*2.5,FSS_mean)
        plt.plot(np.arange(0,domain,1)*2.5,fssrandom)
        plt.plot(np.arange(0,domain,1)*2.5,0.5+fssrandom/2)
        plt.ylim(0,1.1)
        plt.xlabel("Horizontal Scale [km]")
        plt.ylabel("Fractional Skill Score")
        plt.legend(["Span","0.75km","FSS random","FSS useful"],loc="lower right")
        plt.show()
    
        plt.figure()
        for i in range(0,len(avg_fss)):
            plt.plot(np.arange(0,domain,1)*2.5,avg_fss[i])
        plt.plot(np.arange(0,domain,1)*2.5,fssrandom)
        plt.plot(np.arange(0,domain,1)*2.5,0.5+fssrandom/2)
        plt.ylim(0,1.1)
        plt.xlabel("Horizontal Scale [km]")
        plt.ylabel("Fractional Skill Score")
        plt.legend(["1","2","3","4","5","6","7","8","9","FSS random","FSS uniform"],loc=0)
        plt.show()
    return avg_fss

#Intensity and spatial array should be given in some logspace array.
#spatial scale should go from high to low, intensity array the other way around
def plot_spatial_threshold_matrix(intensity_array,spatial_array,zt_radar,zt_nwp):  
    fss=np.zeros((len(spatial_array),(len(intensity_array))))
    start=time.time()
    for i in range(0,len(spatial_array)):
        print(time.time()-start)
        domain=spatial_array[i]
        for j in range(0,len(intensity_array)):
            precipitation=intensity_array[j]
            fss[i,j]=np.mean(Fractional_skillscore(zt_radar,zt_nwp,domain,intensity=precipitation))

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    #cmap = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
    cmap = colors.ListedColormap(["orangered","orange","gold","yellow","limegreen","forestgreen"])
    boundaries = [0.0, 0.2, 0.4,0.6, 0.8,0.9, 1.0]
    norm=colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.pcolor(intensity_array, spatial_array, fss, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Fractional Skill Score (FSS)', rotation=90)
    plt.xticks(np.round(intensity_array))
    plt.yticks([10,50,100,200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_tick_params(which='minor', width=0) 
    plt.xlabel("Intensity [mm/hr]")
    plt.ylabel("Scale [km]")
    plt.show()

def produce_SAL(radar,nwp):
    #loc_dict={'minref':5,'maxref':100,'minsize':2,'mindis':5,'minmax': 10}
    loc_dict={'minref':3.0,'maxref':100,'minsize':1,'mindis':3,'minmax': 4,'mindiff':2}
    amp=[]
    struc=[]
    #struc=np.empty((0,len(radar)))
    loca=[]
    l1_tst=[]
    l_tst=[]
    for i in range(0,len(radar)):
        #print(pvs_sal.sal(nwp[i].transpose(1,0),radar[i].transpose(1,0),thr_factor=None,thr_quantile=None,tstorm_kwargs=loc_dict))
        l1=pvs_sal._sal_l1_param(nwp[i].transpose(1,0),radar[i].transpose(1,0))
        l2=pvs_sal._sal_l2_param(nwp[i].transpose(1,0),radar[i].transpose(1,0),thr_factor=None,thr_quantile=None,tstorm_kwargs=loc_dict)
        amp.append(pvs_sal.sal_amplitude(nwp[i].transpose(1,0),radar[i].transpose(1,0)))
        struc.append(pvs_sal.sal_structure(nwp[i].transpose(1,0),radar[i].transpose(1,0),tstorm_kwargs=loc_dict))
        loca.append(np.nansum((l1,l2)))
        print(l1,l2)
        
        #max_dis_tst=np.sqrt(((radar[i].transpose(1,0).shape[0]) ** 2) + ((radar[i].transpose(1,0).shape[1]) ** 2))
        #obs_r=pvs_sal._sal_weighted_distance(nwp[i].transpose(1,0),thr_factor=None,thr_quantile=None,tstorm_kwargs=loc_dict)/(np.nanmean(nwp[i]))
        #forc_r=pvs_sal._sal_weighted_distance(radar[i].transpose(1,0),thr_factor=None,thr_quantile=None,tstorm_kwargs=loc_dict)/(np.nanmean(radar[i]))
        #print(2 * ((abs(obs_r - forc_r)) / max_dis_tst))
        #l2_tst=(2 * ((abs(obs_r - forc_r)) / max_dis_tst))
        #l_tst.append(np.nansum(l1,l2_tst))

    return struc,amp,loca

def plot_SAL(s,a,l,title):
    plt.figure()
    cm=plt.cm.get_cmap('coolwarm')
    plt.axhline(0,0,color="black",alpha=0.5)
    plt.axvline(0,0,color="black",alpha=0.5)
    plt.axhline(np.median(a),0,color="red",alpha=0.5)
    plt.axvline(np.median(s),0,color="red",alpha=0.5)
    sc=plt.scatter(s,a,c=l,vmin=0,vmax=2,cmap=cm,zorder=2)
    for i in range(0,len(s)):
        plt.text(s[i],a[i],str(i))
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.xlabel("Structure")
    plt.ylabel("Amplitude")
    cbar=plt.colorbar(sc)
    cbar.ax.set_ylabel('Location', rotation=90)
    plt.title(title)
    plt.show()
    
