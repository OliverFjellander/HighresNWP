# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:31:43 2022

@author: olive
"""
import os
# change directory to where the files are located!
os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Scripts")
from pylab import plot, legend
from scipy import ndimage
import pysteps.verification.salscores as pvs_sal
import pysteps.verification.spatialscores as pvs
import pysteps
from matplotlib import ticker 

import csv
from Other_functions import *
from ClimateGrid_handling import *
from radar_data_handling import *
from nwp750_read_plot import *
import Other_functions
import ClimateGrid_handling
import radar_data_handling
import nwp750_read_plot
import geopandas as gpd
from matplotlib import colors, cm
import matplotlib.pyplot as plt
import numpy as np

from bisect import bisect_left
import collections

###########################################################################
###########################################################################
###########################################################################
os.chdir("C:/Users/olive/Desktop/Speciale/Kode")
# file path to a shapefile with outline of Denmark
world_map_file = "C:/Users/olive/OneDrive/Desktop/Speciale/Kode/pygrib_functionality/pygrib_functionality/world_map_cut/world_map_background.shp"
################################################################################
#Functions

#FSS
def produce_FSS_matrix(dates, model, threshold):
    os.chdir("C:/Users/olive/Desktop/Speciale/Kode")
    fss_collect = [[] for _ in range(len(np.unique(dates)))]
    random_collect = [[] for _ in range(len(np.unique(dates)))]
    for i in range(0, len(np.unique(dates))):
        date = str(np.unique(dates)[i])
        try:
            load_radar = np.loadtxt(
                "./25grid/Radar/radar_20%s.txt" % date[:-2])
        except FileNotFoundError:
            continue

        load_NEA = np.loadtxt("./25grid/NEA/nea_2d_%s.txt" % date)
        load_dk750 = np.loadtxt("./25grid/NWP750/nwp_2d_%s.txt" % date)
        load_radar = np.loadtxt("./25grid/Radar/radar_20%s.txt" % date[:-2])
        # print(date)
        #orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1] // 184, 184)
        orig_nea = load_NEA.reshape(load_NEA.shape[0], load_NEA.shape[1])
        #orig_dk750=load_dk750.reshape(load_dk750.shape[0],load_dk750.shape[1] // 184, 184)
        orig_dk750 = load_dk750.reshape(load_dk750.shape[0], load_dk750.shape[1])
        #orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1] // 184, 184)
        orig_radar = load_radar.reshape(
            load_radar.shape[0], load_radar.shape[1])
        spatial_scale = [2.5, 5, 10, 20, 40, 80, 160, 320]
        if threshold == "Precipitation":
            intensity_scale = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5]
        elif threshold == "Extreme Precipitation":
            intensity_scale = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
        elif threshold == "Cloudbursts":
            intensity_scale = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
        elif threshold == "Severe Rain":
            intensity_scale = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]
        else:
            return("Threshold needs to be either Precipitation or Extreme Precipitation")
        if model == "NEA":
            fss_collect[i], random_collect[i] = spatial_threshold_matrix(
                intensity_scale, spatial_scale, orig_radar[int(date[-2:]):int(date[-2:])+5], orig_nea, orig_dk750)
        elif model == "DK750":
            fss_collect[i], random_collect[i] = spatial_threshold_matrix(
                intensity_scale, spatial_scale, orig_radar[int(date[-2:]):int(date[-2:])+5], orig_dk750, orig_nea)
        else:
            return("Model needs to be either NEA or DK750")
    fss_collect = [x for x in fss_collect if x != []]
    random_collect = [x for x in random_collect if x != []]
    fss_array = np.stack(fss_collect, axis=-1)

    fss_array = np.nanmean(fss_array, axis=-1)

    random_array = [np.mean(k) for k in zip(*random_collect)]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cmap = colors.ListedColormap(["darkred", "firebrick", "orangered", "orange",
                                 "gold", "yellow", "greenyellow", "limegreen", "forestgreen", "green"])
    boundaries = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.set_yscale('log')
    plt.pcolor(intensity_scale, spatial_scale, fss_array,
               shading="auto", alpha=1, cmap=cmap, norm=norm)
    #ax.grid(True, axis='both', linestyle='-', color='k')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Fractional Skill Score (FSS)', rotation=90)
    for i in range(0, len(intensity_scale)):
        for j in range(0, len(spatial_scale)):
            if fss_array[j, i] > 0.1:
                if fss_array[j, i] >= (0.5+random_array[i]/2):
                    ax.text(intensity_scale[i], spatial_scale[j], '{:0.2f}'.format(
                        (fss_array[j, i])), ha='center', va='center', size="small", weight="bold")
                else:
                    ax.text(intensity_scale[i], spatial_scale[j], '{:0.2f}'.format(
                        (fss_array[j, i])), ha='center', va='center', size="small")
            else:
                pass
    plt.xticks(intensity_scale)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_tick_params(which='minor', width=0)
    ax.get_yaxis().set_tick_params(which='minor', size=0)
    ax.get_yaxis().set_tick_params(which='minor', width=0)
    plt.yticks([2.5, 5, 10, 20, 40, 80, 160, 320])
    plt.xlabel("Intensity [mm/hr]")
    plt.ylabel("Scale [km]")
    plt.title(model+" "+threshold)
    plt.show()


def produce_fss_percentiles(dates, percentile, plot=True):
    fss_dk750 = [[] for _ in range(len(np.unique(dates)))]
    fss_nea = [[] for _ in range(len(np.unique(dates)))]
    scalemin_dk750 = []
    scalemin_nea = []
    full_day = Counter([str(i)[:-2] for i in np.unique(dates)])
    for i in range(0, len(np.unique(dates))):
        date = str(np.unique(dates)[i])
        if date[:-2] == str(np.unique(dates)[i-1])[:-2]:
            pass
        else:

            try:
                load_radar = np.loadtxt(
                    "./25grid/Radar/radar_20%s.txt" % date[:-2])
            except FileNotFoundError:
                continue

            load_NEA = np.loadtxt("./25grid/NEA/nea_2d_%s.txt" % date)[0:6]
            load_dk750 = np.loadtxt("./25grid/NWP750/nwp_2d_%s.txt" % date)[0:6]
            load_radar = np.loadtxt(
                "./25grid/Radar/radar_20%s.txt" % date[:-2])

            if full_day[date[:-2]] > 1:
                radar_list = []
                orig_radar = load_radar.reshape(
                    load_radar.shape[0], load_radar.shape[1])
                radar_concat = orig_radar[int(date[-2:])-1:int(date[-2:])+5]
                for times in range(0, full_day[date[:-2]]):
                    date_new = str(np.unique(dates)[i+1])
                    load_NEA = np.concatenate((load_NEA, np.loadtxt(
                        "./25grid/NEA/nea_2d_%s.txt" % date_new)[0:6]), axis=0)
                    load_dk750 = np.concatenate((load_dk750, np.loadtxt(
                        "./25grid/NWP750/nwp_2d_%s.txt" % date_new)[0:6]), axis=0)

                    radar_list.append(date_new[-2:])

                orig_nea = load_NEA.reshape(
                    load_NEA.shape[0], load_NEA.shape[1])
                orig_dk750 = load_dk750.reshape(
                    load_dk750.shape[0], load_dk750.shape[1])

                for k in range(0, len(radar_list)):
                    radar_concat = np.concatenate(
                        (radar_concat, orig_radar[int(radar_list[k]):int(radar_list[k])+6]), axis=0)

                fss_qt, fss_qt_nea, random_qt = Fractional_skillscore(
                    radar_concat, orig_dk750, orig_nea, 366, date, percentile=percentile, plot=plot)
                scalemin_dk750.append(
                    2.5*bisect_left(fss_qt, 0.5+(random_qt[0]/2)))
                scalemin_nea.append(
                    2.5*bisect_left(fss_qt_nea, 0.5+(random_qt[0]/2)))
            else:

                # print(date)

                #orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1] // 184, 184)
                orig_nea = load_NEA.reshape(
                    load_NEA.shape[0], load_NEA.shape[1])
                #orig_dk750=load_dk750.reshape(load_dk750.shape[0],load_dk750.shape[1] // 184, 184)
                orig_dk750 = load_dk750.reshape(
                    load_dk750.shape[0], load_dk750.shape[1])
                #orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1] // 184, 184)
                orig_radar = load_radar.reshape(
                    load_radar.shape[0], load_radar.shape[1])
                fss_qt, fss_qt_nea, random_qt = Fractional_skillscore(orig_radar[int(
                    date[-2:])-1:int(date[-2:])+5], orig_dk750, orig_nea, 366, date, percentile=percentile, plot=plot)
                scalemin_dk750.append(
                    2.5*bisect_left(fss_qt, 0.5+(random_qt[0]/2)))
                scalemin_nea.append(
                    2.5*bisect_left(fss_qt_nea, 0.5+(random_qt[0]/2)))
    return np.array(scalemin_dk750), np.array(scalemin_nea)

def saveFSS(FSS,savepath=os.getcwd()):    
    workingdir=os.getcwd()
    os.chdir(savepath)
    np.savetxt("FSS.csv",FSS,delimiter=",",fmt="%s")
    with open("FSS.csv","w") as f:
        writer=csv.writer(f)
        writer.writerows(FSS)
    os.chdir(workingdir)

def Fractional_skillscore(zs_radar,zs_dk750,zs_nea,domain,date,intensity=0,percentile=0,plot=False):
    avg_fss=[[] for _ in range(len(zs_radar))]
    avg_fss_nea=[[] for _ in range(len(zs_radar))]
    fssrandom=0
    random_divide=0
    qt_dk750=[[] for _ in range(len(zs_dk750))]
    qt_nea=[[] for _ in range(len(zs_nea))]
    for i in range(0,len(zs_radar)):

        #IF QUANTILE IS DEFINED
        if percentile!=0.0:
            radar_grid_2d=np.reshape(zs_radar[i],(156,184))
            dk750_grid_2d=np.reshape(zs_dk750[i],(156,184))
            nea_grid_2d=np.reshape(zs_nea[i],(156,184))
            threshold=np.quantile(np.nan_to_num(radar_grid_2d),percentile)
            qt=(dk750_grid_2d>=np.quantile(np.nan_to_num(dk750_grid_2d),percentile))*1*(threshold)
            qt_n=(nea_grid_2d>=np.quantile(np.nan_to_num(nea_grid_2d),percentile))*1*(threshold)
            qt_dk750[i]=qt
            qt_nea[i]=qt_n
            
            fssrandom=1.0-percentile
            random_divide=1.0
            if plot==False:
                avg_fss[i]=(pvs.fss(qt,radar_grid_2d,threshold,domain))
                avg_fss_nea[i]=(pvs.fss(qt_n,radar_grid_2d,threshold,domain))
            else:
                for scale in range(0,domain):
                    avg_fss[i].append(pvs.fss(qt,radar_grid_2d,threshold,scale))
                    avg_fss_nea[i].append(pvs.fss(qt_n,radar_grid_2d,threshold,scale))
        
        elif intensity!=0.0:
            radar_grid_2d=np.reshape(zs_radar[i],(156,184))
            dk750_grid_2d=np.reshape(zs_dk750[i],(156,184))
            nea_grid_2d=np.reshape(zs_nea[i],(156,184))
            dk750_grid_2d[dk750_grid_2d==0]=np.nan
            radar_grid_2d[radar_grid_2d==0]=np.nan
            nea_grid_2d[nea_grid_2d==0]=np.nan
        
            fssrandom+=np.sum((zs_radar[i]>intensity)*1)
            random_divide+=len(zs_radar[i])
        
            if plot==False:
                avg_fss[i]=(pvs.fss(dk750_grid_2d,radar_grid_2d,intensity,domain))
                avg_fss_nea[i]=(pvs.fss(nea_grid_2d,radar_grid_2d,intensity,domain))
            else: 
                for scale in range(0,domain):
                    avg_fss[i].append(pvs.fss(dk750_grid_2d,radar_grid_2d,intensity,scale))
                    avg_fss_nea[i].append(pvs.fss(nea_grid_2d,radar_grid_2d,intensity,scale))  
        else:
            print("Error. Intensity or percentile needs to be defined")
            break
        

    if plot==False:
        return avg_fss,avg_fss_nea,fssrandom/random_divide
    else:
        FSS_mean_dk750=[np.mean(k) for k in zip(*avg_fss)]
        FSS_mean_nea=[np.mean(k) for k in zip(*avg_fss_nea)]
        dk750_useful=2.5*bisect_left(FSS_mean_dk750,0.5+(fssrandom/random_divide)/2)
        nea_useful=2.5*bisect_left(FSS_mean_nea,0.5+(fssrandom/random_divide)/2)
        fssrandom=np.repeat(fssrandom/random_divide,domain)
    #return FSS_mean_dk750, FSS_mean_nea, fssrandom/random_divide #TEMPORARY
        plt.figure()

        plt.plot(np.arange(0,domain,1)*2.5,FSS_mean_dk750,"--",color="darkgrey")
        plt.plot(np.arange(0,domain,1)*2.5,FSS_mean_nea,"-",color="black")
        plt.plot(np.arange(0,domain,1)*2.5,fssrandom,color="firebrick",alpha=0.5)
        plt.plot(np.arange(0,domain,1)*2.5,0.5+fssrandom/2,color="gold")
        plt.text(10,0.95,"Scale above FSS uniform. \nDK 750: %s km \nNEA 2500: %s km"%(int(np.round(dk750_useful)),int(np.round(nea_useful))),bbox=dict(boxstyle="round",facecolor="white"))
        plt.ylim(0,1.1)
        plt.xlim(0,800)
        plt.xlabel("Horizontal Scale [km]")
        plt.ylabel("Fractional Skill Score")
        plt.title(date[4:6]+"/"+date[2:4])
        plt.legend(["DK 750","NEA 2500","FSS random","FSS uniform"],loc="lower right")
        plt.show()
    return FSS_mean_dk750, FSS_mean_nea, fssrandom/random_divide

#Spatial array should be given in some logspace array. Intensity array linear
#spatial scale should go from high to low, intensity array the other way around
def spatial_threshold_matrix(intensity_array,spatial_array,zt_radar,zt_dk750,zt_nea):  
    fss=np.zeros((len(spatial_array),(len(intensity_array))))
    fss_random=[[] for _ in range(len(intensity_array))]
    for i in range(0,len(spatial_array)):
        domain=spatial_array[i]
        for j in range(0,len(intensity_array)):
            precipitation=intensity_array[j]
            fss_temp,fss_throw,fss_random[j]=Fractional_skillscore(zt_radar,zt_dk750,zt_nea,domain,None,intensity=precipitation)
            fss[i,j]=np.nanmean(fss_temp)
    return fss,fss_random

# SCALEMIN PLOT
def scalemin_plot(dk750,nea,dates):
    
    scalemin_dk750 = []
    scalemin_nea = []
    for i in np.arange(0.98, 0.999, 0.001):
        temp_dk750, temp_nea = produce_fss_percentiles(dates, i, plot=True)
        temp_dk750[dk750 >= 914] = np.nan
        temp_nea[nea >= 914] = np.nan
        scalemin_dk750.append(temp_dk750)
        scalemin_nea.append(temp_nea)

    dk750_meanscale = np.array([np.nanmean(scalemin_dk750[k])
                         for k in range(0, len(scalemin_dk750))])
    nea_meanscale = np.array([np.nanmean(scalemin_nea[k])
                         for k in range(0, len(scalemin_nea))])
    plt.plot(np.arange(0.985, 0.999, 0.001),
             dk750_meanscale[5:20], "--", color="black")
    plt.plot(np.arange(0.985, 0.999, 0.001),
             nea_meanscale[5:20], "-", color="black")
    plt.legend(["DK750", "NEA2500", "IQR DK750", "IQR NEA2500"], loc="upper left")
    plt.xlabel("Percentile threshold (%)")
    plt.ylabel("Minimum scale [km]")
    plt.show()

def plot_fss_boxplot(dates):
    
    boxplot_nea = []
    boxplot_dk750 = []
    for j in [2.5, 5, 10, 20, 40, 80, 160, 320]:
        nea_temp = []
        dk750_temp = []
        for i in range(0, len(np.unique(dates))):
            event = str(np.unique(dates)[i])

            try:
                load_radar = np.loadtxt(
                    "./25grid/Radar/radar_20%s.txt" % event[:-2])
            except FileNotFoundError:
                continue

            load_NEA = np.loadtxt("./25grid/NEA/nea_2d_%s.txt" % event)[0:7]
            load_dk750 = np.loadtxt("./25grid/NWP750/nwp_2d_%s.txt" % event)[0:7]

            load_radar = load_radar[int(event[-2:])-1:int(event[-2:])+6]
            fss_dk750, fss_nea, fssrandom = Fractional_skillscore(
                load_radar, load_dk750, load_NEA, j, event, intensity=15, plot=False)
            nea_temp.append(np.nanmean(fss_nea))
            dk750_temp.append(np.nanmean(fss_dk750))
        boxplot_nea.append(np.nan_to_num(np.array(nea_temp)))
        boxplot_dk750.append(np.nan_to_num(np.array(dk750_temp)))


    plt.figure(figsize=(10, 5))
    plt.boxplot(boxplot_dk750, medianprops=dict(color="black"), boxprops=dict(linestyle="--", color="grey"), whiskerprops=dict(linestyle="--",
            color="grey"), positions=[0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8], widths=(0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25))
    plt.boxplot(boxplot_nea, medianprops=dict(color="black"), positions=[
            1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2], widths=(0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25))
    plt.xlabel("Scale [km]", fontsize=13)
    plt.ylabel("FSS", fontsize=13)
    plt.xlim(0, 9)
    plt.xticks(np.arange(1, 9), ["2.5", "5", "10", "20", "40", "80", "160", "320"])
    hB, = plot([1, 1], '--', color="grey")
    hR, = plot([1, 1], '-', color="black")
    legend((hB, hR), ('DK750', 'NEA'), loc="upper left")
    hB.set_visible(False)
    hR.set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_fss_leadtime_boxplot(events):
    boxplot_lead_nea = []
    boxplot_lead_dk750 = []
    for j in np.arange(0, 7, 1):
        nea_temp = []
        dk750_temp = []
        for i in range(0, len(np.unique(events))):
            day = str(np.unique(events)[i])
    
            try:
                load_radar = np.loadtxt(
                    "./25grid/Radar/radar_20%s.txt" % day[:-2])
            except FileNotFoundError:
                continue
    
            if int(day[-2:]) == 18:
                try:
                    load_radar1 = np.loadtxt(
                        "./25grid/Radar/radar_20%s.txt" % str(int(day[:-2])+1))
                except FileNotFoundError:
                    continue
                load_radar = np.vstack(
                    (load_radar[int(day[-2:]):int(day[-2:])+5], load_radar1[0:2]))
            else:
                load_NEA = np.loadtxt("./25grid/NEA/nea_2d_%s.txt" % day)[0:7]
                load_dk750 = np.loadtxt(
                    "./25grid/NWP750/nwp_2d_%s.txt" % day)[0:7]
                load_radar = load_radar[int(day[-2:]):int(day[-2:])+7]
            dk750_grid_2d = np.reshape(load_dk750[j], (156, 184))
            nea_grid_2d = np.reshape(load_NEA[j], (156, 184))
            radar_grid_2d = np.reshape(load_radar[j], (156, 184))
            dk750_grid_2d[dk750_grid_2d == 0] = np.nan
            radar_grid_2d[radar_grid_2d == 0] = np.nan
            nea_grid_2d[nea_grid_2d == 0] = np.nan
            fss_dk750 = (pvs.fss(dk750_grid_2d, radar_grid_2d, 5, 20))
            fss_nea = (pvs.fss(nea_grid_2d, radar_grid_2d, 5, 20))
    
            nea_temp.append(np.nan_to_num(fss_nea))
            dk750_temp.append(np.nan_to_num(fss_dk750))
        boxplot_lead_nea.append(np.array(nea_temp))
        boxplot_lead_dk750.append(np.array(dk750_temp))

    plt.figure(figsize=(10, 5))
    plt.boxplot(boxplot_lead_dk750, medianprops=dict(color="black"), boxprops=dict(linestyle="--", color="grey"), whiskerprops=dict(
        linestyle="--", color="grey"), positions=[0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8], widths=(0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25))
    plt.boxplot(boxplot_lead_nea, medianprops=dict(color="black"), positions=[
                1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2], widths=(0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25))
    plt.xlabel("Lead time [hr]", fontsize=13)
    plt.ylabel("FSS", fontsize=13)
    plt.xlim(0, 8)
    plt.xticks(np.arange(1, 8), ["1", "2", "3", "4", "5", "6", "7"])
    hB, = plot([1, 1], '--', color="grey")
    hR, = plot([1, 1], '-', color="black")
    legend((hB, hR), ('DK750', 'NEA'), loc="upper right")
    hB.set_visible(False)
    hR.set_visible(False)
    plt.tight_layout()
    plt.show()

    
###############################################################################    
#SAL
def objects(date,leadtime,lons,lats):
    load_radar = np.loadtxt("./25grid/Radar/radar_20%s.txt" % date[:-2]) #15
    load_dk750 = np.loadtxt("./25grid/NWP750/nwp_2d_%s.txt" % date) 

    load_dk750 = load_dk750.reshape(
        load_dk750.shape[0], load_dk750.shape[1] // 184, 184)[leadtime]
    load_radar = load_radar.reshape(
        load_radar.shape[0], load_radar.shape[1] // 184, 184)[int(date[-2:])+leadtime]

    #RADAR
    threshold_object=1/15*np.nanquantile(load_radar[load_radar > np.nanmin(load_radar)],0.95)
    tstorm_kwargs=dict()
    tstorm_kwargs = {
                "minmax": tstorm_kwargs.get("minmax", threshold_object),
                "maxref": tstorm_kwargs.get("maxref", threshold_object+ 1e-5),
                "mindiff": tstorm_kwargs.get("mindiff", 1e-5),
                "minref": tstorm_kwargs.get("minref", threshold_object),}

    _, objects=pysteps.feature.tstorm.detection(load_radar, **tstorm_kwargs)
    objects=remove_values_below(objects,0.5)

    #dk750
    threshold_object_dk750=1/15*np.nanquantile(load_dk750[load_dk750 > np.nanmin(load_dk750)],0.95)
    tstorm_kwargs_dk750=dict()
    tstorm_kwargs_dk750 = {
                "minmax": tstorm_kwargs_dk750.get("minmax", threshold_object_dk750),
                "maxref": tstorm_kwargs_dk750.get("maxref", threshold_object_dk750+ 1e-5),
                "mindiff": tstorm_kwargs_dk750.get("mindiff", 1e-5),
                "minref": tstorm_kwargs_dk750.get("minref", threshold_object_dk750),}

    _, objects_dk750=pysteps.feature.tstorm.detection(load_dk750, **tstorm_kwargs_dk750)
    objects_dk750=remove_values_below(objects_dk750,0.5)

    plot_objects(load_radar, objects,load_dk750, objects_dk750,
                 lons, lats, r"$\bf{Rainfall}$"+ "\n Radar500 \n 2021/%s/%s - %s:00 UTC"%(str(date[2:4]),str(date[4:6]),str(int(date[-2:])+leadtime+1)), r"$\bf{Objects}$"+ "\n Radar500 \n 2021/%s/%s - %s:00 UTC"%(str(date[2:4]),str(date[4:6]),str(int(date[-2:])+leadtime+1)),"DK750 \n 2021/%s/%s - %s:00+%s UTC"%(str(date[2:4]),str(date[4:6]),str(date[-2:]),str(leadtime+1)), "DK750 \n 2021/%s/%s - %s:00+%s UTC"%(str(date[2:4]),str(date[4:6]),str(date[-2:]),str(leadtime+1)))

    print(pvs_sal.sal(np.nan_to_num(load_dk750),np.nan_to_num(load_radar),thr_factor=1/15,thr_quantile=0.95,tstorm_kwargs=None))
    print(pvs_sal._sal_l1_param(np.nan_to_num(load_dk750),np.nan_to_num(load_radar)))
    print(pvs_sal._sal_l2_param(np.nan_to_num(load_dk750),np.nan_to_num(load_radar),thr_factor=1/15,thr_quantile=0.95,tstorm_kwargs=None))


def plot_objects(radar_array,radar_objects, nwp_array,nwp_objects, lons, lats, plot_title_radar, plot_title_radar_objects,plot_title_nwp,plot_title_nwp_objects):

    # file path to a shapefile with outline of Denmark
    world_map_file = world_map_file = "C:/Users/olive/OneDrive/Desktop/Speciale/Kode/pygrib_functionality/pygrib_functionality/world_map_cut/world_map_background.shp"
    world_map = gpd.read_file(world_map_file)
    bound = np.max([np.unique(radar_objects)[-2],np.unique(nwp_objects)[-2]]) #Determines the max number of objects for the colorscale
    
    
    # # create custom color map
    #from matplotlib import cm
    cmap_real = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
    #boundaries = [0, .5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
    boundaries_real = [0, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100]
    norm_real = colors.BoundaryNorm(boundaries_real, cmap_real.N, clip=True)
    
    cmap = cm.get_cmap('inferno')
    boundaries = np.arange(0, bound+1, 1)
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)


    center_radar = np.round(ndimage.measurements.center_of_mass(
        np.nan_to_num(radar_objects) > 0)*1)
    center_nwp = np.round(ndimage.measurements.center_of_mass(
        np.nan_to_num(nwp_objects) > 0)*1)


    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, gridspec_kw={
                            "width_ratios": [1, 1]}, figsize=(11, 15))
    fig.text(0.5, 0.04, "Longitude", ha='center')
    fig.text(0.08, 0.5, "Latitude", va='center', rotation='vertical')
    fig.text(0.04,0.5," ⟵ Horizontal resolution  ⟵",va='center',rotation='vertical',fontsize=14)
    
    for i in np.arange(0,2,1):
        for j in np.arange(0,2,1):
            world_map.plot(facecolor="lightgrey", ax=axs[i,j])
            axs[i,j].set_xlim(7.5, 13)
            axs[i,j].set_ylim(53.8, 57.8)
    pcm_real = axs[0,0].pcolor(lons, lats, radar_array,
                        shading="auto", alpha=1, cmap=cmap_real, norm=norm_real)

    axs[0,0].title.set_text(plot_title_radar)
    pcm_real = axs[1,0].pcolor(lons, lats, nwp_array,
                        shading="auto", alpha=1, cmap=cmap_real, norm=norm_real)
    cax1=plt.axes([0.48,0.15,0.020,0.7])
    cbar_real = fig.colorbar(pcm_real, ax=axs[:,0], cax=cax1,fraction=0.46,pad=.04)
    cbar_real.ax.set_ylabel('Rainfall Intensity [mm/hr]', rotation=90)

    axs[1,0].title.set_text(plot_title_nwp)
    pcm = axs[0,1].pcolor(lons, lats, radar_objects, shading="auto",
                        alpha=1, cmap=cmap, norm=norm)
    axs[0,1].plot(lons[int(center_nwp[0]), int(center_radar[1])], lats[int(
        center_radar[0]), int(center_radar[1])], "+", markersize=25, color="limegreen")
    axs[0,1].title.set_text(plot_title_radar_objects)
    pcm = axs[1,1].pcolor(lons, lats, nwp_objects, shading="auto",
                        alpha=1, cmap=cmap, norm=norm)
    axs[1,1].plot(lons[int(center_nwp[0]), int(center_nwp[1])], lats[int(
        center_nwp[0]), int(center_nwp[1])], "+", markersize=25, color="limegreen")

    axs[1,1].title.set_text(plot_title_nwp_objects)
    plt.tick_params('y', labelleft=False)
    # axs[1].set_yticks([])
    cax=plt.axes([0.92,0.15,0.020,0.7])
    cbar = fig.colorbar(pcm, ax=axs[:,1], cax=cax, fraction=0.46,pad=.04)
    cbar.ax.set_ylabel('Uniquely identified cells', rotation=90)
    cbar.set_ticks([])
    plt.subplots_adjust(wspace=0.3, hspace=0)
    plt.show()

def produce_SAL(radar,nwp):
    amp=[]
    struc=[]
    loca=[]
    #l1_tst=[]
    l_tst=[]
    for i in range(0,len(radar)):
        s,a,l=pvs_sal.sal(np.nan_to_num(nwp[i]),np.nan_to_num(radar[i]),thr_factor=1/15,thr_quantile=0.985,tstorm_kwargs=None)
        struc.append(s)
        amp.append(a)
        loca.append(l)
        #l1=pvs_sal._sal_l1_param(np.nan_to_num(nwp[i]),np.nan_to_num(radar[i]))
        #l2=pvs_sal._sal_l2_param(np.nan_to_num(nwp[i]),np.nan_to_num(radar[i]),thr_factor=None,thr_quantile=None,tstorm_kwargs=loc_dict)
        #l2=pvs_sal._sal_l2_param(np.nan_to_num(nwp[i]),np.nan_to_num(radar[i]),thr_factor=1/15,thr_quantile=0.99,tstorm_kwargs=None)
        #print(l2)
        #amp.append(pvs_sal.sal_amplitude(np.nan_to_num(nwp[i]),np.nan_to_num(radar[i])))
        #struc.append(pvs_sal.sal_structure(np.nan_to_num(nwp[i]),np.nan_to_num(radar[i]),tstorm_kwargs=None))
        #loca.append(np.nansum((l1,l2)))
    return struc,amp,loca


def plot_SAL(s,a,l,title):
    plt.figure()
    cm=plt.cm.get_cmap('RdYlGn_r')
    plt.axhline(0,0,color="black",alpha=0.5)
    plt.axvline(0,0,color="black",alpha=0.5)
    plt.axhline(np.nanmedian(a),0,color="red",alpha=0.5)
    plt.axvline(np.nanmedian(s),0,color="red",alpha=0.5)
    plt.fill_between(x=(np.nanquantile(s,0.25),np.nanquantile(s,0.75)),y1=(np.nanquantile(a,0.25),np.nanquantile(a,0.25)),y2=(np.nanquantile(a,0.75),np.nanquantile(a,0.75)))
    for i in range(0,len(s)):
        sc=plt.scatter(s[i],a[i],c=l[i],vmin=0,vmax=2,cmap=cm,zorder=2)

    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.xlabel("Structure")
    plt.ylabel("Amplitude")
    cbar=plt.colorbar(sc)
    cbar.ax.axhline(y=np.nanmedian(l), c='w')  
    cbar.ax.annotate("M",(0.2,np.nanmedian(l)+0.03))
    cbar.ax.set_ylabel('Location', rotation=90)
    plt.title(title)
    plt.show()
    
def SAL_output(dates, model, threshold):
    os.chdir("C:/Users/olive/Desktop/Speciale/Kode")
    struc = [[] for _ in range(len(np.unique(dates)))]
    amp = [[] for _ in range(len(np.unique(dates)))]
    loca = [[] for _ in range(len(np.unique(dates)))]
    for i in range(0, len(np.unique(dates))):
        date = str(np.unique(dates)[i])
        try:
            load_radar = np.loadtxt(
                "./25grid/Radar/radar_20%s.txt" % date[:-2])
        except FileNotFoundError:
            continue

        load_NEA = np.loadtxt("./25grid/NEA/nea_2d_%s.txt" % date)
        load_dk750 = np.loadtxt("./25grid/NWP750/nwp_2d_%s.txt" % date)
        load_radar = np.loadtxt("./25grid/Radar/radar_20%s.txt" % date[:-2])

        orig_nea = load_NEA.reshape(
            load_NEA.shape[0], load_NEA.shape[1] // 184, 184)
        # orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1])
        orig_dk750 = load_dk750.reshape(
            load_dk750.shape[0], load_dk750.shape[1] // 184, 184)
        orig_radar = load_radar.reshape(
            load_radar.shape[0], load_radar.shape[1] // 184, 184)
        if model == "NEA":
            struc[i], amp[i], loca[i] = produce_SAL(
                orig_radar[int(date[-2:]):int(date[-2:])+6], orig_nea)
        elif model == "DK750":
            struc[i], amp[i], loca[i] = produce_SAL(
                orig_radar[int(date[-2:]):int(date[-2:])+6], orig_dk750)
    struc = [x for x in struc if x != []]
    amp = [x for x in amp if x != []]
    loca = [x for x in loca if x != []]
    #print("Structure: %s"%struc)
    #print("Amplitude: %s"%amp)
    #print("Location: %s"%loca)
    title = model+" "+threshold
    plot_SAL(struc, amp, loca, title)
    return struc, amp, loca

def plot_SAL_l(s,a,l,title):
    plt.figure()
    plt.axhline(0,0,color="black",alpha=0.5)
    plt.axvline(0,0,color="black",alpha=0.5)
    plt.axhline(np.nanmedian(l),0,color="red",alpha=0.5)
    plt.axvline(np.nanmedian(a),0,color="red",alpha=0.5)
    plt.scatter(a,l)
    plt.fill_between(x=(np.nanquantile(a,0.25),np.nanquantile(a,0.75)),y1=(np.nanquantile(l,0.25),np.nanquantile(l,0.25)),y2=(np.nanquantile(l,0.75),np.nanquantile(l,0.75)))
    plt.xlim(-2,2)
    plt.ylim(0,2)
    plt.xlabel("Amplitude")
    plt.ylabel("Location")
    plt.title(title)
    plt.show()


