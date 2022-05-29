# -*- coding: utf-8 -*-
"""
Created on Sat May 28 08:25:24 2022

@author: olive
"""

#PLOTTING FUNCTIONS"
import os
os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Scripts")

import nwp750_read_plot
import radar_data_handling
import ClimateGrid_handling
from nwp750_read_plot import *
from radar_data_handling import *
from ClimateGrid_handling import *
import numpy as np
import matplotlib.pyplot as plt
import gpd
from matplotlib import colors


os.chdir("C:/Users/olive/Desktop/Speciale/Kode")

#The below shows two side-by-side plots showing the radar product and an NWP product
def plot_together(rain_array,NWP_data, radar_lons, radar_lats, nwp_lons, nwp_lats, world_map_file, plot_title_radar,plot_title_nwp):
    
    world_map = gpd.read_file(world_map_file)
    
    # create custom color map
    cmap = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
    boundaries = [0, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    
    # Make plot
    fig,axs=plt.subplots(1,2,sharex=True,sharey=True,gridspec_kw={"width_ratios":[1,1.09]},figsize=(15,15))
    fig.text(0.5,0.04,"Longitude",ha='center')
    fig.text(0.04,0.5,"Latitude",va='center',rotation='vertical')
    for i in np.arange(0,2,1):
        world_map.plot(facecolor="lightgrey",ax=axs[i])
        axs[i].set_xlim(7,15)
        axs[i].set_ylim(54.5,58)
    pcm=axs[0].pcolor(radar_lons, radar_lats, rain_array, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[0].title.set_text(plot_title_radar)
    pcm=axs[1].pcolor(nwp_lons, nwp_lats, NWP_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[1].title.set_text(plot_title_nwp)
    plt.tick_params('y',labelleft=False)
    cbar = fig.colorbar(pcm,ax=axs[1],fraction=0.040, pad=0.04)
    cbar.ax.set_ylabel('Rainfall intensity [mm/h]', rotation=90)
    plt.subplots_adjust(wspace=0.05,hspace=0)
    plt.show()

#The below shows three side-by-side plots showing the radar product and both NWP product
def plot_all(radar_data,nwp_data,nea_data, radar_lons, radar_lats, nwp_lons, nwp_lats, nea_lons, nea_lats, world_map_file, plot_title_radar,plot_title_nwp,plot_title_nea,save_name,timestep,preciptype):
    
    world_map = gpd.read_file(world_map_file)
    
    # create custom color map
    cmap = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
    #boundaries = [0, .5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
    boundaries = [0, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    
    # Make plot
    fig,axs=plt.subplots(1,3,sharex=True,sharey=True,gridspec_kw={"width_ratios":[1,1,1.09]},figsize=(16,8))
    fig.text(0.5,0.2,"Longitude",ha='center')
    fig.text(0.06,0.5,"Latitude",va='center',rotation='vertical')
    #fig.tight_layout(pad=3.0)
    for i in np.arange(0,3,1):
        world_map.plot(facecolor="lightgrey",ax=axs[i])
        axs[i].set_xlim(7,15)
        axs[i].set_ylim(54.5,58)
    pcm=axs[0].pcolor(radar_lons, radar_lats, radar_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[0].title.set_text(plot_title_radar)
    pcm=axs[1].pcolor(nwp_lons, nwp_lats, nwp_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[1].title.set_text(plot_title_nwp)
    axs[1].tick_params('y',labelleft=False,left=False)
    pcm=axs[2].pcolor(nea_lons, nea_lats, nea_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[2].title.set_text(plot_title_nea)
    axs[2].tick_params('y',labelleft=False,left=False)
    cbar = fig.colorbar(pcm,ax=axs[2],fraction=0.040, pad=0.04)
    cbar.ax.set_ylabel('Rainfall intensity [mm/h]', rotation=90)
    plt.subplots_adjust(wspace=0.05,hspace=0)
    plt.show()
    current=os.getcwd()
    os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Pics/%s"%preciptype) 
    newpath=r'./%s/'%save_name[-6:-2]
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Pics/%s/%s"%(preciptype,newpath[-6:-1]))
    save_name=os.getcwd()+"/"+str(save_name[-2:])+"_"+str(timestep)
    if os.path.isfile(save_name):
        os.remove(save_name)
    plt.savefig(save_name,bbox_inches='tight')
    plt.close()
    os.chdir(current)  


def plot_3x3(radar_data,radar_data1,radar_data2,nwp_data,nwp_data1,nwp_data2,nea_data,nea_data1,nea_data2, radar_lons, radar_lats, nwp_lons, nwp_lats, nea_lons, nea_lats, world_map_file, plot_title_radar,plot_title_radar1,plot_title_radar2,plot_title_nwp,plot_title_nwp1,plot_title_nwp2,plot_title_nea,plot_title_nea1,plot_title_nea2,save_name,timestep,preciptype):
    
    world_map = gpd.read_file(world_map_file)
    
    # create custom color map
    cmap = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
    boundaries = [0, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    
    # Make plot
    fig,axs=plt.subplots(3,3,sharex=True,sharey=True,figsize=(10,15),gridspec_kw={"width_ratios":[1,1,1]},constrained_layout=True)
    fig.text(0.5,0.04,"⟶ Horizontal Resolution ⟶",ha='center',fontsize=14)
    fig.text(0.04,0.5," ⟵ Lead time  ⟵",va='center',rotation='vertical',fontsize=14)  
    for i in np.arange(0,3,1):
        for j in np.arange(0,3,1):
            world_map.plot(facecolor="lightgrey",ax=axs[i,j])
            axs[i,j].set_xlim(7.5,13)
            axs[i,j].set_ylim(54.5,58)         
            
    pcm=axs[0,0].pcolor(radar_lons, radar_lats, radar_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[0,0].set_ylabel("Latitude")
    pcm=axs[1,0].pcolor(radar_lons, radar_lats, radar_data1, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[1,0].set_ylabel("Latitude")
    pcm=axs[2,0].pcolor(radar_lons, radar_lats, radar_data2, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[2,0].set_ylabel("Latitude")
    axs[2,0].set_xlabel("Longitude")
    axs[0,0].title.set_text(plot_title_radar)
    axs[1,0].title.set_text(plot_title_radar1)
    axs[2,0].title.set_text(plot_title_radar2)
    pcm=axs[0,1].pcolor(nwp_lons, nwp_lats, nwp_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    pcm=axs[1,1].pcolor(nwp_lons, nwp_lats, nwp_data1, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    pcm=axs[2,1].pcolor(nwp_lons, nwp_lats, nwp_data2, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[2,1].set_xlabel("Longitude")
    axs[0,1].title.set_text(plot_title_nwp)
    axs[1,1].title.set_text(plot_title_nwp1)
    axs[2,1].title.set_text(plot_title_nwp2)
    axs[0,1].tick_params(axis='both',labelleft=False,left=False)
    axs[1,1].tick_params(axis='both',labelleft=False,left=False)
    axs[2,1].tick_params('y',labelleft=False,left=False)
    pcm=axs[0,2].pcolor(nea_lons, nea_lats, nea_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[0,2].title.set_text(plot_title_nea)
    axs[1,2].title.set_text(plot_title_nea1)
    axs[2,2].title.set_text(plot_title_nea2)
    axs[0,2].tick_params(axis='both',labelleft=False,left=False)
    pcm=axs[1,2].pcolor(nea_lons, nea_lats, nea_data1, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[1,2].tick_params(axis='both',labelleft=False,left=False)
    pcm=axs[2,2].pcolor(nea_lons, nea_lats, nea_data2, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[2,2].set_xlabel("Longitude")
    axs[2,2].tick_params('y',labelleft=False,left=False)
    cax=plt.axes([0.92,0.15,0.020,0.7])
    cbar = fig.colorbar(pcm,ax=axs[2,:],cax=cax,shrink=1.0)
    cbar.ax.set_ylabel('Rainfall intensity [mm/h]', rotation=90)
    plt.subplots_adjust(wspace=0.05,hspace=0.0)
    plt.show()
    