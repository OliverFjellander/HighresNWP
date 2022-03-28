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
#from skimage.feature import peak_local_max
from pygeoprocessing import zonal_statistics

os.chdir("C:/Users/olive/Desktop/Speciale/Kode")

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
    #plt.title(plot_title)
    #save_name=save_path%str(save_name[-1:])[-17:-7]
    #if os.path.isfile(save_name):
    #    os.remove(save_name)
    #plt.savefig(save_name,bbox_inches='tight')
    #plt.close()

#This contains scripts that uses data from multiple sources
def verification_area(radar_data,radar_lons,radar_lats,extracted_fields_nwp,neighbourhood,intensity,scale,threshold_qt=0.975):
    
    #Neighbourhood=1 should represent squares of 7.5 km but I have had issues with needing to use np.floor and np.ceil below because some numbers are floats
    bound_radar=neighbourhood*5*3/2
    bound_nwp=neighbourhood*5
    radar_mid=int(bound_radar*2)
    
    lons_nwp=extracted_fields_nwp['lons']
    lats_nwp=extracted_fields_nwp['lats']
    rain_nwp=extracted_fields_nwp['values']
    
    indices = peak_local_max(radar_data, min_distance=radar_mid,threshold_abs=intensity)
#    indices = peak_local_max(radar_data, min_distance=radar_mid,threshold_abs=np.nanquantile(radar_data,threshold_qt))
    rain_nwp_out=[[] for _ in range(len(indices))]
    rain_radar_out=[[] for _ in range(len(indices))]
    gridcells_nwp=[[] for _ in range(len(indices))]
    gridcells_radar=[[] for _ in range(len(indices))]
    for k in range(0,len(indices)):
        #So far just increased the area with 100 each way which should be 3-4 squares each direction (squares of 15 km dependent on neighbourhood)
        horizontal=np.concatenate(((np.flip(np.arange(-indices[k][1]+radar_mid,-indices[k][1]+scale*radar_mid,radar_mid)))*-1,np.arange(indices[k][1],indices[k][1]+scale*radar_mid,radar_mid)))
        vertical=np.concatenate(((np.flip(np.arange(-indices[k][0]+radar_mid,-indices[k][0]+scale*radar_mid,radar_mid)))*-1,np.arange(indices[k][0],indices[k][0]+scale*radar_mid,radar_mid)))
    
    
        lons_nwp=extracted_fields_nwp['lons']
        lats_nwp=extracted_fields_nwp['lats']
        rain_nwp=extracted_fields_nwp['values']
        for i in range(0,len(horizontal)):
            for j in range(0,len(vertical)):
                rain=radar_data[int(vertical[j]-np.floor(bound_radar)):int(vertical[j]+np.ceil(bound_radar)),int(horizontal[i]-np.floor(bound_radar)):int(horizontal[i]+np.ceil(bound_radar))]
                lons=radar_lons[int(vertical[j]-np.floor(bound_radar)):int(vertical[j]+np.ceil(bound_radar)),int(horizontal[i]-np.floor(bound_radar)):int(horizontal[i]+np.ceil(bound_radar))]
                lats=radar_lats[int(vertical[j]-np.floor(bound_radar)):int(vertical[j]+np.ceil(bound_radar)),int(horizontal[i]-np.floor(bound_radar)):int(horizontal[i]+np.ceil(bound_radar))]
                #radar_plot(rain, lons, lats, world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_hr[1][-1:])[-11:-9],str(Myfiles_hr[1][-1:])[-13:-11],str(Myfiles_hr[1][-1:])[-17:-13],str(Myfiles_hr[1][-1:])[-9:-7]),Myfiles_hr)
                lons_mid=radar_lons[vertical[j],horizontal[i]]
                lats_mid=radar_lats[vertical[j],horizontal[i]]
            
                #This resembles which NWP coordinates are the closest to the radar coordinates of the middle of the event
                coor_1=np.where(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)==np.min(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)))[0][0]
                coor_2=np.where(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)==np.min(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)))[1][0]
            
            #This makes sure that no areas but by boundary coordinates are used
                if coor_1==0 or coor_2==0 or coor_1==len(lats_nwp)-1 or coor_2==len(lons_nwp)-1 or int(coor_1-np.floor(bound_nwp))<=0 or int(coor_2-np.floor(bound_nwp))<=0 or coor_2>=690 or coor_1>=620:
                    pass
                else:
                    lats_nwp_new=lats_nwp[int(coor_1-np.floor(bound_nwp)):int(coor_1+np.ceil(bound_nwp)),int(coor_2-np.floor(bound_nwp)):int(coor_2+np.ceil(bound_nwp))]
                    lons_nwp_new=lons_nwp[int(coor_1-np.floor(bound_nwp)):int(coor_1+np.ceil(bound_nwp)),int(coor_2-np.floor(bound_nwp)):int(coor_2+np.ceil(bound_nwp))]
                    rain_nwp_new=rain_nwp[int(coor_1-np.floor(bound_nwp)):int(coor_1+np.ceil(bound_nwp)),int(coor_2-np.floor(bound_nwp)):int(coor_2+np.ceil(bound_nwp))]
                    rain_nwp_new=remove_values_below(rain_nwp_new, 0.5)
                    gridcells_nwp[k].append([lons_nwp_new,lats_nwp_new])
                    gridcells_radar[k].append([lons,lats])
                    rain_nwp_out[k].append(rain_nwp_new)
                    rain_radar_out[k].append(rain)
    return gridcells_nwp,gridcells_radar,rain_nwp_out,rain_radar_out

def total_precipitation(data,data_nwp):
    total=0
    total_nwp=0
    for i in range(0,len(data)):
        total=total+np.nansum(data[i])
        total_nwp=total_nwp+np.nansum(data_nwp[i+1])
    return total,total_nwp

def FSS_func(radar_data,radar_lo,radar_la,nwp_coor,neighbourhood,intensity,scale,threshold_qt=0.975):
    FSS=[[] for _ in range(len(radar_data))]
    FSSrandom=[]
    for k in range(0,len(radar_data)):
        grid_nwp,grid_radar,precip_nwp,precip_radar=verification_area(np.nan_to_num(radar_data[k]),radar_lo,radar_la,nwp_coor[k+1],neighbourhood,intensity,scale,threshold_qt)       
        FSSrandom.append(np.sum((radar_data[k]>0)*1)/np.size(radar_data[k]))
        for i in range(0,len(precip_nwp)):
            top=0
            bottom_pf=0
            bottom_po=0
            for j in range(0,len(precip_nwp[i])):
                bin_radar=(precip_radar[i][j]>0)*1
                bin_nwp=(precip_nwp[i][j]>0)*1
                fraction_radar=np.sum(bin_radar)/np.size(bin_radar)
                fraction_nwp=np.sum(bin_nwp)/np.size(bin_nwp)
                top=top+(fraction_nwp-fraction_radar)**2
                bottom_pf=bottom_pf+(fraction_nwp)**2
                bottom_po=bottom_po+(fraction_radar)**2
            if bottom_pf==0 and bottom_po==0:
               pass
            else:
                FSS[k].append(1-(top/len(precip_nwp))/((bottom_pf+bottom_po)/len(precip_nwp)))
    return FSS,FSSrandom

def saveFSS(FSS,savepath=os.getcwd()):    
    workingdir=os.getcwd()
    os.chdir(savepath)
    np.savetxt("FSS.csv",FSS,delimiter=",",fmt="%s")
    with open("FSS.csv","w") as f:
        writer=csv.writer(f)
        writer.writerows(FSS)
    os.chdir(workingdir)


def verification_alldomain(radar_data,radar_lons,radar_lats,extracted_fields_nwp,neighbourhood,intensity,threshold_qt=0.975):
    
    #Neighbourhood=1 should represent squares of 7.5 km but I have had issues with needing to use np.floor and np.ceil below because some numbers are floats
    bound_radar=neighbourhood*5*3/2
    bound_nwp=neighbourhood*5
    radar_mid=int(bound_radar*2)
    
    lons_nwp=extracted_fields_nwp['lons']
    lats_nwp=extracted_fields_nwp['lats']
    rain_nwp=extracted_fields_nwp['values']
    
    indices = peak_local_max(radar_data, min_distance=radar_mid,threshold_abs=intensity)
#    indices = peak_local_max(radar_data, min_distance=radar_mid,threshold_abs=np.nanquantile(radar_data,threshold_qt))
    rain_nwp_out=[]
    rain_radar_out=[]
    gridcells=[]
    for k in range(0,1):
        #So far just increased the area with 100 each way which should be 3-4 squares each direction (squares of 15 km dependent on neighbourhood)
        horizontal=np.concatenate(((np.flip(np.arange(-indices[k][1]+radar_mid,0,radar_mid)))*-1,np.arange(indices[k][1],len(radar_data[1]),radar_mid)))
        vertical=np.concatenate(((np.flip(np.arange(-indices[k][0]+radar_mid,0,radar_mid)))*-1,np.arange(indices[k][0],len(radar_data),radar_mid)))
    
    
        lons_nwp=extracted_fields_nwp['lons']
        lats_nwp=extracted_fields_nwp['lats']
        rain_nwp=extracted_fields_nwp['values']
        for i in range(0,len(horizontal)):
            for j in range(0,len(vertical)):
                rain=radar_data[int(vertical[j]-np.floor(bound_radar)):int(vertical[j]+np.ceil(bound_radar)),int(horizontal[i]-np.floor(bound_radar)):int(horizontal[i]+np.ceil(bound_radar))]
                lons=radar_lons[int(vertical[j]-np.floor(bound_radar)):int(vertical[j]+np.ceil(bound_radar)),int(horizontal[i]-np.floor(bound_radar)):int(horizontal[i]+np.ceil(bound_radar))]
                lats=radar_lats[int(vertical[j]-np.floor(bound_radar)):int(vertical[j]+np.ceil(bound_radar)),int(horizontal[i]-np.floor(bound_radar)):int(horizontal[i]+np.ceil(bound_radar))]
                #radar_plot(rain, lons, lats, world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_hr[1][-1:])[-11:-9],str(Myfiles_hr[1][-1:])[-13:-11],str(Myfiles_hr[1][-1:])[-17:-13],str(Myfiles_hr[1][-1:])[-9:-7]),Myfiles_hr)
                lons_mid=radar_lons[vertical[j],horizontal[i]]
                lats_mid=radar_lats[vertical[j],horizontal[i]]
            
                #This resembles which NWP coordinates are the closest to the radar coordinates of the middle of the event
                coor_1=np.where(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)==np.min(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)))[0][0]
                coor_2=np.where(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)==np.min(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)))[1][0]
            
            #This makes sure that no areas but by boundary coordinates are used
                if coor_1==0 or coor_2==0 or coor_1==len(lats_nwp)-1 or coor_2==len(lons_nwp)-1 or int(coor_1-np.floor(bound_nwp))<=0 or int(coor_2-np.floor(bound_nwp))<=0 or coor_2>=690 or coor_1>=620:
                    pass
                else:
                    lats_nwp_new=lats_nwp[int(coor_1-np.floor(bound_nwp)):int(coor_1+np.ceil(bound_nwp)),int(coor_2-np.floor(bound_nwp)):int(coor_2+np.ceil(bound_nwp))]
                    lons_nwp_new=lons_nwp[int(coor_1-np.floor(bound_nwp)):int(coor_1+np.ceil(bound_nwp)),int(coor_2-np.floor(bound_nwp)):int(coor_2+np.ceil(bound_nwp))]
                    rain_nwp_new=rain_nwp[int(coor_1-np.floor(bound_nwp)):int(coor_1+np.ceil(bound_nwp)),int(coor_2-np.floor(bound_nwp)):int(coor_2+np.ceil(bound_nwp))]
                    rain_nwp_new=remove_values_below(rain_nwp_new, 0.5)
                    gridcells.append([i,j,lons,lats,lons_nwp_new,lats_nwp_new,coor_1,coor_2])
                    rain_nwp_out.append(rain_nwp_new)
                    rain_radar_out.append(rain)
    return gridcells,rain_nwp_out,rain_radar_out



def confusion_mat(twod_radar,twod_nwp):
    #Confusion matrix [1,1]=TP (hits), [0,0]=TN (correct nulls), [0,1]=FP (false alarm) , [1,0]=FN (miss)
    cm_final=[np.array([[0,0],[0,0]],dtype=int) for _ in range(len(twod_radar))]
    ETS=[[] for _ in range(len(twod_radar))]
    PSS=[[] for _ in range(len(twod_radar))]
    FBI=[[] for _ in range(len(twod_radar))]
    for i in range(len(twod_radar)):
        for j in range(len(twod_radar[i])):
            cm=confusion_matrix((twod_radar[i][j]>0)*1,(twod_nwp[i][j]>0)*1)
            cm_final[i]=cm_final[i]+cm
        
        a=cm_final[i][1,1]
        b=cm_final[i][0,1]
        c=cm_final[i][1,0]
        d=cm_final[i][0,0]
        a_ref=((a+b)*(a+c))/(a+b+c+d)
        ETS[i]=(a-a_ref)/((a-a_ref)+b+c)
        PSS[i]=((a*d)-(b*c))/((a+c)*(b+d))
        FBI[i]=(a+b)/(c+d)
    return cm_final,ETS,PSS,FBI

def squares_for_plot(grid_radar):
    radar_squares=[]
    if len(grid_radar[0])==1:
        corner=0
        right=0
        l=len(grid_radar[0])
    elif len(grid_radar[0])==9:
        corner=2
        right=8
        l=len(grid_radar[0])
    elif len(grid_radar[0])==25:
        corner=4
        right=24
        l=len(grid_radar[0])
    elif len(grid_radar[0])==49:
        corner=7
        right=48
        l=len(grid_radar[0])
    else:
        print("Error")
        right=-1
    for i in range(0,len(grid_radar)-1):
        #pdb.set_trace()
        if right==-1:
            break
        elif len(grid_radar[i])==0:
            continue
        elif len(grid_radar[i])!=l:
            continue
        else:
            low_lon=grid_radar[i][corner][0][np.shape(grid_radar[i][corner][0])[0]-1][0]
            low_lat=grid_radar[i][corner][1][np.shape(grid_radar[i][corner][1])[0]-1][0]
            width=-(grid_radar[i][corner][0][np.shape(grid_radar[i][corner][0])[0]-1][0]-grid_radar[i][right][0][np.shape(grid_radar[i][right][0])[0]-1][np.shape(grid_radar[i][right][0])[0]-1])
            height=-(grid_radar[i][corner][1][np.shape(grid_radar[i][corner][1])[0]-1][0]-grid_radar[i][0][1][0][0])
            radar_squares.append((low_lon,low_lat,width,height))
    df_squares=pd.DataFrame(radar_squares,columns=["low_lon","low_lat","width","height"])
    return df_squares

def plot_w_squares(rain_array,NWP_data, radar_lons, radar_lats, nwp_lons, nwp_lats, world_map_file, plot_title_radar,plot_title_nwp,squares):
    
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
    for i in range(0,len(squares)):
        axs[0].add_patch(Rectangle((squares['low_lon'][i],squares['low_lat'][i]),squares['width'][i],squares['height'][i],edgecolor="black",facecolor="none"))
    world_map.plot(facecolor="lightgrey",ax=axs[1])
    pcm=axs[1].pcolor(nwp_lons, nwp_lats, NWP_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
    axs[1].set_xlim(7,15)
    axs[1].set_ylim(54.5,58)
    for i in range(0,len(squares)):
        axs[1].add_patch(Rectangle((squares['low_lon'][i],squares['low_lat'][i]),squares['width'][i],squares['height'][i],edgecolor="black",facecolor="none"))
    axs[1].title.set_text(plot_title_nwp)
    plt.tick_params('y',labelleft=False)
    #axs[1].set_yticks([])
    cbar = fig.colorbar(pcm,ax=axs[1],fraction=0.040, pad=0.04)
    cbar.ax.set_ylabel('Rainfall intensity [mm/h]', rotation=90)
    plt.subplots_adjust(wspace=0.05,hspace=0)
    plt.show()
 

def spatial_agg(nwp_data,radar_data,radar_lats,radar_lons):
        distance_radar=15
        distance_nwp=10
        low=int(np.floor(distance_radar/2))
        high=int(np.ceil(distance_radar/2))
        nwp_low=int(np.floor(distance_nwp/2))
        nwp_high=int(np.ceil(distance_nwp/2))
        
        horizontal_nwp=np.arange(0,np.shape(nwp_data['lats'])[1],distance_nwp)
        vertical_nwp=np.arange(0,np.shape(nwp_data['lats'])[0],distance_nwp)
        inter_nwp=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
        inter_nwp_lats=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
        inter_nwp_lons=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
        nwp=[]
        radar=[]
        Radar_small_0=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
        Radar_small_1=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
        inter_radar=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
        inter_radar_lats=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
        inter_radar_lons=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
        
        #inter_nwp=np.reshape([np.nansum(nwp_data['values'][j-nwp_low:j+nwp_high,i-nwp_low:i+nwp_high])/np.size(nwp_data['values'][j-nwp_low:j+nwp_high,i-nwp_low:i+nwp_high]) for j in vertical_nwp for i in horizontal_nwp],(len(vertical_nwp),len(horizontal_nwp)))
        #inter_nwp_lats=np.reshape([nwp_data['lats'][j:j+1,i:i+1] for j in vertical_nwp for i in horizontal_nwp],(len(vertical_nwp),len(horizontal_nwp)))
        #inter_nwp_lons=np.reshape([nwp_data['lons'][j:j+1,i:i+1] for j in vertical_nwp for i in horizontal_nwp],(len(vertical_nwp),len(horizontal_nwp)))
        for i in range(1,len(horizontal_nwp)-1):
            for j in range(1,len(vertical_nwp)-1):
                inter_nwp[j,i]=np.nansum(nwp_data['values'][vertical_nwp[j]-nwp_low:vertical_nwp[j]+nwp_high,horizontal_nwp[i]-nwp_low:horizontal_nwp[i]+nwp_high])/np.size(nwp_data['values'][vertical_nwp[j]-nwp_low:vertical_nwp[j]+nwp_high,horizontal_nwp[i]-nwp_low:horizontal_nwp[i]+nwp_high])
                inter_nwp_lats[j,i]=nwp_data['lats'][vertical_nwp[j]:vertical_nwp[j]+1,horizontal_nwp[i]:horizontal_nwp[i]+1]
                inter_nwp_lons[j,i]=nwp_data['lons'][vertical_nwp[j]:vertical_nwp[j]+1,horizontal_nwp[i]:horizontal_nwp[i]+1]
                #start=time.time()
                Radar_small_0=np.where(np.abs(np.subtract(radar_lats,inter_nwp_lats[j,i]))+np.abs(np.subtract(radar_lons,inter_nwp_lons[j,i]))==np.min(np.abs(np.subtract(radar_lats,inter_nwp_lats[j,i]))+np.abs(np.subtract(radar_lons,inter_nwp_lons[j,i]))))[1][0]
                Radar_small_1=np.where(np.abs(np.subtract(radar_lats,inter_nwp_lats[j,i]))+np.abs(np.subtract(radar_lons,inter_nwp_lons[j,i]))==np.min(np.abs(np.subtract(radar_lats,inter_nwp_lats[j,i]))+np.abs(np.subtract(radar_lons,inter_nwp_lons[j,i]))))[0][0]
                inter_radar[j,i]=np.nansum(radar_data[int(Radar_small_0-low):int(Radar_small_0+high),int(Radar_small_1-low):int(Radar_small_1+high)])/np.size(radar_data[int(Radar_small_0-low):int(Radar_small_0+high),int(Radar_small_1-low):int(Radar_small_1+high)])
                inter_radar_lats[j,i]=radar_lats[int(Radar_small_0):int(Radar_small_0+1),int(Radar_small_1):int(Radar_small_1+1)]
                inter_radar_lons[j,i]=radar_lons[int(Radar_small_0):int(Radar_small_0+1),int(Radar_small_1):int(Radar_small_1+1)]
                #print(time.time()-start)
            print(i)
        radar.append((inter_radar,inter_radar_lats,inter_radar_lons))
        nwp.append((inter_nwp,inter_nwp_lats,inter_nwp_lons))
        for k in range(0,len(radar[0])):
            radar[0][k][radar[0][k]==0]=np.nan
            nwp[0][k][nwp[0][k]==0]=np.nan
        return nwp,radar
    
def produce_zonalstat(data_radar,radar_lons,radar_lats,files_radar,data_nwp,nwp_lon,nwp_lat,files_nwp,grid):
    zs_radar=[]
    zs_nwp=[]
    for i in range(0,len(data_radar)):
        tif_path_nwp="./Tiff_files/"+files_nwp[i+1][-3:]+'_NWP'+'.tif'
        data_to_raster_NWP(data_nwp[i+1],nwp_lon,nwp_lat,tif_path_nwp)
        
        tif_path_radar ="./Tiff_files/"+files_radar[i][0][-15:-5] + "_RADAR" + ".tif" # file path for a tif file that will be generated
        data_to_raster_RADAR(data_radar[i],radar_lons,radar_lats,tif_path_radar)
        print("done")
        #raster_radar=rasterio.open(tif_path_radar,masked=False)
        #raster_nwp=rasterio.open(tif_path_nwp,masked=False)
        
        #start=time.time()
        #zs_radar.append(zonal_stats("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI.shp",tif_path_radar,stats="sum"))
        #print(time.time()-start)
        #zs_nwp.append(zonal_stats("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI_NWP.shp",tif_path_nwp,stats="sum"))
        #print(time.time()-start) 
        
        start=time.time()
        zs_radar.append(zonal_statistics((tif_path_radar,1),"C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI.shp",ignore_nodata=False))
        print(time.time()-start)
        zs_nwp.append(zonal_statistics((tif_path_nwp,1),"C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI_NWP.shp",ignore_nodata=False))
        print(time.time()-start) 
    return zs_radar,zs_nwp


def extract_zonalstat(data_zonalstat,proportion):
    data_ts=np.empty((0,len(data_zonalstat)),np.float64)
    for i in range(0,len(data_zonalstat)):
        if np.isnan(data_zonalstat[i]['sum']):
            data_ts=np.append(data_ts,0.0)
        else:
            data_ts=np.append(data_ts,data_zonalstat[i]['sum']/proportion)
    return data_ts

def zone_to_2d(z_stat_radar,z_stat_nwp):
    radar_grid_2d=[[] for _ in range(len(z_stat_nwp))]
    nwp_grid_2d=[[] for _ in range(len(z_stat_radar))]
    for i in range(0,len(z_stat_radar)):
        start=time.time()
        radar_grid_2d[i]=np.reshape(extract_zonalstat(z_stat_radar[i],25),(125,157))
        nwp_grid_2d[i]=np.reshape(extract_zonalstat(z_stat_nwp[i],10),(125,157))
        radar_grid_2d[i][radar_grid_2d[i]==0]=np.nan
        nwp_grid_2d[i][nwp_grid_2d[i]==0]=np.nan
    return radar_grid_2d,nwp_grid_2d

def Fractional_skillscore(zs_radar,zs_nwp,domain,intensity=0,percentile=0,plot=False):
    avg_fss=[[] for _ in range(len(zs_radar))]
    fssrandom=0
    threshold_radar=[]
    qt_nwp=[[] for _ in range(len(zs_nwp))]
    for i in range(0,len(zs_radar)):
        #start=time.time()
        radar_grid_2d=np.reshape(extract_zonalstat(zs_radar[i],25),(125,157))
        nwp_grid_2d=np.reshape(extract_zonalstat(zs_nwp[i],10),(125,157))
        
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
        ci_low=[np.mean(k)-((np.std(k)/np.sqrt(len(avg_fss)))*t_dist) for k in zip(*avg_fss)]
        ci_high=[np.mean(k)+((np.std(k)/np.sqrt(len(avg_fss)))*t_dist) for k in zip(*avg_fss)]
        plt.figure()
        plt.fill_between(np.arange(0,domain,1)*2.5,ci_low,ci_high,alpha=0.5,color="grey")
        plt.plot(np.arange(0,domain,1)*2.5,FSS_mean)
        plt.plot(np.arange(0,domain,1)*2.5,fssrandom)
        plt.plot(np.arange(0,domain,1)*2.5,0.5+fssrandom/2)
        plt.ylim(0,1.1)
        plt.xlabel("Horizontal Scale [km]")
        plt.ylabel("Fractional Skill Score")
        plt.legend(["Confidence Interval","0.75km","FSS random","FSS useful"],loc="lower right")
        plt.show()
    
        plt.figure()
        for i in range(0,len(avg_fss)):
            plt.plot(np.arange(0,domain,1)*2.5,avg_fss[i])
        plt.plot(np.arange(0,domain,1)*2.5,fssrandom)
        plt.plot(np.arange(0,domain,1)*2.5,0.5+fssrandom/2)
        plt.ylim(0,1.1)
        plt.xlabel("Horizontal Scale [km]")
        plt.ylabel("Fractional Skill Score")
        plt.legend(["1","2","3","4","5","6","7","8","9","FSS random","FSS useful"],loc=0)
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
    cmap = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
    boundaries = [0.0 ,0.1, 0.2,0.3, 0.4,0.5,0.6,0.7, 0.8,0.9, 1.0]
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
