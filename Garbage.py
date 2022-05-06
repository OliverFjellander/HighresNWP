# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:43:55 2022

@author: olive
"""

#ALL the things I might need later

#######################Making FSS plot########################
# FSS_out=[]
# for i in range(1,6):
#     FSS,random_FSS=FSS_func(Radar_agg,radar_lons,radar_lats,extracted_nwp_coor,2,30,i)
#     FSS_uniform=[0.5+x/2 for x in random_FSS]
#     FSS_out.append(FSS)

# FSS_mean=[]
# FSS_mean.append([sum(sub_list) / len(sub_list) for i in FSS_out for sub_list in i])
# FSS_mean=[FSS_mean[0][i:i+9] for i in range(0,len(FSS_mean[0]),9)]

# plt.plot(np.arange(0,9,1),FSS_mean[0])
# plt.plot(np.arange(0,9,1),FSS_mean[1])
# plt.plot(np.arange(0,9,1),FSS_mean[2])
# plt.plot(np.arange(0,9,1),FSS_mean[3])
# plt.plot(np.arange(0,9,1),FSS_mean[4])
# plt.plot(random_FSS,color="black")
# plt.plot(FSS_uniform,color="black")
# plt.xlabel("Lead time [hr]")
# plt.ylabel("FSS")
# plt.legend(["NWP750 1","NWP750 2","NWP750 3","NWP750 4","NWP750 5","FSS random","FSS uniform"])

#saveFSS(FSS)


#########################Producing points and join within polygon#
#c=[]
#for i in range(0,np.shape(radar_lons)[0]):
#    for j in range(0,np.shape(radar_lons)[1]):
#        c.append(Point(radar_lons[i,j],radar_lats[i,j]))

#gdf=gpd.GeoDataFrame(np.asarray(Radar_agg[0].flatten(),dtype="float64"),geometry=gpd.points_from_xy(np.asarray(radar_lons.flatten(), dtype="float64"),np.asarray(radar_lats.flatten(), dtype="float64")))
#gdf_nwp=gpd.GeoDataFrame(np.asarray(extracted_nwp_coor[1]['values'].flatten(),dtype="float64"),geometry=gpd.points_from_xy(np.asarray(extracted_nwp_coor[1]['lons'].flatten(), dtype="float64"),np.asarray(extracted_nwp_coor[1]['lats'].flatten(), dtype="float64")))
#grid_gdf=gpd.GeoDataFrame(grid['geometry'])
#joined=gpd.sjoin(gdf,grid_gdf,how="inner",op="within")
#np.nanmax(joined.groupby(['index_right']).mean())


########################################
# def binary_to_confusion(rain_radar,rain_nwp):
#     radar_outside=[]
#     nwp_outside=[]
#     for i in range(0,len(rain_radar)):
#         #radar_inside=[]
#         radar_inside=([((np.sum((j>0)*1)/np.size(j))>0)*1 for j in rain_radar[i]])
#         nwp_inside=([((np.sum((j>0)*1)/np.size(j))>0)*1 for j in rain_nwp[i]])
#         #for j in range(0,len(rain_radar)):
#             #radar_inside.append(int(np.round(np.sum((rain_radar[i][j]>0)*1)/np.size(rain_radar[i][j]))))
#         radar_outside.append(int((np.sum(radar_inside)>0)*1))
#         nwp_outside.append(int((np.sum(nwp_inside)>0)*1))
#     return confusion_matrix(np.array(nwp_outside),np.array(radar_outside))


###########################################
# def spatial_agg(nwp_data,radar_data,radar_lats,radar_lons):
#         distance_radar=15
#         distance_nwp=10
#         low=int(np.floor(distance_radar/2))
#         high=int(np.ceil(distance_radar/2))
#         nwp_low=int(np.floor(distance_nwp/2))
#         nwp_high=int(np.ceil(distance_nwp/2))
        
#         horizontal_nwp=np.arange(0,np.shape(nwp_data['lats'])[1],distance_nwp)
#         vertical_nwp=np.arange(0,np.shape(nwp_data['lats'])[0],distance_nwp)
#         inter_nwp=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
#         inter_nwp_lats=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
#         inter_nwp_lons=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
#         nwp=[]
#         radar=[]
#         Radar_small_0=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
#         Radar_small_1=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
#         inter_radar=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
#         inter_radar_lats=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
#         inter_radar_lons=np.zeros((len(vertical_nwp),len(horizontal_nwp)))
        
#         #inter_nwp=np.reshape([np.nansum(nwp_data['values'][j-nwp_low:j+nwp_high,i-nwp_low:i+nwp_high])/np.size(nwp_data['values'][j-nwp_low:j+nwp_high,i-nwp_low:i+nwp_high]) for j in vertical_nwp for i in horizontal_nwp],(len(vertical_nwp),len(horizontal_nwp)))
#         #inter_nwp_lats=np.reshape([nwp_data['lats'][j:j+1,i:i+1] for j in vertical_nwp for i in horizontal_nwp],(len(vertical_nwp),len(horizontal_nwp)))
#         #inter_nwp_lons=np.reshape([nwp_data['lons'][j:j+1,i:i+1] for j in vertical_nwp for i in horizontal_nwp],(len(vertical_nwp),len(horizontal_nwp)))
#         for i in range(1,len(horizontal_nwp)-1):
#             for j in range(1,len(vertical_nwp)-1):
#                 inter_nwp[j,i]=np.nansum(nwp_data['values'][vertical_nwp[j]-nwp_low:vertical_nwp[j]+nwp_high,horizontal_nwp[i]-nwp_low:horizontal_nwp[i]+nwp_high])/np.size(nwp_data['values'][vertical_nwp[j]-nwp_low:vertical_nwp[j]+nwp_high,horizontal_nwp[i]-nwp_low:horizontal_nwp[i]+nwp_high])
#                 inter_nwp_lats[j,i]=nwp_data['lats'][vertical_nwp[j]:vertical_nwp[j]+1,horizontal_nwp[i]:horizontal_nwp[i]+1]
#                 inter_nwp_lons[j,i]=nwp_data['lons'][vertical_nwp[j]:vertical_nwp[j]+1,horizontal_nwp[i]:horizontal_nwp[i]+1]
#                 #start=time.time()
#                 Radar_small_0=np.where(np.abs(np.subtract(radar_lats,inter_nwp_lats[j,i]))+np.abs(np.subtract(radar_lons,inter_nwp_lons[j,i]))==np.min(np.abs(np.subtract(radar_lats,inter_nwp_lats[j,i]))+np.abs(np.subtract(radar_lons,inter_nwp_lons[j,i]))))[1][0]
#                 Radar_small_1=np.where(np.abs(np.subtract(radar_lats,inter_nwp_lats[j,i]))+np.abs(np.subtract(radar_lons,inter_nwp_lons[j,i]))==np.min(np.abs(np.subtract(radar_lats,inter_nwp_lats[j,i]))+np.abs(np.subtract(radar_lons,inter_nwp_lons[j,i]))))[0][0]
#                 inter_radar[j,i]=np.nansum(radar_data[int(Radar_small_0-low):int(Radar_small_0+high),int(Radar_small_1-low):int(Radar_small_1+high)])/np.size(radar_data[int(Radar_small_0-low):int(Radar_small_0+high),int(Radar_small_1-low):int(Radar_small_1+high)])
#                 inter_radar_lats[j,i]=radar_lats[int(Radar_small_0):int(Radar_small_0+1),int(Radar_small_1):int(Radar_small_1+1)]
#                 inter_radar_lons[j,i]=radar_lons[int(Radar_small_0):int(Radar_small_0+1),int(Radar_small_1):int(Radar_small_1+1)]
#                 #print(time.time()-start)
#             print(i)
#         radar.append((inter_radar,inter_radar_lats,inter_radar_lons))
#         nwp.append((inter_nwp,inter_nwp_lats,inter_nwp_lons))
#         for k in range(0,len(radar[0])):
#             radar[0][k][radar[0][k]==0]=np.nan
#             nwp[0][k][nwp[0][k]==0]=np.nan
#         return nwp,radar
####################
# def verification_alldomain(radar_data,radar_lons,radar_lats,extracted_fields_nwp,neighbourhood,intensity,threshold_qt=0.975):
    
#     #Neighbourhood=1 should represent squares of 7.5 km but I have had issues with needing to use np.floor and np.ceil below because some numbers are floats
#     bound_radar=neighbourhood*5*3/2
#     bound_nwp=neighbourhood*5
#     radar_mid=int(bound_radar*2)
    
#     lons_nwp=extracted_fields_nwp['lons']
#     lats_nwp=extracted_fields_nwp['lats']
#     rain_nwp=extracted_fields_nwp['values']
    
#     indices = peak_local_max(radar_data, min_distance=radar_mid,threshold_abs=intensity)
# #    indices = peak_local_max(radar_data, min_distance=radar_mid,threshold_abs=np.nanquantile(radar_data,threshold_qt))
#     rain_nwp_out=[]
#     rain_radar_out=[]
#     gridcells=[]
#     for k in range(0,1):
#         #So far just increased the area with 100 each way which should be 3-4 squares each direction (squares of 15 km dependent on neighbourhood)
#         horizontal=np.concatenate(((np.flip(np.arange(-indices[k][1]+radar_mid,0,radar_mid)))*-1,np.arange(indices[k][1],len(radar_data[1]),radar_mid)))
#         vertical=np.concatenate(((np.flip(np.arange(-indices[k][0]+radar_mid,0,radar_mid)))*-1,np.arange(indices[k][0],len(radar_data),radar_mid)))
    
    
#         lons_nwp=extracted_fields_nwp['lons']
#         lats_nwp=extracted_fields_nwp['lats']
#         rain_nwp=extracted_fields_nwp['values']
#         for i in range(0,len(horizontal)):
#             for j in range(0,len(vertical)):
#                 rain=radar_data[int(vertical[j]-np.floor(bound_radar)):int(vertical[j]+np.ceil(bound_radar)),int(horizontal[i]-np.floor(bound_radar)):int(horizontal[i]+np.ceil(bound_radar))]
#                 lons=radar_lons[int(vertical[j]-np.floor(bound_radar)):int(vertical[j]+np.ceil(bound_radar)),int(horizontal[i]-np.floor(bound_radar)):int(horizontal[i]+np.ceil(bound_radar))]
#                 lats=radar_lats[int(vertical[j]-np.floor(bound_radar)):int(vertical[j]+np.ceil(bound_radar)),int(horizontal[i]-np.floor(bound_radar)):int(horizontal[i]+np.ceil(bound_radar))]
#                 #radar_plot(rain, lons, lats, world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_hr[1][-1:])[-11:-9],str(Myfiles_hr[1][-1:])[-13:-11],str(Myfiles_hr[1][-1:])[-17:-13],str(Myfiles_hr[1][-1:])[-9:-7]),Myfiles_hr)
#                 lons_mid=radar_lons[vertical[j],horizontal[i]]
#                 lats_mid=radar_lats[vertical[j],horizontal[i]]
            
#                 #This resembles which NWP coordinates are the closest to the radar coordinates of the middle of the event
#                 coor_1=np.where(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)==np.min(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)))[0][0]
#                 coor_2=np.where(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)==np.min(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)))[1][0]
            
#             #This makes sure that no areas but by boundary coordinates are used
#                 if coor_1==0 or coor_2==0 or coor_1==len(lats_nwp)-1 or coor_2==len(lons_nwp)-1 or int(coor_1-np.floor(bound_nwp))<=0 or int(coor_2-np.floor(bound_nwp))<=0 or coor_2>=690 or coor_1>=620:
#                     pass
#                 else:
#                     lats_nwp_new=lats_nwp[int(coor_1-np.floor(bound_nwp)):int(coor_1+np.ceil(bound_nwp)),int(coor_2-np.floor(bound_nwp)):int(coor_2+np.ceil(bound_nwp))]
#                     lons_nwp_new=lons_nwp[int(coor_1-np.floor(bound_nwp)):int(coor_1+np.ceil(bound_nwp)),int(coor_2-np.floor(bound_nwp)):int(coor_2+np.ceil(bound_nwp))]
#                     rain_nwp_new=rain_nwp[int(coor_1-np.floor(bound_nwp)):int(coor_1+np.ceil(bound_nwp)),int(coor_2-np.floor(bound_nwp)):int(coor_2+np.ceil(bound_nwp))]
#                     rain_nwp_new=remove_values_below(rain_nwp_new, 0.5)
#                     gridcells.append([i,j,lons,lats,lons_nwp_new,lats_nwp_new,coor_1,coor_2])
#                     rain_nwp_out.append(rain_nwp_new)
#                     rain_radar_out.append(rain)
#     return gridcells,rain_nwp_out,rain_radar_out




####################################################
# def squares_for_plot(grid_radar):
#     radar_squares=[]
#     if len(grid_radar[0])==1:
#         corner=0
#         right=0
#         l=len(grid_radar[0])
#     elif len(grid_radar[0])==9:
#         corner=2
#         right=8
#         l=len(grid_radar[0])
#     elif len(grid_radar[0])==25:
#         corner=4
#         right=24
#         l=len(grid_radar[0])
#     elif len(grid_radar[0])==49:
#         corner=7
#         right=48
#         l=len(grid_radar[0])
#     else:
#         print("Error")
#         right=-1
#     for i in range(0,len(grid_radar)-1):
#         #pdb.set_trace()
#         if right==-1:
#             break
#         elif len(grid_radar[i])==0:
#             continue
#         elif len(grid_radar[i])!=l:
#             continue
#         else:
#             low_lon=grid_radar[i][corner][0][np.shape(grid_radar[i][corner][0])[0]-1][0]
#             low_lat=grid_radar[i][corner][1][np.shape(grid_radar[i][corner][1])[0]-1][0]
#             width=-(grid_radar[i][corner][0][np.shape(grid_radar[i][corner][0])[0]-1][0]-grid_radar[i][right][0][np.shape(grid_radar[i][right][0])[0]-1][np.shape(grid_radar[i][right][0])[0]-1])
#             height=-(grid_radar[i][corner][1][np.shape(grid_radar[i][corner][1])[0]-1][0]-grid_radar[i][0][1][0][0])
#             radar_squares.append((low_lon,low_lat,width,height))
#     df_squares=pd.DataFrame(radar_squares,columns=["low_lon","low_lat","width","height"])
#     return df_squares



##########################################
# def plot_w_squares(rain_array,NWP_data, radar_lons, radar_lats, nwp_lons, nwp_lats, world_map_file, plot_title_radar,plot_title_nwp,squares):
    
#     world_map = gpd.read_file(world_map_file)
    
#     # create custom color map
#     cmap = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
#     #boundaries = [0, .5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
#     boundaries = [0, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100]
#     norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    
#     # Make plot
#     #world_map.plot(facecolor="lightgrey")
#     fig,axs=plt.subplots(1,2,sharex=True,sharey=True,gridspec_kw={"width_ratios":[1,1.09]},figsize=(15,15))
#     fig.text(0.5,0.04,"Longitude",ha='center')
#     fig.text(0.04,0.5,"Latitude",va='center',rotation='vertical')
#     #fig.tight_layout(pad=3.0)
#     world_map.plot(facecolor="lightgrey",ax=axs[0])
#     pcm=axs[0].pcolor(radar_lons, radar_lats, rain_array, shading = "auto", alpha=1, cmap=cmap, norm=norm)
#     axs[0].set_xlim(7,15)
#     axs[0].set_ylim(54.5,58)
#     axs[0].title.set_text(plot_title_radar)
#     for i in range(0,len(squares)):
#         axs[0].add_patch(Rectangle((squares['low_lon'][i],squares['low_lat'][i]),squares['width'][i],squares['height'][i],edgecolor="black",facecolor="none"))
#     world_map.plot(facecolor="lightgrey",ax=axs[1])
#     pcm=axs[1].pcolor(nwp_lons, nwp_lats, NWP_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
#     axs[1].set_xlim(7,15)
#     axs[1].set_ylim(54.5,58)
#     for i in range(0,len(squares)):
#         axs[1].add_patch(Rectangle((squares['low_lon'][i],squares['low_lat'][i]),squares['width'][i],squares['height'][i],edgecolor="black",facecolor="none"))
#     axs[1].title.set_text(plot_title_nwp)
#     plt.tick_params('y',labelleft=False)
#     #axs[1].set_yticks([])
#     cbar = fig.colorbar(pcm,ax=axs[1],fraction=0.040, pad=0.04)
#     cbar.ax.set_ylabel('Rainfall intensity [mm/h]', rotation=90)
#     plt.subplots_adjust(wspace=0.05,hspace=0)
#     plt.show()
 

#####################################################
# #This contains scripts that uses data from multiple sources
# def verification_area(radar_data,radar_lons,radar_lats,extracted_fields_nwp,neighbourhood,intensity,scale,threshold_qt=0.975):
    
#     #Neighbourhood=1 should represent squares of 7.5 km but I have had issues with needing to use np.floor and np.ceil below because some numbers are floats
#     bound_radar=neighbourhood*5*3/2
#     bound_nwp=neighbourhood*5
#     radar_mid=int(bound_radar*2)
    
#     lons_nwp=extracted_fields_nwp['lons']
#     lats_nwp=extracted_fields_nwp['lats']
#     rain_nwp=extracted_fields_nwp['values']
    
#     indices = peak_local_max(radar_data, min_distance=radar_mid,threshold_abs=intensity)
# #    indices = peak_local_max(radar_data, min_distance=radar_mid,threshold_abs=np.nanquantile(radar_data,threshold_qt))
#     rain_nwp_out=[[] for _ in range(len(indices))]
#     rain_radar_out=[[] for _ in range(len(indices))]
#     gridcells_nwp=[[] for _ in range(len(indices))]
#     gridcells_radar=[[] for _ in range(len(indices))]
#     for k in range(0,len(indices)):
#         #So far just increased the area with 100 each way which should be 3-4 squares each direction (squares of 15 km dependent on neighbourhood)
#         horizontal=np.concatenate(((np.flip(np.arange(-indices[k][1]+radar_mid,-indices[k][1]+scale*radar_mid,radar_mid)))*-1,np.arange(indices[k][1],indices[k][1]+scale*radar_mid,radar_mid)))
#         vertical=np.concatenate(((np.flip(np.arange(-indices[k][0]+radar_mid,-indices[k][0]+scale*radar_mid,radar_mid)))*-1,np.arange(indices[k][0],indices[k][0]+scale*radar_mid,radar_mid)))
    
    
#         lons_nwp=extracted_fields_nwp['lons']
#         lats_nwp=extracted_fields_nwp['lats']
#         rain_nwp=extracted_fields_nwp['values']
#         for i in range(0,len(horizontal)):
#             for j in range(0,len(vertical)):
#                 rain=radar_data[int(vertical[j]-np.floor(bound_radar)):int(vertical[j]+np.ceil(bound_radar)),int(horizontal[i]-np.floor(bound_radar)):int(horizontal[i]+np.ceil(bound_radar))]
#                 lons=radar_lons[int(vertical[j]-np.floor(bound_radar)):int(vertical[j]+np.ceil(bound_radar)),int(horizontal[i]-np.floor(bound_radar)):int(horizontal[i]+np.ceil(bound_radar))]
#                 lats=radar_lats[int(vertical[j]-np.floor(bound_radar)):int(vertical[j]+np.ceil(bound_radar)),int(horizontal[i]-np.floor(bound_radar)):int(horizontal[i]+np.ceil(bound_radar))]
#                 #radar_plot(rain, lons, lats, world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_hr[1][-1:])[-11:-9],str(Myfiles_hr[1][-1:])[-13:-11],str(Myfiles_hr[1][-1:])[-17:-13],str(Myfiles_hr[1][-1:])[-9:-7]),Myfiles_hr)
#                 lons_mid=radar_lons[vertical[j],horizontal[i]]
#                 lats_mid=radar_lats[vertical[j],horizontal[i]]
            
#                 #This resembles which NWP coordinates are the closest to the radar coordinates of the middle of the event
#                 coor_1=np.where(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)==np.min(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)))[0][0]
#                 coor_2=np.where(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)==np.min(np.abs(lats_nwp-lats_mid)+np.abs(lons_nwp-lons_mid)))[1][0]
            
#             #This makes sure that no areas but by boundary coordinates are used
#                 if coor_1==0 or coor_2==0 or coor_1==len(lats_nwp)-1 or coor_2==len(lons_nwp)-1 or int(coor_1-np.floor(bound_nwp))<=0 or int(coor_2-np.floor(bound_nwp))<=0 or coor_2>=690 or coor_1>=620:
#                     pass
#                 else:
#                     lats_nwp_new=lats_nwp[int(coor_1-np.floor(bound_nwp)):int(coor_1+np.ceil(bound_nwp)),int(coor_2-np.floor(bound_nwp)):int(coor_2+np.ceil(bound_nwp))]
#                     lons_nwp_new=lons_nwp[int(coor_1-np.floor(bound_nwp)):int(coor_1+np.ceil(bound_nwp)),int(coor_2-np.floor(bound_nwp)):int(coor_2+np.ceil(bound_nwp))]
#                     rain_nwp_new=rain_nwp[int(coor_1-np.floor(bound_nwp)):int(coor_1+np.ceil(bound_nwp)),int(coor_2-np.floor(bound_nwp)):int(coor_2+np.ceil(bound_nwp))]
#                     rain_nwp_new=remove_values_below(rain_nwp_new, 0.5)
#                     gridcells_nwp[k].append([lons_nwp_new,lats_nwp_new])
#                     gridcells_radar[k].append([lons,lats])
#                     rain_nwp_out[k].append(rain_nwp_new)
#                     rain_radar_out[k].append(rain)
#     return gridcells_nwp,gridcells_radar,rain_nwp_out,rain_radar_out

#####################################################
# def FSS_func(radar_data,radar_lo,radar_la,nwp_coor,neighbourhood,intensity,scale,threshold_qt=0.975):
#     FSS=[[] for _ in range(len(radar_data))]
#     FSSrandom=[]
#     for k in range(0,len(radar_data)):
#         grid_nwp,grid_radar,precip_nwp,precip_radar=verification_area(np.nan_to_num(radar_data[k]),radar_lo,radar_la,nwp_coor[k+1],neighbourhood,intensity,scale,threshold_qt)       
#         FSSrandom.append(np.sum((radar_data[k]>0)*1)/np.size(radar_data[k]))
#         for i in range(0,len(precip_nwp)):
#             top=0
#             bottom_pf=0
#             bottom_po=0
#             for j in range(0,len(precip_nwp[i])):
#                 bin_radar=(precip_radar[i][j]>0)*1
#                 bin_nwp=(precip_nwp[i][j]>0)*1
#                 fraction_radar=np.sum(bin_radar)/np.size(bin_radar)
#                 fraction_nwp=np.sum(bin_nwp)/np.size(bin_nwp)
#                 top=top+(fraction_nwp-fraction_radar)**2
#                 bottom_pf=bottom_pf+(fraction_nwp)**2
#                 bottom_po=bottom_po+(fraction_radar)**2
#             if bottom_pf==0 and bottom_po==0:
#                pass
#             else:
#                 FSS[k].append(1-(top/len(precip_nwp))/((bottom_pf+bottom_po)/len(precip_nwp)))
#     return FSS,FSSrandom

###################################################
#from script
###################################################
# #grid,precip_nwp,precip_radar=verification_area(np.nan_to_num(Radar_agg[0]),radar_lons,radar_lats,extracted_nwp_coor[1],2,15,2)       


# #FSS
# nwp_newdom=[[] for _ in range(len(Radar_agg))]
# radar_newdom=[[] for _ in range(len(Radar_agg))]
# for i in range(0,len(Radar_agg)):
#     nwp_newdom[i],radar_newdom[i]=spatial_agg(extracted_nwp_coor[i+1],Radar_agg[i],radar_lats,radar_lons)
#     print(i)
 
    
# nwp_newdom,radar_newdom=spatial_agg(extracted_nwp_coor[1],Radar_agg[0],radar_lats,radar_lons)
# file_name="7.5 common domain"

# #open_file=open(file_name1,"wb")
# #pickle.dump([nwp_tst,radar_tst],open_file)
# #open_file.close()

# open_file=open(file_name,"rb")
# loaded_list=pickle.load(open_file)
# open_file.close()

# open_file1=open(file_name1,"rb")
# loaded_list1=pickle.load(open_file1)
# open_file1.close()

    
# #plot squares over verification areas
# plot_together(loaded_list[1][0][0],loaded_list[0][0][0], loaded_list[1][0][2], loaded_list[1][0][1], loaded_list[0][0][2],loaded_list[0][0][1], world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[0][-1:])[-11:-9],str(Myfiles_radar_hr[0][-1:])[-13:-11],str(Myfiles_radar_hr[0][-1:])[-17:-13],str(Myfiles_radar_hr[0][-1:])[-9:-7]),
#                   "NWP \n %s/%s/%s - %s:00 + %s UTC"%(Myfiles_nwp[0][-8:-6],Myfiles_nwp[0][-10:-8],Myfiles_nwp[0][-14:-10],Myfiles_nwp[0][-6:-4],str(1)),Myfiles_radar_hr[0],"./Pics0807/%s.png")


# FSS_random=np.repeat(np.sum((radar_grid_2d*1)>0)/np.size(radar_grid_2d),250)
# FSS_uniform=0.5+FSS_random/2
# plt.plot(avg_fss)
# plt.plot(FSS_random)
# plt.plot(FSS_uniform)
# plt.xlabel("Horizontal Scale (Pixels)")
# plt.ylabel("Fractional Skill Score (FSS)")
# plt.legend(["0.75km","FSS random","FSS uniform"])
# plt.show()
       

# pvs.fss(loaded_list[0][0][0],loaded_list[1][0][0],np.quantile(np.nan_to_num(np.append(loaded_list[1][0][0],loaded_list[0][0][0])),0.95),400)

# #Confusion matrix
# grid,precip_nwp,precip_radar=verification_alldomain(np.nan_to_num(Radar_agg[0]),radar_lons,radar_lats,extracted_nwp_coor[1],2,30)       
# cf=binary_to_confusion(precip_radar,precip_nwp)


# #PI index
# grid_nwp,grid_radar,precip_nwp_pi,precip_radar_pi=verification_area(np.nan_to_num(Radar_agg[1]),radar_lons,radar_lats,extracted_nwp_coor[2],2,30,2)       


# binary_to_confusion(precip_radar_pi,precip_nwp_pi)
# PI=precip_radar_tst[0][4][15,15]/((1/(2*15+1)**2-1)*np.sum(precip_radar_tst[0][4]-precip_radar_tst[0][4][15,15]))


# #plot squares over verification areas
# #plot_together(Radar_agg[0],extracted_nwp[1], radar_lons, radar_lats, extracted_nwp_coor[0]['lons'],extracted_nwp_coor[0]['lats'], world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[0][-1:])[-11:-9],str(Myfiles_radar_hr[0][-1:])[-13:-11],str(Myfiles_radar_hr[0][-1:])[-17:-13],str(Myfiles_radar_hr[0][-1:])[-9:-7]),
# #                  "NWP \n %s/%s/%s - %s:00 + %s UTC"%(Myfiles_nwp[0][-8:-6],Myfiles_nwp[0][-10:-8],Myfiles_nwp[0][-14:-10],Myfiles_nwp[0][-6:-4],str(1+i)),Myfiles_radar_hr[0],"./Pics0807/%s.png")


# #sq=squares_for_plot(grid_radar)
# #plot_w_squares(Radar_agg[0],extracted_nwp[1], radar_lons, radar_lats, extracted_nwp_coor[0]['lons'],extracted_nwp_coor[0]['lats'], world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[0][-1:])[-11:-9],str(Myfiles_radar_hr[0][-1:])[-13:-11],str(Myfiles_radar_hr[0][-1:])[-17:-13],str(Myfiles_radar_hr[0][-1:])[-9:-7]),"NWP \n %s/%s/%s - %s:00 + %s UTC"%(Myfiles_nwp[0][-8:-6],Myfiles_nwp[0][-10:-8],Myfiles_nwp[0][-14:-10],Myfiles_nwp[0][-6:-4],str(1+i)),df_squares)
# for i in range(0,len(Radar_agg)):
#    grid_nwp_sq,grid_radar_sq,precip_nwp_sq,precip_radar_sq=verification_area(np.nan_to_num(Radar_agg[i]),radar_lons,radar_lats,extracted_nwp_coor[i+1],2,30,2)       
#    sq=squares_for_plot(grid_radar_sq)
#    plot_w_squares(Radar_agg[i],extracted_nwp[i+1], radar_lons, radar_lats, extracted_nwp_coor[0]['lons'],extracted_nwp_coor[0]['lats'], world_map_file, "Radar \n %s/%s/%s - %s:00 UTC"%(str(Myfiles_radar_hr[0][-1:])[-11:-9],str(Myfiles_radar_hr[0][-1:])[-13:-11],str(Myfiles_radar_hr[0][-1:])[-17:-13],str(Myfiles_radar_hr[0][-1:])[-9:-7]),"NWP \n %s/%s/%s - %s:00 + %s UTC"%(Myfiles_nwp[0][-8:-6],Myfiles_nwp[0][-10:-8],Myfiles_nwp[0][-14:-10],Myfiles_nwp[0][-6:-4],str(1+i)),sq)
########################################

#grid_NWP=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_DMI_NWP.shp")

#coor_grid="C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/DMI_grid_coor1.shp"
#coor_grid_NWP="C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_NWP_coor.shp"

#grid_25_coor=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/DMI_grid_coor1.shp")
#grid_25_coor_NWP=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_NWP_coor.shp")
#grid_NWP['ycoor']=grid_25_coor_NWP['ycoord']
#grid_NWP['xcoor']=grid_25_coor_NWP['xcoord']

#grid_small=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_northernzealand.shp")
#grid_small_coor=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/coor_nothernzealand.shp")
#grid_small_nwp=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_northernzealand_nwp.shp")
#grid_small_nwp['ycoor']=grid_small_coor['ycoord']
#grid_small_nwp['xcoor']=grid_small_coor['xcoord']

#grid_coor=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/grid_op.shp") #In original rotated lat/lon
#grid=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/25gridnwp.shp") #In original rotated lat/lon

#grid_radar=gpd.read_file("C:/Users/olive/Desktop/Speciale/Dokumenter/Rain_observation_network/Rain_gauge_network/25gridradar.shp")

###############################################

# Make a raster that the data can be inserted into. Produces a tiff file.
# def data_array_to_raster(data_array, tif_path):
#     #transform = rasterio.transform.from_origin(-422114.8, 469381, 500, 500) # define coordinates for the DMI grid
#     transform = rasterio.transform.from_origin(-174865, 263131, 500, 500) # define coordinates for the DMI grid
#     proj4string_dmistere = '+proj=stere +ellps=WGS84 +lat_0=56 +lon_0=10.5666 +lat_ts=56' # The raw data's projection
#     #transform = rasterio.transform.from_origin(7.582964884675271,58.32806110474353,0.008519049771461227, 0.004481469249843848) # define coordinates for the DMI grid
#     #proj4string_dmistere = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs " # The raw data's projection

#     # Produce raster with rasterio package
#     with rasterio.open(tif_path, 'w', driver='GTiff',
#                        height = data_array.shape[0], width = data_array.shape[1],
#                        count=1, dtype=str(data_array.dtype),
#                        crs=proj4string_dmistere,
#                        transform=transform) as file:
#         file.write(data_array, 1)
    
#     raster_array = rxr.open_rasterio('./{}'.format(tif_path),tif_path).squeeze() # load data
    
#     with rxr.open_rasterio(tif_path) as file:
#         raster_array = file.squeeze()
    
#     #raster_array.close()
#     #os.remove('./data_array.tif')
    
#     return(raster_array)
    
#####################################################################

# [Not recommended to use this function!] Function that reprojects a raster
# Be careful with this function. It can reproject a raster, but it changes the shape of the array and thus the data resolution!
# def reproject_raster(raster_data, destination_epsg_crs):
#     dest_crs = rasterio.crs.CRS.from_epsg(destination_epsg_crs) # define object for target crs
#     raster_projected = raster_data.rio.reproject(dest_crs)
#     raster_projected.data[raster_projected.data==np.nanmin(raster_projected.data)] = np.nan # reprojection creates -9999 values, that need to be removed
#     return(raster_projected)


###########################################################################
####################### WORKS FINE BUT IS JUST NOT WHAT I NEED#############
  
# def plot_all_samegrid(radar_data,nwp_data,nea_data, lons, lats, world_map_file, plot_title_radar,plot_title_nwp,plot_title_nea,save_name,timestep,preciptype):
    
#     world_map = gpd.read_file(world_map_file)
    
#     # create custom color map
#     cmap = colors.ListedColormap(["#85E3E4", '#42D8D8', '#42AFD8', '#4282D8', "#FFE600", '#FFAF00', '#FF5050', '#FF1A1A', "#BD0000", "#8C0000"])
#     #boundaries = [0, .5, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
#     boundaries = [0, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100]
#     norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    
#     # Make plot
#     #world_map.plot(facecolor="lightgrey")
#     fig,axs=plt.subplots(1,3,sharex=True,sharey=True,gridspec_kw={"width_ratios":[1,1,1.09]},figsize=(16,8))
#     fig.text(0.5,0.2,"Longitude",ha='center')
#     fig.text(0.06,0.5,"Latitude",va='center',rotation='vertical')
#     #fig.tight_layout(pad=3.0)
#     world_map.plot(facecolor="lightgrey",ax=axs[0])
#     pcm=axs[0].pcolor(lons, lats, radar_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
#     axs[0].set_xlim(7,15)
#     axs[0].set_ylim(54.5,58)

#     axs[0].title.set_text(plot_title_radar)
#     world_map.plot(facecolor="lightgrey",ax=axs[1])
#     pcm=axs[1].pcolor(lons, lats, nwp_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
#     axs[1].set_xlim(7,15)
#     axs[1].set_ylim(54.5,58)

#     axs[1].title.set_text(plot_title_nwp)
#     axs[1].tick_params('y',labelleft=False,left=False)
#     #axs[1].set_yticks([])
#     world_map.plot(facecolor="lightgrey",ax=axs[2])
#     pcm=axs[2].pcolor(lons, lats, nea_data, shading = "auto", alpha=1, cmap=cmap, norm=norm)
#     axs[2].set_xlim(7,15)
#     axs[2].set_ylim(54.5,58)
#     axs[2].title.set_text(plot_title_nea)
#     axs[2].tick_params('y',labelleft=False,left=False)
#     cbar = fig.colorbar(pcm,ax=axs[2],fraction=0.040, pad=0.04)
#     cbar.ax.set_ylabel('Rainfall intensity [mm/h]', rotation=90)
#     plt.subplots_adjust(wspace=0.05,hspace=0)
#     plt.show()
#     current=os.getcwd()
#     os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Pics/%s"%preciptype) 
#     #newpath="r"+save_path[:-7]%str(save_name[-6:-2])
#     newpath=r'./%s/'%save_name[-6:-2]
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)
#     os.chdir("C:/Users/olive/Desktop/Speciale/Kode/Pics/%s/%s"%(preciptype,newpath[-6:-1]))
#     save_name=os.getcwd()+"/"+str(save_name[-2:])+"_"+str(timestep)
#     if os.path.isfile(save_name):
#         os.remove(save_name)
#     plt.savefig(save_name,bbox_inches='tight')
#     plt.close()
#     os.chdir(current)

# def produce_gifs_samegrid(dates,preciptype):
#     os.chdir("C:/Users/olive/Desktop/Speciale/Kode") 
#     for i in range(0,len(np.unique(dates))):
#         date=str(np.unique(dates)[i])
#         try:
#             load_radar=np.loadtxt("./25grid/Radar/radar_20%s.txt"%date[:-2])
#         except FileNotFoundError:
#             continue

#         load_NEA=np.loadtxt("./25grid/NEA/nea_2d_%s.txt"%date)
#         load_nwp=np.loadtxt("./25grid/NWP750/nwp_2d_%s.txt"%date)
#         load_radar=np.loadtxt("./25grid/Radar/radar_20%s.txt"%date[:-2])

#         orig_nea=load_NEA.reshape(load_NEA.shape[0],load_NEA.shape[1] // 184, 184)
#         orig_nwp=load_nwp.reshape(load_nwp.shape[0],load_nwp.shape[1] // 184, 184)
#         orig_radar=load_radar.reshape(load_radar.shape[0],load_radar.shape[1] // 184, 184)

#         for timestep in range(0,len(orig_nea)): 
#             if (int(date[-2:])+timestep)>23:
#                 continue
#             else:
#                 radar_format="Radar \n %s/%s/%s - %s:00 UTC"%(str('20')+date[:2],date[2:4],date[4:6],str(int(date[-2:])+timestep))
#                 nwp_format="NWP750 \n %s/%s/%s - %s:00 + %s UTC"%(str('20')+date[:2],date[2:4],date[4:6],date[-2:],str(timestep))
#                 nea_format="NWP2500 \n %s/%s/%s - %s:00 + %s UTC"%(str('20')+date[:2],date[2:4],date[4:6],date[-2:],str(timestep))    
#                 plot_all_samegrid(orig_radar[int(date[-2:])+timestep],orig_nwp[timestep],orig_nea[timestep],lons_25, lats_25,world_map_file,radar_format, nwp_format, nea_format, str(date),str(timestep),preciptype)

#         make_gif("./Pics/%s/%s/*.png"%(preciptype,date[2:6]),'./Pics/%s/gifs/%s.gif'%(preciptype,date),2)


# produce_gifs_samegrid(cloudburst_dates,"cloudburst")
# produce_gifs_samegrid(severe_dates,"severerain")
#########################################################################################################
#########################################################################################################

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

##########################################################################################################
############################################################################
