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
