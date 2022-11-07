import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import time

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from haversine import haversine, inverse_haversine, Direction, Unit

from functions import *
from LiRA_functions import *

#################################################### Crossvalidation ###############################################################################

lat_len = 1# 111332.67
lon_len = 1#63195.85

print('synth_test')
synth_acc = synthetic_data()
routes = []
for i in range(len(synth_acc)): 
    routes.append(synth_acc[i].axes[0].name)

synth_segments, aran_segments, route_details = synthetic_segmentation(synth_acc,routes,segment_size=5,overlap=0)

features,feature_names = feature_extraction(synth_segments,'synth_data/extracted_features',fs=250)

cut = [10500,18500,15000]

cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split3')

fig,ax = plt.subplots(1,2)
for i in range(len(splits)):
    if i < 3:
        ax[0].scatter(x=aran_segments['LongitudeFrom'][splits[str(i+1)]],y=aran_segments['LatitudeFrom'][splits[str(i+1)]],s=1,label="split: "+str(i+1))
    else:
        ax[1].scatter(x=aran_segments['LongitudeFrom'][splits[str(i+1)]],y=aran_segments['LatitudeFrom'][splits[str(i+1)]],s=1,label="split: "+str(i+1))
ax[0].legend()
ax[1].legend()
plt.show()


####################################################################################################################################################

#################################################### SEGMENTATION PLOTS ############################################################################
synth_acc = synthetic_data()
routes = []
for i in range(len(synth_acc)): 
    routes.append(synth_acc[i].axes[0].name)

synth_segments, aran_segments, route_details = synthetic_segmentation(synth_acc,routes,segment_size=5,overlap=0)


hdf5file = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

DI, allig, cracks, potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)

seg_len = 1 
seg_cap = 0
segm_nr = 0
DI = []
alligator = []
cracks = []
potholes = []
for i in tqdm(range(int(np.shape(aran_segments)[0]/seg_len))):
    aran_details = aran_segments.iloc[i*seg_len:i*seg_len+seg_cap+1]
    aran_alligator = aran_details[aran_alligator.columns]
    aran_cracks = aran_details[aran_cracks.columns]
    aran_potholes = aran_details[aran_potholes.columns]
    temp_DI, temp_alligator, temp_cracks, temp_potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
    DI.append(np.max(temp_DI))
    alligator.append(np.max(temp_alligator))
    cracks.append(np.max(temp_cracks))
    potholes.append(np.max(temp_potholes))


idx_max_DI = np.argmax(DI)
idx_min_DI = np.argmin(DI)
# idx_min_DI = 3541

cr = np.array(cracks)
cr = cr[(np.array(alligator) < 0.1) & (np.array(potholes) < 0.1)]
idx_max_cracks = cracks.index(np.max(cr))
idx_max_cracks = 24474-0

all = np.array(alligator)
all = all[(np.array(cracks) < 0.1) & (np.array(potholes) < 0.1)]
idx_max_alligator = alligator.index(np.max(all))
idx_max_alligator = alligator.index(all[38])
idx_max_alligator = 30774

pot = np.array(potholes)
# pot = pot[(np.array(alligator) < 2) & (np.array(cracks) < 1.5)]
idx_max_potholes = potholes.index(np.max(pot)) # 454
# idx_max_potholes = int(33590/seg_len)

pots, crac, alli = 0, 0, 0
for j in range(len(potholes)):
    if potholes[j] > 0.0:
        pots += 1
    if cracks[j] > 0.0:
        crac += 1
    if alligator[j] > 0.0:
        alli += 1


route = route_details[int(idx_max_potholes*(seg_len/5))][:7]
p79_gps = pd.read_csv("p79/"+route+".csv")
synth = synth_acc[routes.index(route_details[int(idx_max_potholes*(seg_len/5))])]
synth = synth[synth['synth_acc'].notna()]
synth = synth[synth['gm_speed'] >= 20]
synth = synth.reset_index(drop=True)
aran_start = [aran_segments['LatitudeFrom'][idx_max_potholes*seg_len],aran_segments['LongitudeFrom'][ idx_max_potholes*seg_len]]
aran_end = [aran_segments['LatitudeTo'][ idx_max_potholes*seg_len+seg_cap],aran_segments['LongitudeTo'][ idx_max_potholes*seg_len+seg_cap]]
p79_start_idx, _ = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
p79_end_idx, _ = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
p79_details_max_potholes = p79_gps[p79_start_idx:p79_end_idx+1]
start_dist = p79_details_max_potholes['Distance'].iloc[0]
end_dist = p79_details_max_potholes['Distance'].iloc[-1]
acc_synth_max_potholes = synth[ (synth['Distance']<=end_dist) & (synth['Distance']>=start_dist)]



route = route_details[int(idx_max_cracks*(seg_len/5))][:7]
p79_gps = pd.read_csv("p79/"+route+".csv")
synth = synth_acc[routes.index(route_details[int(idx_max_cracks*(seg_len/5))])]
synth = synth[synth['synth_acc'].notna()]
synth = synth[synth['gm_speed'] >= 20]
synth = synth.reset_index(drop=True)
aran_start = [aran_segments['LatitudeFrom'][ idx_max_cracks*seg_len],aran_segments['LongitudeFrom'][ idx_max_cracks*seg_len]]
aran_end = [aran_segments['LatitudeTo'][ idx_max_cracks*seg_len+seg_cap],aran_segments['LongitudeTo'][ idx_max_cracks*seg_len+seg_cap]]
p79_start_idx, _ = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
p79_end_idx, _ = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
p79_details_max_cracks = p79_gps[p79_start_idx:p79_end_idx+1]
start_dist = p79_details_max_cracks['Distance'].iloc[0]
end_dist = p79_details_max_cracks['Distance'].iloc[-1]
acc_synth_max_cracks = synth[ (synth['Distance']<=end_dist) & (synth['Distance']>=start_dist)]



route = route_details[int(idx_max_alligator*(seg_len/5))][:7]
p79_gps = pd.read_csv("p79/"+route+".csv")
synth = synth_acc[routes.index(route_details[int(idx_max_alligator*(seg_len/5))])]
synth = synth[synth['synth_acc'].notna()]
synth = synth[synth['gm_speed'] >= 20]
synth = synth.reset_index(drop=True)
aran_start = [aran_segments['LatitudeFrom'][ idx_max_alligator*seg_len],aran_segments['LongitudeFrom'][ idx_max_alligator*seg_len]]
aran_end = [aran_segments['LatitudeTo'][ idx_max_alligator*seg_len+seg_cap],aran_segments['LongitudeTo'][ idx_max_alligator*seg_len+seg_cap]]
p79_start_idx, _ = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
p79_end_idx, _ = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
p79_details_max_alligator = p79_gps[p79_start_idx:p79_end_idx+1]
start_dist = p79_details_max_alligator['Distance'].iloc[0]
end_dist = p79_details_max_alligator['Distance'].iloc[-1]
acc_synth_max_alligator = synth[ (synth['Distance']<=end_dist) & (synth['Distance']>=start_dist)]




route = route_details[int(idx_max_DI*(seg_len/5))][:7]
p79_gps = pd.read_csv("p79/"+route+".csv")
synth = synth_acc[routes.index(route_details[int(idx_max_DI*(seg_len/5))])]
synth = synth[synth['synth_acc'].notna()]
synth = synth[synth['gm_speed'] >= 20]
synth = synth.reset_index(drop=True)
aran_start = [aran_segments['LatitudeFrom'][ idx_max_DI*seg_len],aran_segments['LongitudeFrom'][ idx_max_DI*seg_len]]
aran_end = [aran_segments['LatitudeTo'][ idx_max_DI*seg_len+seg_cap],aran_segments['LongitudeTo'][ idx_max_DI*seg_len+seg_cap]]
p79_start_idx, _ = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
p79_end_idx, _ = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
p79_details_max_DI = p79_gps[p79_start_idx:p79_end_idx+1]
start_dist = p79_details_max_DI['Distance'].iloc[0]
end_dist = p79_details_max_DI['Distance'].iloc[-1]
acc_synth_max_DI = synth[ (synth['Distance']<=end_dist) & (synth['Distance']>=start_dist)]


route = route_details[int(idx_min_DI*(seg_len/5))][:7]
p79_gps = pd.read_csv("p79/"+route+".csv")
synth = synth_acc[routes.index(route_details[int(idx_min_DI*(seg_len/5))])]
synth = synth[synth['synth_acc'].notna()]
synth = synth[synth['gm_speed'] >= 20]
synth = synth.reset_index(drop=True)
aran_start = [aran_segments['LatitudeFrom'][ idx_min_DI*seg_len],aran_segments['LongitudeFrom'][ idx_min_DI*seg_len]]
aran_end = [aran_segments['LatitudeTo'][ idx_min_DI*seg_len+seg_cap],aran_segments['LongitudeTo'][ idx_min_DI*seg_len+seg_cap]]
p79_start_idx, _ = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
p79_end_idx, _ = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
p79_details_min_DI = p79_gps[p79_start_idx:p79_end_idx+1]
start_dist = p79_details_min_DI['Distance'].iloc[0]
end_dist = p79_details_min_DI['Distance'].iloc[-1]
acc_synth_min_DI = synth[ (synth['Distance']<=end_dist) & (synth['Distance']>=start_dist)]

laser = np.concatenate([p79_details_max_potholes[['Laser5','Laser21']].values, p79_details_max_cracks[['Laser5','Laser21']].values, p79_details_max_alligator[['Laser5','Laser21']].values, p79_details_max_DI[['Laser5','Laser21']].values, p79_details_min_DI[['Laser5','Laser21']].values])
laser_max = np.max(laser)
laser_min = np.min(laser)

acc = np.concatenate([acc_synth_max_potholes['synth_acc'],acc_synth_max_cracks['synth_acc'],acc_synth_max_alligator['synth_acc'],acc_synth_max_DI['synth_acc'],acc_synth_min_DI['synth_acc']])
acc_max = np.max(acc)
acc_min = np.min(acc)

fig, axs = plt.subplots(2,5)
dist = np.concatenate([p79_details_max_potholes['Distance'],acc_synth_max_potholes['Distance']])
axs[0,0].plot(p79_details_max_potholes['Distance'],p79_details_max_potholes['Laser5'],label='Laser 5')
axs[0,0].plot(p79_details_max_potholes['Distance'],p79_details_max_potholes['Laser21'],label='Laser 21')
axs[0,0].plot(p79_details_max_potholes['Distance'],(p79_details_max_potholes['Laser21']+p79_details_max_potholes['Laser5'])/2,color='green',label='quarter car')
axs[0,0].set(xlim=(np.min(dist),np.max(dist)),ylim=(laser_min,laser_max))
axs[0,0].legend()
axs[0,0].set_title('Potholes: ' + str(round(potholes[idx_max_potholes],2))+'\n Cracks: '+str(round(cracks[idx_max_potholes],2))+'\n Alligator: '+str(round(alligator[idx_max_potholes],2)))
axs[1,0].plot(acc_synth_max_potholes['Distance'],acc_synth_max_potholes['synth_acc'])
axs[1,0].set(xlim=(np.min(dist),np.max(dist)),ylim=(acc_min,acc_max))

dist = np.concatenate([p79_details_max_cracks['Distance'],acc_synth_max_cracks['Distance']])
axs[0,1].plot(p79_details_max_cracks['Distance'],p79_details_max_cracks['Laser5'],label='Laser 5')
axs[0,1].plot(p79_details_max_cracks['Distance'],p79_details_max_cracks['Laser21'],label='Laser 21')
axs[0,1].plot(p79_details_max_cracks['Distance'],(p79_details_max_cracks['Laser21']+p79_details_max_cracks['Laser5'])/2,color='green',label='quarter car')
axs[0,1].set(xlim=(np.min(dist),np.max(dist)),ylim=(laser_min,laser_max))
axs[0,1].legend()
axs[0,1].set_title('Potholes: ' + str(round(potholes[idx_max_cracks],2))+'\n Cracks: '+str(round(cracks[idx_max_cracks],2))+'\n Alligator: '+str(round(alligator[idx_max_cracks],2)))
axs[1,1].plot(acc_synth_max_cracks['Distance'],acc_synth_max_cracks['synth_acc'])
axs[1,1].set(xlim=(np.min(dist),np.max(dist)),ylim=(acc_min,acc_max))

dist = np.concatenate([p79_details_max_alligator['Distance'],acc_synth_max_alligator['Distance']])
axs[0,2].plot(p79_details_max_alligator['Distance'],p79_details_max_alligator['Laser5'],label='Laser 5')
axs[0,2].plot(p79_details_max_alligator['Distance'],p79_details_max_alligator['Laser21'],label='Laser 21')
axs[0,2].plot(p79_details_max_alligator['Distance'],(p79_details_max_alligator['Laser21']+p79_details_max_alligator['Laser5'])/2,color='green',label='quarter car')
axs[0,2].set(xlim=(np.min(dist),np.max(dist)),ylim=(laser_min,laser_max))
axs[0,2].legend()
axs[0,2].set_title('Potholes: ' + str(round(potholes[idx_max_alligator],2))+'\n Cracks: '+str(round(cracks[idx_max_alligator],2))+'\n Alligator: '+str(round(alligator[idx_max_alligator],2)))
axs[1,2].plot(acc_synth_max_alligator['Distance'],acc_synth_max_alligator['synth_acc'])
axs[1,2].set(xlim=(np.min(dist),np.max(dist)),ylim=(acc_min,acc_max))

dist = np.concatenate([p79_details_max_DI['Distance'],acc_synth_max_DI['Distance']])
axs[0,3].plot(p79_details_max_DI['Distance'],p79_details_max_DI['Laser5'],label='Laser 5')
axs[0,3].plot(p79_details_max_DI['Distance'],p79_details_max_DI['Laser21'],label='Laser 21')
axs[0,3].plot(p79_details_max_DI['Distance'],(p79_details_max_DI['Laser21']+p79_details_max_DI['Laser5'])/2,color='green',label='quarter car')
axs[0,3].set(xlim=(np.min(dist),np.max(dist)),ylim=(laser_min,laser_max))
axs[0,3].legend()
axs[0,3].set_title('High DI: '+str(round(DI[idx_max_DI],2)))
axs[1,3].plot(acc_synth_max_DI['Distance'],acc_synth_max_DI['synth_acc'])
axs[1,3].set(xlim=(np.min(dist),np.max(dist)),ylim=(acc_min,acc_max))

dist = np.concatenate([p79_details_min_DI['Distance'],acc_synth_min_DI['Distance']])
axs[0,4].plot(p79_details_min_DI['Distance'],p79_details_min_DI['Laser5'],label='Laser 5')
axs[0,4].plot(p79_details_min_DI['Distance'],p79_details_min_DI['Laser21'],label='Laser 21')
axs[0,4].plot(p79_details_min_DI['Distance'],(p79_details_min_DI['Laser21']+p79_details_min_DI['Laser5'])/2,color='green',label='quarter car')
axs[0,4].set(xlim=(np.min(dist),np.max(dist)),ylim=(laser_min,laser_max))
axs[0,4].legend()
axs[0,4].set_title('Low DI: '+str(round(DI[idx_min_DI],2)))
axs[1,4].plot(acc_synth_min_DI['Distance'],acc_synth_min_DI['synth_acc'])
axs[1,4].set(xlim=(np.min(dist),np.max(dist)),ylim=(acc_min,acc_max))

axs[0,0].set(ylabel='Laser distance [mm]')
axs[1,0].set(ylabel='Synthetic z acceleration')
for ax in axs.flat:
    ax.set(xlabel='Distance [m]')
# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()
plt.show()


# ###################################################################################################################################
    
# ################################################## HISTOGRAM PLOTS - Aran #####################################################
# # synth_acc = synthetic_data()
# # routes = []
# # for i in range(len(synth_acc)): 
# #     routes.append(synth_acc[i].axes[0].name)
# # synth_segments, aran_segments, route_details = synthetic_segmentation(synth_acc,routes,segment_size=5,overlap=0)

# # hdf5file = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
# # aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
# # aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
# # aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
# # aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

# # DI, allig, cracks, potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)

# # seg_len = 5 
# # seg_cap = 4
# # segm_nr = 0
# # DI = []
# # alligator = []
# # cracks = []
# # potholes = []
# # for i in tqdm(range(int(np.shape(aran_segments)[0]/seg_len))):
# #     aran_details = aran_segments.iloc[i*seg_len:i*seg_len+seg_cap+1]
# #     aran_alligator = aran_details[aran_alligator.columns]
# #     aran_cracks = aran_details[aran_cracks.columns]
# #     aran_potholes = aran_details[aran_potholes.columns]
# #     temp_DI, temp_alligator, temp_cracks, temp_potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
# #     DI.append(np.max(temp_DI))
# #     alligator.append(np.max(temp_alligator))
# #     cracks.append(np.max(temp_cracks))
# #     potholes.append(np.max(temp_potholes))


# # pots, crac, alli, dam = 0, 0, 0, 0
# # for j in range(len(potholes)):
# #     if potholes[j] > 0.0:
# #         pots += 1
# #     if cracks[j] > 0.0:
# #         crac += 1
# #     if alligator[j] > 0.0:
# #         alli += 1
# #     if DI[j] > 0.0:
# #         dam += 1

# # DI = np.array(DI)
# # alligator = np.array(alligator)
# # cracks = np.array(cracks)
# # potholes = np.array(potholes)

# # fig, axs = plt.subplots(2,2)
# # axs[0,0].hist(DI[DI > 0.0],bins=50)
# # axs[0,0].set_title('DI: '+str(dam))
# # axs[0,0].set(ylim=(0,600))
# # axs[1,0].hist(alligator[alligator > 0.0],bins=50)
# # axs[1,0].set_title('Alligator: '+str(alli))
# # axs[1,0].set(ylim=(0,600))
# # axs[0,1].hist(cracks[cracks > 0.0],bins=50)
# # axs[0,1].set_title('Cracks: '+str(crac))
# # axs[0,1].set(ylim=(0,600))
# # axs[1,1].hist(potholes[potholes > 0.0],bins=50)
# # axs[1,1].set_title('Potholes: '+str(pots))
# # axs[1,1].set(ylim=(0,50))
# # fig.suptitle('Total nr of segmentations: '+str(len(route_details)))
# # plt.show()

# # fig, axs = plt.subplots(2,2)
# # axs[0,0].hist(DI,bins=50)
# # axs[0,0].set_title('DI: '+str(dam))
# # axs[1,0].hist(alligator,bins=50)
# # axs[1,0].set_title('Alligator: '+str(alli))
# # axs[0,1].hist(cracks,bins=50)
# # axs[0,1].set_title('Cracks: '+str(crac))
# # axs[1,1].hist(potholes[potholes > 0.0],bins=50)
# # axs[1,1].set_title('Potholes: '+str(pots))
# # fig.suptitle('Total nr of segmentations: '+str(len(route_details)))
# # plt.show()




# #################################################### SEGMENTATION PLOTS ############################################################################
# synth_acc = synthetic_data()
# routes = []
# for i in range(len(synth_acc)): 
#     routes.append(synth_acc[i].axes[0].name)

# synth_segments, aran_segments, route_details = synthetic_segmentation(synth_acc,routes,segment_size=5,overlap=0)


# hdf5file = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
# aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
# aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
# aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
# aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

# DI, allig, cracks, potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)

# seg_len = 1 
# seg_cap = 0
# segm_nr = 0
# DI = []
# alligator = []
# cracks = []
# potholes = []
# for i in tqdm(range(int(np.shape(aran_segments)[0]/seg_len))):
#     aran_details = aran_segments.iloc[i*seg_len:i*seg_len+seg_cap+1]
#     aran_alligator = aran_details[aran_alligator.columns]
#     aran_cracks = aran_details[aran_cracks.columns]
#     aran_potholes = aran_details[aran_potholes.columns]
#     temp_DI, temp_alligator, temp_cracks, temp_potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
#     DI.append(np.max(temp_DI))
#     alligator.append(np.max(temp_alligator))
#     cracks.append(np.max(temp_cracks))
#     potholes.append(np.max(temp_potholes))


# idx_max_DI = np.argmax(DI)
# np.argpartition(DI,-4)[-4:]
# idx_max_DI = 12678


# route = route_details[int(idx_max_DI*(seg_len/5))][:7]
# p79_gps = pd.read_csv("p79/"+route+".csv")
# synth = synth_acc[routes.index(route_details[int(idx_max_DI*(seg_len/5))])]
# synth = synth[synth['synth_acc'].notna()]
# synth = synth[synth['gm_speed'] >= 20]
# synth = synth.reset_index(drop=True)
# aran_start = [aran_segments['LatitudeFrom'][ idx_max_DI*seg_len-20],aran_segments['LongitudeFrom'][ idx_max_DI*seg_len-20]]
# aran_end = [aran_segments['LatitudeTo'][ idx_max_DI*seg_len+19],aran_segments['LongitudeTo'][ idx_max_DI*seg_len+19]]
# p79_start_idx, _ = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
# p79_end_idx, _ = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
# p79_details_max_DI = p79_gps[p79_start_idx:p79_end_idx+1]
# start_dist = p79_details_max_DI['Distance'].iloc[0]
# end_dist = p79_details_max_DI['Distance'].iloc[-1]
# acc_synth_max_DI = synth[ (synth['Distance']<=end_dist) & (synth['Distance']>=start_dist)]


# chain = aran_segments['EndChainage'].iloc[idx_max_DI*seg_len+19] - aran_segments['BeginChainage'].iloc[idx_max_DI*seg_len-20]
# chain

# laser = np.concatenate([p79_details_max_DI[['Laser5','Laser21']].values])
# # laser_max = np.max(laser)
# # laser_min = np.min(laser)
# laser_max = 55.5
# laser_min = -162.6

# acc = np.concatenate([acc_synth_max_DI['synth_acc']])
# acc_max = np.max(acc)
# acc_min = np.min(acc)


# x_len = np.arange(0,int(abs(chain)+10),10)
# DI_len = DI[idx_max_DI*seg_len-20:idx_max_DI*seg_len+19+1+1]
# alli_len = alligator[idx_max_DI*seg_len-20:idx_max_DI*seg_len+19+1+1]
# cracks_len = cracks[idx_max_DI*seg_len-20:idx_max_DI*seg_len+19+1+1]
# potholes_len = potholes[idx_max_DI*seg_len-20:idx_max_DI*seg_len+19+1+1]


# fig, axs = plt.subplots(6,1)
# dist = np.concatenate([p79_details_max_DI['Distance'],acc_synth_max_DI['Distance']])
# axs[0].plot(p79_details_max_DI['Distance'],p79_details_max_DI['Laser5'],label='Laser 5')
# axs[0].plot(p79_details_max_DI['Distance'],p79_details_max_DI['Laser21'],label='Laser 21')
# axs[0].set(xlim=(np.min(dist),np.max(dist)),ylim=(laser_min,laser_max))
# axs[0].legend()
# axs[0].set_title('High DI: '+str(DI[idx_max_DI]))
# axs[1].plot(acc_synth_max_DI['Distance'],acc_synth_max_DI['synth_acc'])
# axs[1].set(xlim=(np.min(dist),np.max(dist)),ylim=(acc_min,acc_max))
# axs[2].step(x_len,DI_len,label='DI')
# axs[2].step(x_len,alli_len,label='Alligator')
# axs[2].step(x_len,cracks_len,label='Cracks')
# axs[2].step(x_len,potholes_len,label='Potholes')
# axs[2].set(xlim=(0,int(abs(chain))),ylim=(0,7))
# axs[2].legend()

# axs[0].set(ylabel='Laser distance [mm]')
# axs[1].set(ylabel='Synthetic z acceleration')
# axs[2].set(ylabel='Damage index')
# for ax in axs.flat:
#     ax.set(xlabel='Distance [m]')
# # plt.show()

# # ################################################# MINIMUM ##############################################


# idx_max_DI = np.argmin(DI)
# np.argpartition(DI,4)[:4]
# idx_max_DI = 26879


# route = route_details[int(idx_max_DI*(seg_len/5))][:7]
# p79_gps = pd.read_csv("p79/"+route+".csv")
# synth = synth_acc[routes.index(route_details[int(idx_max_DI*(seg_len/5))])]
# synth = synth[synth['synth_acc'].notna()]
# synth = synth[synth['gm_speed'] >= 20]
# synth = synth.reset_index(drop=True)
# aran_start = [aran_segments['LatitudeFrom'][ idx_max_DI*seg_len-20],aran_segments['LongitudeFrom'][ idx_max_DI*seg_len-20]]
# aran_end = [aran_segments['LatitudeTo'][ idx_max_DI*seg_len+19],aran_segments['LongitudeTo'][ idx_max_DI*seg_len+19]]
# p79_start_idx, _ = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
# p79_end_idx, _ = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
# p79_details_max_DI = p79_gps[p79_start_idx:p79_end_idx+1]
# start_dist = p79_details_max_DI['Distance'].iloc[0]
# end_dist = p79_details_max_DI['Distance'].iloc[-1]
# acc_synth_max_DI = synth[ (synth['Distance']<=end_dist) & (synth['Distance']>=start_dist)]


# chain = aran_segments['EndChainage'].iloc[idx_max_DI*seg_len+19] - aran_segments['BeginChainage'].iloc[idx_max_DI*seg_len-20]
# chain

# laser = np.concatenate([p79_details_max_DI[['Laser5','Laser21']].values])
# laser_max = np.max(laser)
# laser_min = np.min(laser)

# # acc = np.concatenate([acc_synth_max_DI['synth_acc']])
# # acc_max = np.max(acc)
# # acc_min = np.min(acc)


# x_len = np.arange(0,int(abs(chain)+10),10)
# DI_len = DI[idx_max_DI*seg_len-20:idx_max_DI*seg_len+19+1+1]
# alli_len = alligator[idx_max_DI*seg_len-20:idx_max_DI*seg_len+19+1+1]
# cracks_len = cracks[idx_max_DI*seg_len-20:idx_max_DI*seg_len+19+1+1]
# potholes_len = potholes[idx_max_DI*seg_len-20:idx_max_DI*seg_len+19+1+1]


# # fig, axs = plt.subplots(3,1)
# dist = np.concatenate([p79_details_max_DI['Distance'],acc_synth_max_DI['Distance']])
# axs[3].plot(p79_details_max_DI['Distance'],p79_details_max_DI['Laser5'],label='Laser 5')
# axs[3].plot(p79_details_max_DI['Distance'],p79_details_max_DI['Laser21'],label='Laser 21')
# axs[3].set(xlim=(np.min(dist),np.max(dist)),ylim=(laser_min,laser_max))
# axs[3].legend()
# axs[3].set_title('High DI: '+str(DI[idx_max_DI]))
# axs[4].plot(acc_synth_max_DI['Distance'],acc_synth_max_DI['synth_acc'])
# axs[4].set(xlim=(np.min(dist),np.max(dist)),ylim=(acc_min,acc_max))
# axs[5].step(x_len,DI_len,label='DI')
# axs[5].step(x_len,alli_len,label='Alligator')
# axs[5].step(x_len,cracks_len,label='Cracks')
# axs[5].step(x_len,potholes_len,label='Potholes')
# axs[5].set(xlim=(0,int(abs(chain))),ylim=(0,7))
# axs[5].legend()

# axs[3].set(ylabel='Laser distance [mm]')
# axs[4].set(ylabel='Synthetic z acceleration')
# axs[5].set(ylabel='Damage index')
# for ax in axs.flat:
#     ax.set(xlabel='Distance [m]')
# plt.show()



################################################## Haversine scaled plot #################################################


# lat_len = 111332.67
# lon_len = 63195.85
# fig,ax = plt.subplots(figsize=(10,10))

# synth_acc = synthetic_data()
# routes = []
# for i in range(len(synth_acc)): 
#     routes.append(synth_acc[i].axes[0].name)

# files = glob.glob("p79/*.csv")

# df_cph1_hh = pd.read_csv(files[0])
# df_cph1_hh.drop(df_cph1_hh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
# df_cph1_vh = pd.read_csv(files[1])
# df_cph1_vh.drop(df_cph1_vh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
# df_cph6_hh = pd.read_csv(files[2])
# df_cph6_hh.drop(df_cph6_hh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
# df_cph6_vh = pd.read_csv(files[3])
# df_cph6_vh.drop(df_cph6_vh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)

# hdf5file_cph1_hh = h5py.File('aligned_data/CPH1_HH.hdf5', 'r')
# hdf5file_cph1_vh = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
# hdf5file_cph6_hh = h5py.File('aligned_data/CPH6_HH.hdf5', 'r')
# hdf5file_cph6_vh = h5py.File('aligned_data/CPH6_VH.hdf5', 'r')    


# iter = 0
# segments = {}
# for j in tqdm(range(1)):
#     synth = synth_acc[j]
#     synth = synth[synth['synth_acc'].notna()]
#     synth = synth[synth['gm_speed'] >= 20]
#     synth = synth.reset_index(drop=True)
#     route = routes[j][:7]
#     if route == 'CPH1_HH':
#         p79_gps = df_cph1_hh
#         aran_location = pd.DataFrame(hdf5file_cph1_hh['aran/trip_1/pass_1']['Location'], columns = hdf5file_cph1_hh['aran/trip_1/pass_1']['Location'].attrs['chNames'])
#     elif route == 'CPH1_VH':
#         p79_gps = df_cph1_vh
#         aran_location = pd.DataFrame(hdf5file_cph1_vh['aran/trip_1/pass_1']['Location'], columns = hdf5file_cph1_vh['aran/trip_1/pass_1']['Location'].attrs['chNames'])
#     elif route == 'CPH6_HH':
#         p79_gps = df_cph6_hh
#         aran_location = pd.DataFrame(hdf5file_cph6_hh['aran/trip_1/pass_1']['Location'], columns = hdf5file_cph6_hh['aran/trip_1/pass_1']['Location'].attrs['chNames'])
#     elif route == 'CPH6_VH':
#         p79_gps = df_cph6_vh
#         aran_location = pd.DataFrame(hdf5file_cph6_vh['aran/trip_1/pass_1']['Location'], columns = hdf5file_cph6_vh['aran/trip_1/pass_1']['Location'].attrs['chNames'])


#     p79_start = p79_gps[['Latitude','Longitude']].iloc[0].values
#     # _, aran_start_idx, _ = find_min_gps(p79_start[0], p79_start[1], aran_location['LatitudeFrom'].iloc[:100].values, aran_location['LongitudeFrom'].iloc[:100].values)
#     aran_start_idx, _ = find_min_gps_vector(p79_start,aran_location[['LatitudeFrom','LongitudeFrom']].iloc[:100].values)
#     aran_location = aran_location.iloc[aran_start_idx:].reset_index(drop=True)

#     i = 0
#     # Get 50m from ARAN -> Find gps signal from p79 -> Get measurements from synthetic data
#     while (i < (len(aran_location['LatitudeFrom'])-6) ):
#         aran_start = [aran_location['LatitudeFrom'][i],aran_location['LongitudeFrom'][i]]
#         aran_end = [aran_location['LatitudeTo'][i+4],aran_location['LongitudeTo'][i+4]]

#         p79_start_idx, start_dist = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
#         p79_end_idx, end_dist = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)

#         if start_dist < 15 and end_dist < 15:
#             dfdf = p79_gps['Distance'][p79_start_idx:p79_end_idx]
#             dfdf = dfdf.reset_index(drop=True)   

#             synth_seg = synth[((synth['Distance'] >= np.min(dfdf)) & (synth['Distance'] <= np.max(dfdf)))]
#             synth_seg = synth_seg.reset_index(drop=True)

#             stat1 = synth_seg['Distance'].empty
#             lag = []
#             for h in range(len(synth_seg)-1):
#                 lag.append(synth_seg['Distance'][h+1]-synth_seg['Distance'][h])        
#             large = [y for y in lag if y > 5]
        
#             if stat1:
#                 stat2 = True
#                 stat3 = True
#                 stat4 = True
#             else:
#                 stat2 = (synth_seg['Distance'][len(synth_seg['Distance'])-1]-synth_seg['Distance'][0]) <= 40
#                 stat3 = (len(synth_seg['synth_acc'])) > 5000
#                 stat4 = False if bool(large) == False else (np.max(large) > 5)
#             if stat1 | stat2 | stat3 | stat4:
#                 i += 1
#             else:
#                 i += 5
#                 segments[iter] = synth_seg['synth_acc']
#                 Lat_cord = [p79_gps['Latitude'][p79_start_idx]*lat_len,p79_gps['Latitude'][p79_end_idx]*lat_len]
#                 Lon_cord = [p79_gps['Longitude'][p79_start_idx]*lon_len,p79_gps['Longitude'][p79_end_idx]*lon_len]
#                 _ = plt.gca().add_patch(Rectangle((Lon_cord[0],Lat_cord[0]),Lon_cord[1]-Lon_cord[0],Lat_cord[1]-Lat_cord[0],edgecolor='green',facecolor='none',lw=1))
#                 iter += 1

#                 x_val = [aran_start[1]*lon_len,p79_gps['Longitude'][p79_start_idx]*lon_len]
#                 y_val = [aran_start[0]*lat_len,p79_gps['Latitude'][p79_start_idx]*lat_len]
#                 _ = plt.plot(x_val,y_val,'r')

#                 x_val = [aran_end[1]*lon_len,p79_gps['Longitude'][p79_end_idx]*lon_len]
#                 y_val = [aran_end[0]*lat_len,p79_gps['Latitude'][p79_end_idx]*lat_len]
#                 _ = plt.plot(x_val,y_val,'k')

#         else:
#             i +=1



# p79_gps = p79_gps[p79_gps['Longitude'] != 0]
# p79_gps = p79_gps[p79_gps['Latitude'] != 0]
# p79_gps = p79_gps.reset_index(drop=True)
# plt.scatter(x=p79_gps['Longitude']*lon_len, y=p79_gps['Latitude']*lat_len,s=1,c="blue",label='p79 route')

# plt.scatter(x=aran_location['LongitudeFrom']*lon_len, y=aran_location['LatitudeFrom']*lat_len,s=1,c="red",label='AranFrom')
# plt.scatter(x=aran_location['LongitudeTo']*lon_len, y=aran_location['LatitudeTo']*lat_len,s=1,c="black",label='AranTo')

# custom_lines = [Line2D([0], [0], color='blue', lw=2),
#         Line2D([0], [0], color='red', lw=2),
#         Line2D([0], [0], color='black', lw=2),
#         Line2D([0], [0], color='green', lw=2)]
# plt.legend(custom_lines,['p79 route','AranFrom','AranTo','Segment'])
# plt.title('Segmentation algorithm')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')

# i = 30
# aran_start = [aran_location['LatitudeFrom'][i],aran_location['LongitudeFrom'][i]]
# plt.axis([(aran_start[1])*lon_len-60,(aran_start[1])*lon_len+60,(aran_start[0])*lat_len-60,(aran_start[0])*lat_len+60])

# plt.show()


# ###########################################################################################################################


# idxs = np.argpartition(cracks,-20)[-20:]


# for g in range(10):
#     idx_min_DI = idxs[g]
#     route = route_details[int(idx_max_cracks*(seg_len/5))][:7]
#     p79_gps = pd.read_csv("p79/"+route+".csv")
#     synth = synth_acc[routes.index(route_details[int(idx_max_cracks*(seg_len/5))])]
#     synth = synth[synth['synth_acc'].notna()]
#     synth = synth[synth['gm_speed'] >= 20]
#     synth = synth.reset_index(drop=True)
#     aran_start = [aran_segments['LatitudeFrom'][ idx_min_DI*seg_len],aran_segments['LongitudeFrom'][ idx_min_DI*seg_len]]
#     aran_end = [aran_segments['LatitudeTo'][ idx_min_DI*seg_len+seg_cap],aran_segments['LongitudeTo'][ idx_min_DI*seg_len+seg_cap]]
#     p79_start_idx, _ = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
#     p79_end_idx, _ = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
#     p79_details_min_DI = p79_gps[p79_start_idx:p79_end_idx+1]

#     start_dist = p79_details_min_DI['Distance'].iloc[0]
#     end_dist = p79_details_min_DI['Distance'].iloc[-1]
#     acc_synth_min_DI = synth[ (synth['Distance']<=end_dist) & (synth['Distance']>=start_dist)]

#     fig, axs = plt.subplots(2,1)
#     dist = np.concatenate([p79_details_min_DI['Distance'],acc_synth_min_DI['Distance']])
#     axs[0].plot(p79_details_min_DI['Distance'],p79_details_min_DI['Laser5'],label='Laser 5')
#     axs[0].plot(p79_details_min_DI['Distance'],p79_details_min_DI['Laser21'],label='Laser 21')
#     axs[0].set(xlim=(np.min(dist),np.max(dist)),ylim=(laser_min,laser_max))
#     axs[0].legend()
#     axs[0].set_title('Low DI: '+str(idx_min_DI)+" - "+str(cracks[idx_min_DI]))
#     axs[1].plot(acc_synth_min_DI['Distance'],acc_synth_min_DI['synth_acc'])
    
#     axs[0].set(ylabel='Laser distance [mm]')
#     axs[1].set(ylabel='Synthetic z acceleration')

#     plt.show()


################################################### SPLIT HISTOGRAM ################################################################
synth_acc = synthetic_data()
routes = []
for i in range(len(synth_acc)): 
    routes.append(synth_acc[i].axes[0].name)

synth_segments, aran_segments, route_details = synthetic_segmentation(synth_acc,routes,segment_size=5,overlap=0)

hdf5file = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

seg_len = 5 
seg_cap = 4
DI = []
alligator = []
cracks = []
potholes = []
for i in tqdm(range(int(np.shape(aran_segments)[0]/seg_len))):
    aran_details = aran_segments.iloc[i*seg_len:i*seg_len+seg_cap+1]
    aran_alligator = aran_details[aran_alligator.columns]
    aran_cracks = aran_details[aran_cracks.columns]
    aran_potholes = aran_details[aran_potholes.columns]
    temp_DI, temp_alligator, temp_cracks, temp_potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
    DI.append(np.max(temp_DI))
    alligator.append(np.max(temp_alligator))
    cracks.append(np.max(temp_cracks))
    potholes.append(np.max(temp_potholes))


_, _, cv = custom_splits(aran_segments,route_details,save=True)

for train,test in cv:
    pots_train, cracs_train, allis_train, dams_train = 0, 0, 0, 0
    pots_test, cracs_test, allis_test, dams_test = 0, 0, 0, 0
    DI_train = [DI[i] for i in train]
    DI_test = [DI[i] for i in test]
    pot_train = [potholes[i] for i in train]
    pot_test = [potholes[i] for i in test]
    crac_train = [cracks[i] for i in train]
    crac_test = [cracks[i] for i in test]
    alli_train = [alligator[i] for i in train]
    alli_test = [alligator[i] for i in test]
    
    for j in range(len(pot_train)):
        if pot_train[j] > 0.0:
            pots_train += 1
        if crac_train[j] > 0.0:
            cracs_train += 1
        if alli_train[j] > 0.0:
            allis_train += 1
        if DI_train[j] > 0.0:
            dams_train += 1

    for j in range(len(pot_test)):
        if pot_test[j] > 0.0:
            pots_test += 1
        if crac_test[j] > 0.0:
            cracs_test += 1
        if alli_test[j] > 0.0:
            allis_test += 1
        if DI_test[j] > 0.0:
            dams_test += 1

    DI_train = np.array(DI_train)
    alli_train = np.array(alli_train)
    crac_train = np.array(crac_train)
    pot_train = np.array(pot_train)

    DI_test = np.array(DI_test)
    alli_test = np.array(alli_test)
    crac_test = np.array(crac_test)
    pot_test = np.array(pot_test)


    fig, axs = plt.subplots(2,2)
    axs[0,0].hist(DI_train[DI_train > 0.0],bins=50,color="blue",label="train split")
    axs[1,0].hist(alli_train[alli_train > 0.0],bins=50,color="blue",label="train split")
    axs[0,1].hist(crac_train[crac_train > 0.0],bins=50,color="blue",label="train split")
    axs[1,1].hist(pot_train[pot_train > 0.0],bins=50,color="blue",label="train split")

    axs[0,0].hist(DI_test[DI_test > 0.0],bins=50,color="red",label="test split")
    axs[0,0].set_title('DI: '+str(dams_test)+'/'+str(dams_train))
    axs[0,0].set(ylim=(0,600))
    axs[1,0].hist(alli_test[alli_test > 0.0],bins=50,color="red",label="test split")
    axs[1,0].set_title('Alligator: '+str(allis_test)+'/'+str(allis_train))
    axs[1,0].set(ylim=(0,600))
    axs[0,1].hist(crac_test[crac_test > 0.0],bins=50,color="red",label="test split")
    axs[0,1].set_title('Cracks: '+str(cracs_test)+'/'+str(cracs_train))
    axs[0,1].set(ylim=(0,600))
    axs[1,1].hist(pot_test[pot_test > 0.0],bins=50,color="red",label="test split")
    axs[1,1].set_title('Potholes: '+str(pots_test)+'/'+str(pots_train))
    axs[1,1].set(ylim=(0,50))
    axs[0,0].legend()
    axs[1,0].legend()
    axs[0,1].legend()
    axs[1,1].legend()
    fig.suptitle('nr of segments: '+str(len(test))+'/'+str(len(train))+' - test/train')
    plt.show()

##############################################################################################################################

################################################# FEATURE IMPORTANCE #########################################################
# plt.text(bis_features[0],feature_importance[bis_features[0]],top_10[0][2:])
# plt.text(bis_features[1],feature_importance[bis_features[1]],top_10[1][2:])
# plt.text(bis_features[2],feature_importance[bis_features[2]],top_10[2][2:])
# plt.text(bis_features[3],feature_importance[bis_features[3]],top_10[3][2:])
# plt.text(bis_features[4],feature_importance[bis_features[4]],top_10[4][2:])
# plt.text(bis_features[5],feature_importance[bis_features[5]],top_10[5][2:])
# plt.text(bis_features[6],feature_importance[bis_features[6]],top_10[6][2:])
# plt.text(bis_features[7],feature_importance[bis_features[7]],top_10[7][2:])
# plt.text(bis_features[8],feature_importance[bis_features[8]],top_10[8][2:])
# plt.text(bis_features[9],feature_importance[bis_features[9]],top_10[9][2:])
synth_acc = synthetic_data()
routes = []
for i in range(len(synth_acc)): 
    routes.append(synth_acc[i].axes[0].name)

synth_segments, aran_segments, route_details = synthetic_segmentation(synth_acc,routes,segment_size=5,overlap=0)


hdf5file = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

DI, allig, cracks, potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)


data = synth_segments#.iloc[:,0:100]
features,feature_names = feature_extraction(data)

seg_len = 5 
seg_cap = 4
DI = []
alligator = []
cracks = []
potholes = []
for i in tqdm(range(int(np.shape(aran_segments)[0]/seg_len))):
    aran_details = aran_segments.iloc[i*seg_len:i*seg_len+seg_cap+1]
    aran_alligator = aran_details[aran_alligator.columns]
    aran_cracks = aran_details[aran_cracks.columns]
    aran_potholes = aran_details[aran_potholes.columns]
    temp_DI, temp_alligator, temp_cracks, temp_potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
    DI.append(np.max(temp_DI))
    alligator.append(np.max(temp_alligator))
    cracks.append(np.max(temp_cracks))
    potholes.append(np.max(temp_potholes))

gridsearch = 1
verbose = 0
n_jobs = 4
model = 1

cv, test_split, _ = custom_splits(aran_segments,route_details)

scores_RandomForest_DI        = method_RandomForest(features, DI, 'DI', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
scores_RandomForest_potholes  = method_RandomForest(features, potholes, 'potholes', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
scores_RandomForest_cracks    = method_RandomForest(features, cracks, 'cracks', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
scores_RandomForest_alligator = method_RandomForest(features, alligator, 'alligator', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
obj = scores_RandomForest_DI['Gridsearchcv_obj']
feature_importance = obj.best_estimator_.feature_importances_
bis_features = np.argpartition(feature_importance,-10)[-10:]

top_10 = feature_names.index[bis_features]

plt.plot(feature_importance,label='DI')

obj = scores_RandomForest_potholes['Gridsearchcv_obj']
feature_importance = obj.best_estimator_.feature_importances_
bis_features = np.argpartition(feature_importance,-10)[-10:]
top_10 = feature_names.index[bis_features]
plt.plot(feature_importance,label='Potholes')

obj = scores_RandomForest_cracks['Gridsearchcv_obj']
feature_importance = obj.best_estimator_.feature_importances_
bis_features = np.argpartition(feature_importance,-10)[-10:]
top_10 = feature_names.index[bis_features]
plt.plot(feature_importance,label='Cracks')

obj = scores_RandomForest_alligator['Gridsearchcv_obj']
feature_importance = obj.best_estimator_.feature_importances_
bis_features = np.argpartition(feature_importance,-10)[-10:]
top_10 = feature_names.index[bis_features]
plt.plot(feature_importance,label='Alligator')
plt.axvspan(0, 336, facecolor='green', alpha=0.3)
plt.axvspan(336, 336+36, facecolor='yellow', alpha=0.3)
plt.axvspan(336+36, 336+36+18, facecolor='red', alpha=0.3)
plt.axis([0,390,0,0.016])
plt.legend()
plt.title('Spectral (green) - Statistical (yellow) - Temporal (red)')
plt.show()




########################################## SPEED HISTOGRAM GREEN MOBILITY GM #################################

# counter = len(glob.glob1("p79/","*.csv"))
# files = glob.glob("p79/*.csv")
# k=0
# spd_list = []
# for i in range(counter):
#     #p79
#     p79_file = files[i]

#     df_p79 = pd.read_csv(p79_file)
#     df_p79.drop(df_p79.columns.difference(['Distance','Laser5','Laser21','Latitude','Longitude']),axis=1,inplace=True)

#     #Green Mobility
#     file = files[i][4:11]
#     gm_path = 'aligned_data/'+file+'.hdf5'
#     hdf5file = h5py.File(gm_path, 'r')
#     passage = hdf5file.attrs['GM_full_passes']
        
#     new_passage = []
#     for j in range(len(passage)):
#         passagefile = hdf5file[passage[j]]
#         if "obd.spd_veh" in passagefile.keys(): # some passes only contain gps and gps_match
#             new_passage.append(passage[j])
        
#     passage = np.array(new_passage,dtype=object)
#     for j in range(len(passage)):
#         passagefile = hdf5file[passage[j]]
#         gm_speed = pd.DataFrame(passagefile['obd.spd_veh'], columns = passagefile['obd.spd_veh'].attrs['chNames'])
#         print("Generating Synthetic Profile for trip:",i+1,"/",counter,"- passage:",j+1,"/",len(passage))
#         print(len(gm_speed['spd_veh']))
#         spd_list.extend(gm_speed['spd_veh'].tolist())

# spd_list = np.array(spd_list)
# spd_list = spd_list[spd_list > 20]
# plt.hist(spd_list,bins=100)
# plt.show()
# np.median(spd_list)
# np.mean(spd_list)

#################################################################################################################

##################################################### GM ALIGNED ################################################

lat_len = 1# 111332.67
lon_len = 1#63195.85

synth_acc = synthetic_data()
routes = []
for i in range(len(synth_acc)): 
    routes.append(synth_acc[i].axes[0].name)

GM_segments, aran_segments, route_details = GM_segmentation(segment_size=5,overlap=0)

cut = [10000,19000,14000]
cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split1')

fig,ax = plt.subplots(1,2)
ax[0].scatter(x=aran_segments.loc[splits['1']]['LongitudeFrom'],y=aran_segments.loc[splits['1']]['LatitudeFrom'],s=1,label=len(splits['1']))
ax[0].scatter(x=aran_segments.loc[splits['2']]['LongitudeFrom'],y=aran_segments.loc[splits['2']]['LatitudeFrom'],s=1,label=len(splits['2']))
ax[0].scatter(x=aran_segments.loc[splits['3']]['LongitudeFrom'],y=aran_segments.loc[splits['3']]['LatitudeFrom'],s=1,label=len(splits['3']))
ax[1].scatter(x=aran_segments.loc[splits['4']]['LongitudeFrom'],y=aran_segments.loc[splits['4']]['LatitudeFrom'],s=1,label=len(splits['4']))
ax[1].scatter(x=aran_segments.loc[splits['5']]['LongitudeFrom'],y=aran_segments.loc[splits['5']]['LatitudeFrom'],s=1,label=len(splits['5']))
ax[0].legend()
ax[1].legend()
plt.show()


aran_test = aran_segments

fig,ax = plt.subplots(1,2)
k = 0
for train,test in cv_train:
    if k < 3:
        ax[0].scatter(x=aran_test.loc[test]['LongitudeFrom'],y=aran_test.loc[test]['LatitudeFrom'],s=1,label=len(test))
    else:
        ax[1].scatter(x=aran_test.loc[test]['LongitudeFrom'],y=aran_test.loc[test]['LatitudeFrom'],s=1,label=len(test))
    k+=1
ax[0].legend()
ax[1].legend()
plt.show()

#################################################################################################################

cv_train, split_test, X_train, X_test, splits = route_splits(features,route_details,'cph1_vh')

fig,ax = plt.subplots(1,2)
ax[0].scatter(x=aran_segments.loc[splits['cph1_vh']]['LongitudeFrom'],y=aran_segments.loc[splits['cph1_vh']]['LatitudeFrom'],s=1,label=len(splits['cph1_vh']))
ax[0].scatter(x=aran_segments.loc[splits['cph1_hh']]['LongitudeFrom'],y=aran_segments.loc[splits['cph1_hh']]['LatitudeFrom'],s=1,label=len(splits['cph1_hh']))
ax[1].scatter(x=aran_segments.loc[splits['cph6_vh']]['LongitudeFrom'],y=aran_segments.loc[splits['cph6_vh']]['LatitudeFrom'],s=1,label=len(splits['cph6_vh']))
ax[1].scatter(x=aran_segments.loc[splits['cph6_hh']]['LongitudeFrom'],y=aran_segments.loc[splits['cph6_hh']]['LatitudeFrom'],s=1,label=len(splits['cph6_hh']))
ax[0].legend()
ax[1].legend()
plt.show()



############################### init training ####################3

id = 'GoogleNet_t7.csv'

loss = pd.read_csv('training/loss_save_'+id,sep=',',header=None)
loss = loss.values.reshape((np.shape(loss)[1],-1))
R2_train = pd.read_csv('training/R2_train_'+id,sep=',',header=None)
R2_train = R2_train.values.reshape((np.shape(R2_train)[1],-1))
R2_val = pd.read_csv('training/R2_val_'+id,sep=',',header=None)
R2_val = R2_val.values.reshape((np.shape(R2_val)[1],-1))

points = np.linspace(10,len(loss),len(R2_train))
r2_max = np.max(R2_train) if np.max(R2_train) > np.max(R2_val) else np.max(R2_val)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.plot(loss,c='blue',label='Training loss')
lns2 = ax2.plot(points,R2_train,c='red',label='Training R2')
lns3 = ax2.plot(points,R2_val,c='green',label='Validation R2')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax2.set_ylabel('R2')
ax2.set_ylim([0,r2_max+0.1])
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=9)
ax1.set_title('R2_val = '+str(round(np.max(R2_val),3)) + ' - ' + 'R2_test = '+str('x'))
plt.show()


from DL_functions import *
batch_size = 32
nr_tar=1
path = 'DL_synth_data'
labelsFile = 'DL_synth_data/labelsfile'
sourceTransform = Compose([ToTensor()]) #, Resize((224,224))
test = CustomDataset(labelsFile+"_test.csv", path+'/test/', sourceTransform, nr_tar)
test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
print(len(test_dl.dataset))

test_features, test_labels = next(iter(test_dl))
print(f"Feature batch shape: {test_features.size()}")
print(f"Labels batch shape: {test_labels.size()}")

img = test_features[4] #.squeeze()
img1 = img.permute(1,2,0)
label = test_labels[4]
print(label)


sourceTransform = Compose([ToTensor(), Resize((224,224))]) #, Resize((224,224))
test = CustomDataset(labelsFile+"_test.csv", path+'/test/', sourceTransform, nr_tar)
test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
test_features, test_labels = next(iter(test_dl))
img = test_features[4] #.squeeze()
img2 = img.permute(1,2,0)
fig,axs = plt.subplots(2)
axs[0].imshow(img1)
axs[1].imshow(img2)
plt.show()