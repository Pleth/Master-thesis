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

synth_acc = synthetic_data()
routes = []
for i in range(len(synth_acc)): 
    routes.append(synth_acc[i].axes[0].name)

synth_segments, aran_segments, route_details = synthetic_segmentation(synth_acc,routes,segment_size=5,overlap=0)

cph1_hh = []
cph1_vh = []
cph6_hh = []
cph6_vh = []
for i in range(len(route_details)):
    if route_details[i][:7] == 'CPH1_HH':
        cph1_hh.extend([i*5,i*5+1,i*5+2,i*5+3,i*5+4])
    if route_details[i][:7] == 'CPH1_VH':
        cph1_vh.extend([i*5,i*5+1,i*5+2,i*5+3,i*5+4])
    if route_details[i][:7] == 'CPH6_HH':
        cph6_hh.extend([i*5,i*5+1,i*5+2,i*5+3,i*5+4])
    if route_details[i][:7] == 'CPH6_VH':
        cph6_vh.extend([i*5,i*5+1,i*5+2,i*5+3,i*5+4])

fig,ax = plt.subplots(1,2)
ax[0].scatter(x=aran_segments['LongitudeFrom'][cph1_hh],y=aran_segments['LatitudeFrom'][cph1_hh],s=1,c="blue",label='CPH1_HH')
ax[0].scatter(x=aran_segments['LongitudeFrom'][cph1_vh],y=aran_segments['LatitudeFrom'][cph1_vh],s=1,c="red",label='CPH1_VH')
ax[1].scatter(x=aran_segments['LongitudeFrom'][cph6_hh],y=aran_segments['LatitudeFrom'][cph6_hh],s=1,c="blue",label='CPH6_HH')
ax[1].scatter(x=aran_segments['LongitudeFrom'][cph6_vh],y=aran_segments['LatitudeFrom'][cph6_vh],s=1,c="red",label='CPH6_VH')
ax[0].legend()
ax[1].legend()
plt.show()

total = round(np.sum(abs(aran_segments['EndChainage']-aran_segments['BeginChainage'])))

cph1_hh_tot = round(np.sum(abs(aran_segments['EndChainage'][cph1_hh]-aran_segments['BeginChainage'][cph1_hh])))
cph1_vh_tot = round(np.sum(abs(aran_segments['EndChainage'][cph1_vh]-aran_segments['BeginChainage'][cph1_vh])))
cph6_hh_tot = round(np.sum(abs(aran_segments['EndChainage'][cph6_hh]-aran_segments['BeginChainage'][cph6_hh])))
cph6_vh_tot = round(np.sum(abs(aran_segments['EndChainage'][cph6_vh]-aran_segments['BeginChainage'][cph6_vh])))

cph1_tot = cph1_hh_tot + cph1_vh_tot
cph6_tot = cph6_hh_tot + cph6_vh_tot

total == cph1_hh_tot + cph1_vh_tot + cph6_hh_tot + cph6_vh_tot
total == cph1_tot + cph6_tot


#### CPH1

fig,ax = plt.subplots(1,2)

cph1_len = (len(cph1_hh) + len(cph1_vh))/5

counter = 0
chain_start = 645
for i in tqdm(range(int(cph1_len))):
    counter1 = np.sum(np.sum(aran_segments['EndChainage'].iloc[cph1_hh] < chain_start))
    counter2 = np.sum(np.sum(aran_segments['BeginChainage'].iloc[cph1_vh] < chain_start))
    counter = (counter1 + counter2)/5
    if (counter >= cph1_len/3):
        break
    chain_start += 10

dd = aran_segments['EndChainage'][cph1_hh] < chain_start
temp_cph1_hh = []
for i in range(len(cph1_hh)):
    if dd.iloc[i] == True:
        temp_cph1_hh.append(cph1_hh[i])

dd = aran_segments['BeginChainage'][cph1_vh] < chain_start
temp_cph1_vh = []
for i in range(len(cph1_vh)):
    if dd.iloc[i] == True:
        temp_cph1_vh.append(cph1_vh[i])

ax[0].scatter(x=aran_segments['LongitudeFrom'][temp_cph1_hh],y=aran_segments['LatitudeFrom'][temp_cph1_hh],s=1,c="blue",label='CPH1_HH')
ax[0].scatter(x=aran_segments['LongitudeFrom'][temp_cph1_vh],y=aran_segments['LatitudeFrom'][temp_cph1_vh],s=1,c="red",label='CPH1_VH')


counter = 0
chain_end = chain_start
chain_start = chain_start + 50
for i in tqdm(range(int(cph1_len))):
    counter1 = np.sum(np.sum((chain_end < aran_segments['BeginChainage'].iloc[cph1_hh]) & (aran_segments['EndChainage'].iloc[cph1_hh] < chain_start)))
    counter2 = np.sum(np.sum((chain_end < aran_segments['BeginChainage'].iloc[cph1_vh]) & (aran_segments['EndChainage'].iloc[cph1_vh] < chain_start)))
    counter = (counter1 + counter2)/5
    if (counter >= cph1_len/3):
        break
    chain_start += 10

dd = (chain_end < aran_segments['BeginChainage'][cph1_hh]) & (aran_segments['EndChainage'][cph1_hh] < chain_start)
temp_cph1_hh = []
for i in range(len(cph1_hh)):
    if dd.iloc[i] == True:
        temp_cph1_hh.append(cph1_hh[i])

dd = (chain_end < aran_segments['BeginChainage'][cph1_vh]) & (aran_segments['BeginChainage'][cph1_vh] < chain_start)
temp_cph1_vh = []
for i in range(len(cph1_vh)):
    if dd.iloc[i] == True:
        temp_cph1_vh.append(cph1_vh[i])

ax[0].scatter(x=aran_segments['LongitudeFrom'][temp_cph1_hh],y=aran_segments['LatitudeFrom'][temp_cph1_hh],s=1,c="green",label='CPH1_HH')
ax[0].scatter(x=aran_segments['LongitudeFrom'][temp_cph1_vh],y=aran_segments['LatitudeFrom'][temp_cph1_vh],s=1,c="black",label='CPH1_VH')


counter = 0
chain_end = chain_start
chain_start = chain_start + 50
for i in tqdm(range(int(cph1_len))):
    counter1 = np.sum(np.sum(chain_end < aran_segments['BeginChainage'].iloc[cph1_hh]))
    counter2 = np.sum(np.sum(chain_end < aran_segments['EndChainage'].iloc[cph1_vh]))
    counter = (counter1 + counter2)/5
    if (counter >= cph1_len/3):
        break
    chain_start += 10

dd = chain_end < aran_segments['BeginChainage'][cph1_hh]
temp_cph1_hh = []
for i in range(len(cph1_hh)):
    if dd.iloc[i] == True:
        temp_cph1_hh.append(cph1_hh[i])

dd = chain_end < aran_segments['EndChainage'][cph1_vh]
temp_cph1_vh = []
for i in range(len(cph1_vh)):
    if dd.iloc[i] == True:
        temp_cph1_vh.append(cph1_vh[i])

ax[0].scatter(x=aran_segments['LongitudeFrom'][temp_cph1_hh],y=aran_segments['LatitudeFrom'][temp_cph1_hh],s=1,c="yellow",label='CPH1_HH')
ax[0].scatter(x=aran_segments['LongitudeFrom'][temp_cph1_vh],y=aran_segments['LatitudeFrom'][temp_cph1_vh],s=1,c="magenta",label='CPH1_VH')


### CPH6


cph6_len = (len(cph6_hh) + len(cph6_vh))/5

counter = 0
chain_start = 0
for i in tqdm(range(int(cph6_len))):
    counter1 = np.sum(np.sum(aran_segments['EndChainage'].iloc[cph6_hh] < chain_start))
    counter2 = np.sum(np.sum(aran_segments['BeginChainage'].iloc[cph6_vh] < chain_start))
    counter = (counter1 + counter2)/5
    if (counter >= cph6_len/2):
        break
    chain_start += 10

dd = aran_segments['EndChainage'][cph6_hh] < chain_start
temp_cph6_hh = []
for i in range(len(cph6_hh)):
    if dd.iloc[i] == True:
        temp_cph6_hh.append(cph6_hh[i])

dd = aran_segments['BeginChainage'][cph6_vh] < chain_start
temp_cph6_vh = []
for i in range(len(cph6_vh)):
    if dd.iloc[i] == True:
        temp_cph6_vh.append(cph6_vh[i])

ax[1].scatter(x=aran_segments['LongitudeFrom'][temp_cph6_hh],y=aran_segments['LatitudeFrom'][temp_cph6_hh],s=1,c="blue",label='CPH1_HH')
ax[1].scatter(x=aran_segments['LongitudeFrom'][temp_cph6_vh],y=aran_segments['LatitudeFrom'][temp_cph6_vh],s=1,c="red",label='CPH1_VH')

counter = 0
chain_end = chain_start
for i in tqdm(range(int(cph6_len))):
    counter1 = np.sum(np.sum(chain_end < aran_segments['BeginChainage'].iloc[cph6_hh]))
    counter2 = np.sum(np.sum(chain_end < aran_segments['EndChainage'].iloc[cph6_vh]))
    counter = (counter1 + counter2)/5
    if (counter >= cph6_len/2):
        break
    chain_start += 10

dd = chain_end < aran_segments['BeginChainage'][cph6_hh]
temp_cph6_hh = []
for i in range(len(cph6_hh)):
    if dd.iloc[i] == True:
        temp_cph6_hh.append(cph6_hh[i])

dd = chain_end < aran_segments['EndChainage'][cph6_vh]
temp_cph6_vh = []
for i in range(len(cph6_vh)):
    if dd.iloc[i] == True:
        temp_cph6_vh.append(cph6_vh[i])


ax[1].scatter(x=aran_segments['LongitudeFrom'][temp_cph6_hh],y=aran_segments['LatitudeFrom'][temp_cph6_hh],s=1,c="green",label='CPH1_HH')
ax[1].scatter(x=aran_segments['LongitudeFrom'][temp_cph6_vh],y=aran_segments['LatitudeFrom'][temp_cph6_vh],s=1,c="black",label='CPH1_VH')
ax[0].legend()
ax[1].legend()
plt.show()





####################################################################################################################################################

#################################################### SEGMENTATION PLOTS ############################################################################
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
# idx_min_DI = np.argmin(DI)
# # idx_min_DI = 3541

# cr = np.array(cracks)
# cr = cr[(np.array(alligator) < 0.1) & (np.array(potholes) < 0.1)]
# idx_max_cracks = cracks.index(np.max(cr))
# idx_max_cracks = 24474-4

# all = np.array(alligator)
# all = all[(np.array(cracks) < 0.1) & (np.array(potholes) < 0.1)]
# idx_max_alligator = alligator.index(np.max(all))
# idx_max_alligator = 30774

# pot = np.array(potholes)
# # pot = pot[(np.array(alligator) < 2) & (np.array(cracks) < 1.5)]
# idx_max_potholes = potholes.index(np.max(pot)) # 454
# # idx_max_potholes = int(33590/seg_len)

# pots, crac, alli = 0, 0, 0
# for j in range(len(potholes)):
#     if potholes[j] > 0.0:
#         pots += 1
#     if cracks[j] > 0.0:
#         crac += 1
#     if alligator[j] > 0.0:
#         alli += 1


# route = route_details[int(idx_max_potholes*(seg_len/5))][:7]
# p79_gps = pd.read_csv("p79/"+route+".csv")
# synth = synth_acc[routes.index(route_details[int(idx_max_potholes*(seg_len/5))])]
# synth = synth[synth['synth_acc'].notna()]
# synth = synth[synth['gm_speed'] >= 20]
# synth = synth.reset_index(drop=True)
# aran_start = [aran_segments['LatitudeFrom'][idx_max_potholes*seg_len],aran_segments['LongitudeFrom'][ idx_max_potholes*seg_len]]
# aran_end = [aran_segments['LatitudeTo'][ idx_max_potholes*seg_len+seg_cap],aran_segments['LongitudeTo'][ idx_max_potholes*seg_len+seg_cap]]
# p79_start_idx, _ = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
# p79_end_idx, _ = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
# p79_details_max_potholes = p79_gps[p79_start_idx:p79_end_idx+1]
# start_dist = p79_details_max_potholes['Distance'].iloc[0]
# end_dist = p79_details_max_potholes['Distance'].iloc[-1]
# acc_synth_max_potholes = synth[ (synth['Distance']<=end_dist) & (synth['Distance']>=start_dist)]



# route = route_details[int(idx_max_cracks*(seg_len/5))][:7]
# p79_gps = pd.read_csv("p79/"+route+".csv")
# synth = synth_acc[routes.index(route_details[int(idx_max_cracks*(seg_len/5))])]
# synth = synth[synth['synth_acc'].notna()]
# synth = synth[synth['gm_speed'] >= 20]
# synth = synth.reset_index(drop=True)
# aran_start = [aran_segments['LatitudeFrom'][ idx_max_cracks*seg_len],aran_segments['LongitudeFrom'][ idx_max_cracks*seg_len]]
# aran_end = [aran_segments['LatitudeTo'][ idx_max_cracks*seg_len+seg_cap],aran_segments['LongitudeTo'][ idx_max_cracks*seg_len+seg_cap]]
# p79_start_idx, _ = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
# p79_end_idx, _ = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
# p79_details_max_cracks = p79_gps[p79_start_idx:p79_end_idx+1]
# start_dist = p79_details_max_cracks['Distance'].iloc[0]
# end_dist = p79_details_max_cracks['Distance'].iloc[-1]
# acc_synth_max_cracks = synth[ (synth['Distance']<=end_dist) & (synth['Distance']>=start_dist)]



# route = route_details[int(idx_max_alligator*(seg_len/5))][:7]
# p79_gps = pd.read_csv("p79/"+route+".csv")
# synth = synth_acc[routes.index(route_details[int(idx_max_alligator*(seg_len/5))])]
# synth = synth[synth['synth_acc'].notna()]
# synth = synth[synth['gm_speed'] >= 20]
# synth = synth.reset_index(drop=True)
# aran_start = [aran_segments['LatitudeFrom'][ idx_max_alligator*seg_len],aran_segments['LongitudeFrom'][ idx_max_alligator*seg_len]]
# aran_end = [aran_segments['LatitudeTo'][ idx_max_alligator*seg_len+seg_cap],aran_segments['LongitudeTo'][ idx_max_alligator*seg_len+seg_cap]]
# p79_start_idx, _ = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
# p79_end_idx, _ = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
# p79_details_max_alligator = p79_gps[p79_start_idx:p79_end_idx+1]
# start_dist = p79_details_max_alligator['Distance'].iloc[0]
# end_dist = p79_details_max_alligator['Distance'].iloc[-1]
# acc_synth_max_alligator = synth[ (synth['Distance']<=end_dist) & (synth['Distance']>=start_dist)]




# route = route_details[int(idx_max_DI*(seg_len/5))][:7]
# p79_gps = pd.read_csv("p79/"+route+".csv")
# synth = synth_acc[routes.index(route_details[int(idx_max_DI*(seg_len/5))])]
# synth = synth[synth['synth_acc'].notna()]
# synth = synth[synth['gm_speed'] >= 20]
# synth = synth.reset_index(drop=True)
# aran_start = [aran_segments['LatitudeFrom'][ idx_max_DI*seg_len],aran_segments['LongitudeFrom'][ idx_max_DI*seg_len]]
# aran_end = [aran_segments['LatitudeTo'][ idx_max_DI*seg_len+seg_cap],aran_segments['LongitudeTo'][ idx_max_DI*seg_len+seg_cap]]
# p79_start_idx, _ = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
# p79_end_idx, _ = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
# p79_details_max_DI = p79_gps[p79_start_idx:p79_end_idx+1]
# start_dist = p79_details_max_DI['Distance'].iloc[0]
# end_dist = p79_details_max_DI['Distance'].iloc[-1]
# acc_synth_max_DI = synth[ (synth['Distance']<=end_dist) & (synth['Distance']>=start_dist)]


# route = route_details[int(idx_min_DI*(seg_len/5))][:7]
# p79_gps = pd.read_csv("p79/"+route+".csv")
# synth = synth_acc[routes.index(route_details[int(idx_min_DI*(seg_len/5))])]
# synth = synth[synth['synth_acc'].notna()]
# synth = synth[synth['gm_speed'] >= 20]
# synth = synth.reset_index(drop=True)
# aran_start = [aran_segments['LatitudeFrom'][ idx_min_DI*seg_len],aran_segments['LongitudeFrom'][ idx_min_DI*seg_len]]
# aran_end = [aran_segments['LatitudeTo'][ idx_min_DI*seg_len+seg_cap],aran_segments['LongitudeTo'][ idx_min_DI*seg_len+seg_cap]]
# p79_start_idx, _ = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
# p79_end_idx, _ = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
# p79_details_min_DI = p79_gps[p79_start_idx:p79_end_idx+1]
# start_dist = p79_details_min_DI['Distance'].iloc[0]
# end_dist = p79_details_min_DI['Distance'].iloc[-1]
# acc_synth_min_DI = synth[ (synth['Distance']<=end_dist) & (synth['Distance']>=start_dist)]

# laser = np.concatenate([p79_details_max_potholes[['Laser5','Laser21']].values, p79_details_max_cracks[['Laser5','Laser21']].values, p79_details_max_alligator[['Laser5','Laser21']].values, p79_details_max_DI[['Laser5','Laser21']].values, p79_details_min_DI[['Laser5','Laser21']].values])
# laser_max = np.max(laser)
# laser_min = np.min(laser)

# acc = np.concatenate([acc_synth_max_potholes['synth_acc'],acc_synth_max_cracks['synth_acc'],acc_synth_max_alligator['synth_acc'],acc_synth_max_DI['synth_acc'],acc_synth_min_DI['synth_acc']])
# acc_max = np.max(acc)
# acc_min = np.min(acc)

# fig, axs = plt.subplots(2,5)
# dist = np.concatenate([p79_details_max_potholes['Distance'],acc_synth_max_potholes['Distance']])
# axs[0,0].plot(p79_details_max_potholes['Distance'],p79_details_max_potholes['Laser5'],label='Laser 5')
# axs[0,0].plot(p79_details_max_potholes['Distance'],p79_details_max_potholes['Laser21'],label='Laser 21')
# axs[0,0].set(xlim=(np.min(dist),np.max(dist)),ylim=(laser_min,laser_max))
# axs[0,0].legend()
# axs[0,0].set_title('Potholes: ' + str(potholes[idx_max_potholes]))
# axs[1,0].plot(acc_synth_max_potholes['Distance'],acc_synth_max_potholes['synth_acc'])
# axs[1,0].set(xlim=(np.min(dist),np.max(dist)),ylim=(acc_min,acc_max))

# dist = np.concatenate([p79_details_max_cracks['Distance'],acc_synth_max_cracks['Distance']])
# axs[0,1].plot(p79_details_max_cracks['Distance'],p79_details_max_cracks['Laser5'],label='Laser 5')
# axs[0,1].plot(p79_details_max_cracks['Distance'],p79_details_max_cracks['Laser21'],label='Laser 21')
# axs[0,1].set(xlim=(np.min(dist),np.max(dist)),ylim=(laser_min,laser_max))
# axs[0,1].legend()
# axs[0,1].set_title('Cracks: '+str(cracks[idx_max_cracks]))
# axs[1,1].plot(acc_synth_max_cracks['Distance'],acc_synth_max_cracks['synth_acc'])
# axs[1,1].set(xlim=(np.min(dist),np.max(dist)),ylim=(acc_min,acc_max))

# dist = np.concatenate([p79_details_max_alligator['Distance'],acc_synth_max_alligator['Distance']])
# axs[0,2].plot(p79_details_max_alligator['Distance'],p79_details_max_alligator['Laser5'],label='Laser 5')
# axs[0,2].plot(p79_details_max_alligator['Distance'],p79_details_max_alligator['Laser21'],label='Laser 21')
# axs[0,2].set(xlim=(np.min(dist),np.max(dist)),ylim=(laser_min,laser_max))
# axs[0,2].legend()
# axs[0,2].set_title('Alligator: '+str(alligator[idx_max_alligator]))
# axs[1,2].plot(acc_synth_max_alligator['Distance'],acc_synth_max_alligator['synth_acc'])
# axs[1,2].set(xlim=(np.min(dist),np.max(dist)),ylim=(acc_min,acc_max))

# dist = np.concatenate([p79_details_max_DI['Distance'],acc_synth_max_DI['Distance']])
# axs[0,3].plot(p79_details_max_DI['Distance'],p79_details_max_DI['Laser5'],label='Laser 5')
# axs[0,3].plot(p79_details_max_DI['Distance'],p79_details_max_DI['Laser21'],label='Laser 21')
# axs[0,3].set(xlim=(np.min(dist),np.max(dist)),ylim=(laser_min,laser_max))
# axs[0,3].legend()
# axs[0,3].set_title('High DI: '+str(DI[idx_max_DI]))
# axs[1,3].plot(acc_synth_max_DI['Distance'],acc_synth_max_DI['synth_acc'])
# axs[1,3].set(xlim=(np.min(dist),np.max(dist)),ylim=(acc_min,acc_max))

# dist = np.concatenate([p79_details_min_DI['Distance'],acc_synth_min_DI['Distance']])
# axs[0,4].plot(p79_details_min_DI['Distance'],p79_details_min_DI['Laser5'],label='Laser 5')
# axs[0,4].plot(p79_details_min_DI['Distance'],p79_details_min_DI['Laser21'],label='Laser 21')
# axs[0,4].set(xlim=(np.min(dist),np.max(dist)),ylim=(laser_min,laser_max))
# axs[0,4].legend()
# axs[0,4].set_title('Low DI: '+str(DI[idx_min_DI]))
# axs[1,4].plot(acc_synth_min_DI['Distance'],acc_synth_min_DI['synth_acc'])
# axs[1,4].set(xlim=(np.min(dist),np.max(dist)),ylim=(acc_min,acc_max))

# axs[0,0].set(ylabel='Laser distance [mm]')
# axs[1,0].set(ylabel='Synthetic z acceleration')
# for ax in axs.flat:
#     ax.set(xlabel='Distance [m]')
# # # Hide x labels and tick labels for top plots and y ticks for right plots.
# # for ax in axs.flat:
# #     ax.label_outer()
# plt.show()


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

# ################################################# MINIMUM ##############################################


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