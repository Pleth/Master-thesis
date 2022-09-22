import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import h5py
import matplotlib.pyplot as plt

from functions import *
from LiRA_functions import *

def GM_sample_segmentation(segment_size=150, overlap=0):
    files = glob.glob("aligned_data/*.hdf5")
    iter = 0
    segments = {}
    aran_segment_details = {}
    route_details = {}
    dists = []
    for j in tqdm(range(len(files))):
        route = files[j][13:]
        hdf5_route = ('aligned_data/'+route)
        hdf5file = h5py.File(hdf5_route, 'r')
        aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
        aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
        aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
        aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])                

        aligned_passes = hdf5file.attrs['aligned_passes']
        for k in range(len(aligned_passes)):
            passagefile = hdf5file[aligned_passes[k]]
            aligned_gps = pd.DataFrame(passagefile['aligned_gps'], columns = passagefile['aligned_gps'].attrs['chNames'])
            acc_fs_50 = pd.DataFrame(passagefile['acc_fs_50'], columns = passagefile['acc_fs_50'].attrs['chNames'])
            f_dist = pd.DataFrame(passagefile['f_dist'], columns = passagefile['f_dist'].attrs['chNames'])
            spd_veh = pd.DataFrame(passagefile['obd.spd_veh'], columns = passagefile['obd.spd_veh'].attrs['chNames'])

            GM_start = aligned_gps[['lat','lon']].iloc[0].values
            aran_start_idx, _ = find_min_gps_vector(GM_start,aran_location[['LatitudeFrom','LongitudeFrom']].iloc[:200].values)
            GM_end = aligned_gps[['lat','lon']].iloc[-1].values
            aran_end_idx, _ = find_min_gps_vector(GM_end,aran_location[['LatitudeTo','LongitudeTo']].iloc[-200:].values)
            aran_end_idx = (len(aran_location)-200)+aran_end_idx
            
            aran_start = [aran_location['LatitudeFrom'][aran_start_idx],aran_location['LongitudeFrom'][aran_start_idx]]
            aran_end = [aran_location['LatitudeFrom'][aran_end_idx],aran_location['LongitudeTo'][aran_end_idx]]
            GM_start_idx, _ = find_min_gps_vector(aran_start,aligned_gps[['lat','lon']].values)
            GM_end_idx, _ = find_min_gps_vector(aran_end,aligned_gps[['lat','lon']].values)
            
            i = GM_start_idx
            while (i < (GM_end_idx-segment_size) ):
                GM_start = aligned_gps[['lat','lon']].iloc[i].values
                GM_end = aligned_gps[['lat','lon']].iloc[i+segment_size-1].values
                aran_start_idx, start_dist = find_min_gps_vector(GM_start,aran_location[['LatitudeFrom','LongitudeFrom']].values)
                aran_end_idx, end_dist = find_min_gps_vector(GM_end,aran_location[['LatitudeTo','LongitudeTo']].values)

                if start_dist < 5 and end_dist < 8:
                    dfdf = aligned_gps['TS_or_Distance'][i:i+segment_size]
                    dfdf = dfdf.reset_index(drop=True)   

                    dist_seg = aligned_gps['p79_dist'][i:i+segment_size]
                    dist_seg = dist_seg.reset_index(drop=True)

                    acc_seg = acc_fs_50[((acc_fs_50['TS_or_Distance'] >= np.min(dfdf)) & (acc_fs_50['TS_or_Distance'] <= np.max(dfdf)))]
                    acc_seg = acc_seg.reset_index(drop=True)

                    spd_seg = spd_veh[((spd_veh['TS_or_Distance'] >= np.min(dfdf)) & (spd_veh['TS_or_Distance'] <= np.max(dfdf)))]
                    spd_seg = spd_seg.reset_index(drop=True)

                    stat1 = acc_seg['TS_or_Distance'].empty
                    lag = []
                    for h in range(len(dist_seg)-1):
                        lag.append(dist_seg[h+1]-dist_seg[h])        
                    large = [y for y in lag if y > 5]
                    
                    if stat1:
                        stat4 = True
                    else:
                        stat4 = False if bool(large) == False else (np.max(large) > 5)
                        stat5 = False if (len(spd_seg[spd_seg['spd_veh'] >= 20])) > 100 else True
                        stat6 = False if (len(acc_seg) == 150) else True
                        stat7 = False if abs(aran_end_idx - aran_start_idx) < 100 else True
                        
                    if stat1 | stat4 | stat5 | stat6 | stat7:
                        # print('try again',i,iter)
                        aran_new = [aran_location['LatitudeFrom'][aran_start_idx+1],aran_location['LongitudeFrom'][aran_start_idx+1]]
                        GM_new_idx, _ = find_min_gps_vector(aran_new,aligned_gps[['lat','lon']].values)
                        i = GM_new_idx if (i != GM_new_idx) else GM_new_idx+10
                    else:
                        # print('Jackpot:',i,iter)
                        p1_dist = haversine(GM_end[1], GM_end[0],aran_location['LongitudeFrom'].iloc[aran_end_idx], aran_location['LatitudeFrom'].iloc[aran_end_idx])
                        p2_dist = haversine(GM_end[1], GM_end[0],aran_location['LongitudeTo'].iloc[aran_end_idx+1], aran_location['LatitudeTo'].iloc[aran_end_idx+1])
                        if p1_dist >= p2_dist:
                            aran_new_start = aran_location[['LatitudeFrom','LongitudeFrom']].iloc[aran_end_idx+1].values
                        elif p1_dist < p2_dist:
                            aran_new_start = aran_location[['LatitudeFrom','LongitudeFrom']].iloc[aran_end_idx].values

                        # aran_new_start = aran_location[['LatitudeFrom','LongitudeFrom']].iloc[aran_end_idx+1].values

                        GM_start_idx, _ = find_min_gps_vector(aran_new_start,aligned_gps[['lat','lon']].values)
                        i = GM_start_idx
                        segments[iter] = acc_seg['acc_z']
                        aran_concat = pd.concat([aran_location[aran_start_idx:aran_end_idx+1],aran_alligator[aran_start_idx:aran_end_idx+1],aran_cracks[aran_start_idx:aran_end_idx+1],aran_potholes[aran_start_idx:aran_end_idx+1]],axis=1)
                        aran_segment_details[iter] = aran_concat
                        route_details[iter] = route[:7]+aligned_passes[k]
                        dists.append(dist_seg.iloc[-1] - dist_seg.iloc[0])
                        iter += 1
                else:
                    # print('oh no',i,iter)
                    aran_new = [aran_location['LatitudeFrom'][aran_start_idx+1],aran_location['LongitudeFrom'][aran_start_idx+1]]
                    GM_new_idx, _ = find_min_gps_vector(aran_new,aligned_gps[['lat','lon']].values)
                    if i > GM_new_idx:
                        GM_new_idx = i
                    i = GM_new_idx if (i != GM_new_idx) else GM_new_idx+10

    synth_segments = pd.DataFrame.from_dict(segments,orient='index').transpose()
    aran_segment_details = pd.concat(aran_segment_details)

    return synth_segments, aran_segment_details, route_details, dists
