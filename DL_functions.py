import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import h5py
import matplotlib.pyplot as plt

import csv

from functions import *
from LiRA_functions import *


def GM_sample_segmentation(segment_size=150, overlap=0):
    sz = str(segment_size)
    if os.path.isfile("aligned_data/sample/"+"aran_segments_"+sz+".csv"):
        synth_segments = pd.read_csv("aligned_data/sample/"+"synthetic_segments_"+sz+".csv")
        synth_segments.columns = synth_segments.columns.astype(int)
        aran_segment_details = pd.read_csv("aligned_data/sample/"+"aran_segments_"+sz+".csv",index_col=[0,1])
        route_details = eval(open("aligned_data/sample/routes_details_"+sz+".txt", 'r').read())
        with open("aligned_data/sample/distances_"+sz+".csv", newline='') as f:
            reader = csv.reader(f)
            temp = list(reader)
        dists = [float(i) for i in temp[0]]
        print("Loaded already segmented data")
    else:
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
                aran_max_idx = (len(aran_location)-200)+aran_end_idx
                
                i = aran_start_idx
                while (i < (aran_max_idx-10) ):
                    aran_start = [aran_location['LatitudeFrom'][i],aran_location['LongitudeFrom'][i]]
                    GM_start_idx, start_dist = find_min_gps_vector(aran_start,aligned_gps[['lat','lon']].values)

                    if GM_start_idx+segment_size-1 >= len(aligned_gps):
                        break

                    GM_end = aligned_gps[['lat','lon']].iloc[GM_start_idx+segment_size-1].values
                    aran_end_idx, end_dist = find_min_gps_vector(GM_end,aran_location[['LatitudeTo','LongitudeTo']].values)

                    if start_dist < 5 and end_dist < 10 and i != aran_end_idx:
                        dfdf = aligned_gps['TS_or_Distance'][GM_start_idx:GM_start_idx+segment_size]
                        dfdf = dfdf.reset_index(drop=True)   

                        dist_seg = aligned_gps['p79_dist'][GM_start_idx:GM_start_idx+segment_size]
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
                            stat7 = False if abs(aran_end_idx - i) < 100 else True
                            
                        if stat1 | stat4 | stat5 | stat6 | stat7:
                            i += 1
                        else:
                            segments[iter] = acc_seg['acc_z']
                            aran_concat = pd.concat([aran_location[i:aran_end_idx+1],aran_alligator[i:aran_end_idx+1],aran_cracks[i:aran_end_idx+1],aran_potholes[i:aran_end_idx+1]],axis=1)
                            aran_segment_details[iter] = aran_concat
                            route_details[iter] = route[:7]+aligned_passes[k]
                            dists.append(dist_seg.iloc[-1] - dist_seg.iloc[0])
                            i = aran_end_idx+1
                            iter += 1
                    else:
                        i += 1

        synth_segments = pd.DataFrame.from_dict(segments,orient='index').transpose()
        synth_segments.to_csv("aligned_data/sample/"+"synthetic_segments_"+sz+".csv",index=False)
        aran_segment_details = pd.concat(aran_segment_details)
        aran_segment_details.to_csv("aligned_data/sample/"+"aran_segments_"+sz+".csv",index=True)
        myfile = open("aligned_data/sample/routes_details_"+sz+".txt","w")
        myfile.write(str(route_details))
        myfile.close()
        with open("aligned_data/sample/distances_"+sz+".csv", 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerow(dists)

    return synth_segments, aran_segment_details, route_details, dists


def GM_sample_segmentation2(segment_size=150, overlap=0):
    sz = str(segment_size)
    if os.path.isfile("aligned_data/sample/"+"aran_segments2_"+sz+".csv"):
        synth_segments = pd.read_csv("aligned_data/sample/"+"synthetic_segments2_"+sz+".csv")
        synth_segments.columns = synth_segments.columns.astype(int)
        aran_segment_details = pd.read_csv("aligned_data/sample/"+"aran_segments2_"+sz+".csv",index_col=[0,1])
        route_details = eval(open("aligned_data/sample/routes_details2_"+sz+".txt", 'r').read())
        with open("aligned_data/sample/distances2_"+sz+".csv", newline='') as f:
            reader = csv.reader(f)
            temp = list(reader)
        dists = [float(i) for i in temp[0]]
        print("Loaded already segmented data")
    else:
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
                aran_max_idx = (len(aran_location)-200)+aran_end_idx
                
                i = aran_start_idx
                while (i < (aran_max_idx-10) ):
                    aran_start = [aran_location['LatitudeFrom'][i],aran_location['LongitudeFrom'][i]]
                    GM_start_idx, start_dist = find_min_gps_vector(aran_start,aligned_gps[['lat','lon']].values)

                    if GM_start_idx+segment_size-1 >= len(aligned_gps):
                        break

                    GM_end = aligned_gps[['lat','lon']].iloc[GM_start_idx+segment_size-1].values
                    aran_end_idx, end_dist = find_min_gps_vector(GM_end,aran_location[['LatitudeTo','LongitudeTo']].values)

                    if start_dist < 5 and end_dist < 10 and i != aran_end_idx:
                        dfdf = aligned_gps['TS_or_Distance'][GM_start_idx:GM_start_idx+segment_size]
                        dfdf = dfdf.reset_index(drop=True)   

                        dist_seg = aligned_gps['p79_dist'][GM_start_idx:GM_start_idx+segment_size]
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
                            stat7 = False if abs(aran_end_idx - i) < 100 else True
                            
                        if stat1 | stat4 | stat5 | stat6 | stat7:
                            i += 1
                        else:
                            p1_dist = haversine(GM_end[1], GM_end[0],aran_location['LongitudeFrom'].iloc[aran_end_idx], aran_location['LatitudeFrom'].iloc[aran_end_idx])
                            p2_dist = haversine(GM_end[1], GM_end[0],aran_location['LongitudeTo'].iloc[aran_end_idx+1], aran_location['LatitudeTo'].iloc[aran_end_idx+1])
                            if p1_dist > p2_dist:
                                aran_end_idx = aran_end_idx + 1 
                            elif p1_dist <= p2_dist:
                                aran_end_idx = aran_end_idx

                            segments[iter] = acc_seg['acc_z']
                            aran_concat = pd.concat([aran_location[i:aran_end_idx],aran_alligator[i:aran_end_idx],aran_cracks[i:aran_end_idx],aran_potholes[i:aran_end_idx]],axis=1)
                            aran_segment_details[iter] = aran_concat
                            route_details[iter] = route[:7]+aligned_passes[k]
                            dists.append(dist_seg.iloc[-1] - dist_seg.iloc[0])
                            i = aran_end_idx
                            iter += 1
                    else:
                        i += 1

        synth_segments = pd.DataFrame.from_dict(segments,orient='index').transpose()
        synth_segments.to_csv("aligned_data/sample/"+"synthetic_segments2_"+sz+".csv",index=False)
        aran_segment_details = pd.concat(aran_segment_details)
        aran_segment_details.to_csv("aligned_data/sample/"+"aran_segments2_"+sz+".csv",index=True)
        myfile = open("aligned_data/sample/routes_details2_"+sz+".txt","w")
        myfile.write(str(route_details))
        myfile.close()
        with open("aligned_data/sample/distances2_"+sz+".csv", 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerow(dists)

    return synth_segments, aran_segment_details, route_details, dists
