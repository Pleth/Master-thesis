from faulthandler import disable
import os
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tqdm import tqdm
from haversine import haversine, haversine_vector, Unit
import glob
import h5py

from LiRA_functions import *

def calc_DI(allig, cracks, potholes):

    allig = allig.fillna(0)
    cracks = cracks.fillna(0)
    potholes = potholes.fillna(0)

    alligsum = (3*allig['AlligCracksSmall'] + 4*allig['AlligCracksMed'] + 5*allig['AlligCracksLarge'])**0.3
    cracksum = (cracks['CracksLongitudinalSmall']**2 + cracks['CracksLongitudinalMed']**3 + cracks['CracksLongitudinalLarge']**4 + \
                cracks['CracksLongitudinalSealed']**2 + 3*cracks['CracksTransverseSmall'] + 4*cracks['CracksTransverseMed'] + \
                5*cracks['CracksTransverseLarge'] + 2*cracks['CracksTransverseSealed'])**0.1
    potholesum = (5*potholes['PotholeAreaAffectedLow'] + 7*potholes['PotholeAreaAffectedMed'] + 10*potholes['PotholeAreaAffectedHigh'] + \
                  5*potholes['PotholeAreaAffectedDelam'])**0.1
    DI = alligsum + cracksum + potholesum
    return DI

def synthetic_data():

    df_dict = {}

    counter = len(glob.glob1("p79/","*.csv"))
    files = glob.glob("p79/*.csv")
    k=0
    for i in range(counter):
        #p79
        p79_file = files[i]

        df_p79 = pd.read_csv(p79_file)
        df_p79.drop(df_p79.columns.difference(['Distance','Laser5','Laser21','Latitude','Longitude']),axis=1,inplace=True)

        #Green Mobility
        file = files[i][4:11]
        gm_path = 'aligned_data/'+file+'.hdf5'
        hdf5file = h5py.File(gm_path, 'r')
        passage = hdf5file.attrs['GM_full_passes']
            
        new_passage = []
        for j in range(len(passage)):
            passagefile = hdf5file[passage[j]]
            if "obd.spd_veh" in passagefile.keys(): # some passes only contain gps and gps_match
                new_passage.append(passage[j])
            
        passage = np.array(new_passage,dtype=object)
        for j in range(len(passage)):
            name = (file+passage[j]).replace('/','_')
            if os.path.isfile("synth_data/"+name+".csv"): # Load synthetic profile if already calculated
                df_dict[k] = pd.read_csv("synth_data/"+name+".csv")
                df_dict[k].rename_axis(name,inplace=True)
                print("Loaded Synthetic Profile for trip:",i+1,"/",counter,"- passage:",j+1,"/",len(passage))                  
                k += 1
            else:
                passagefile = hdf5file[passage[j]]
                gm_speed = pd.DataFrame(passagefile['obd.spd_veh'], columns = passagefile['obd.spd_veh'].attrs['chNames'])
                
                print("Generating Synthetic Profile for trip:",i+1,"/",counter,"- passage:",j+1,"/",len(passage))
                synth_data = create_synthetic_signal(
                                        p79_distances=np.array(df_p79["Distance"]),
                                        p79_laser5=np.array(df_p79["Laser5"]),
                                        p79_laser21=np.array(df_p79["Laser21"]),
                                        gm_times=np.array(gm_speed["TS_or_Distance"]),
                                        gm_speed=np.array(gm_speed["spd_veh"]))
                
                df_dict[k] = pd.DataFrame({'time':synth_data["times"].reshape(np.shape(synth_data["times"])[0]),'synth_acc':synth_data["synth_acc"],'Distance':synth_data["p79_distances"].reshape(np.shape(synth_data["p79_distances"])[0]),'gm_speed':synth_data["gm_speed"].reshape(np.shape(synth_data["gm_speed"])[0])})
                df_dict[k].rename_axis(name,inplace=True)
                df_dict[k].to_csv("synth_data/"+name+".csv",index=False)
                k += 1

    return df_dict

def find_min_gps(drd_lat, drd_lon, gm_lat, gm_lon): # From Thea
    """Find the closest gps points between drd_lat, drd_lon
    and gm_lat, gm_lon

    Args:
        drd_lat (_type_): _description_
        drd_lon (_type_): _description_
        gm_lat (_type_): _description_
        gm_lon (_type_): _description_

    Returns:
        _type_: _description_
    """
    if np.isscalar(gm_lat):
        gm_lat = [gm_lat]
        gm_lon = [gm_lon]
    if np.isscalar(drd_lat):
        drd_lat = [drd_lat]
        drd_lon = [drd_lon]
        
    dist = np.zeros(len(drd_lat))
    dist_idx = np.zeros(len(drd_lat))
    for i, (lat,lon) in enumerate(zip(drd_lat, drd_lon)):    
        temp_dist = np.zeros(len(gm_lat))
        for j, (glat, glon) in enumerate(zip(gm_lat, gm_lon)):
            temp_dist[j] = haversine((lat, lon), (glat, glon),unit=Unit.METERS)
        dist[i] = np.min(temp_dist)
        dist_idx[i] = np.argmin(temp_dist)
    
    drd_idx = int(np.argmin(dist))
    gm_idx = int(dist_idx[drd_idx])

    return drd_idx, gm_idx, dist[drd_idx]

def haversine_np(lon1, lat1, lon2, lat2):

    lon1 = np.tile(lon1,(1,len(lon2)))
    lat1 = np.tile(lat1,(1,len(lat2)))
    lon2 = np.reshape(lon2,(1,-1))
    lat2 = np.reshape(lat2,(1,-1))

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6371 * c
    return m*1000

def find_min_gps_vector(drd,gm):
    
    # drd = np.tile(drd,(len(gm),1))

    # res = haversine_vector(drd,gm,Unit.METERS)
    res = haversine_np(drd[1],drd[0],gm[:,1],gm[:,0])
    min_idx = np.argmin(res)
    min_dist = np.min(res)
    return min_idx, min_dist

def tester_test(drd,gm):

    res = (np.sqrt((drd[0]-gm[:,0])**2+(drd[1]-gm[:,1])**2))
    min_idx = np.argmin(res)
    min_dist = np.min(res)

    if min_dist != res[min_idx]:
        print('error')

    return min_idx, min_dist

def find_min_gps_cart(drd,gm):

    res = latlon_cart_dist(drd,gm)
    min_idx = np.argmin(res)
    min_dist = np.min(res)

    return min_idx, min_dist

def latlon_cart_dist(p1,p2):
    x = 6371 * np.cos(np.radians(p1[0])) * np.cos(np.radians(p1[1]))
    y = 6371 * np.cos(np.radians(p1[0])) * np.sin(np.radians(p1[1]))
                
    x1 = 6371 * np.cos(np.radians(p2[:,0])) * np.cos(np.radians(p2[:,1]))
    y1 = 6371 * np.cos(np.radians(p2[:,0])) * np.sin(np.radians(p2[:,1]))

    dist1 = (np.sqrt((x1-x)**2+(y1-y)**2))*1000
    return dist1

def latlon_cart(p1):

    if len(p1) > 2:
        x = 6371 * np.cos(np.radians(p1[0])) * np.cos(np.radians(p1[1]))
        y = 6371 * np.cos(np.radians(p1[0])) * np.sin(np.radians(p1[1]))
    else:
        x = 6371 * np.cos(np.radians(p1[:,0])) * np.cos(np.radians(p1[:,1]))
        y = 6371 * np.cos(np.radians(p1[:,0])) * np.sin(np.radians(p1[:,1]))

    return x, y

def synthetic_segmentation(synth_acc,routes,segment_size=5,overlap=0):
    files = glob.glob("p79/*.csv")
    df_cph1_hh = pd.read_csv(files[0])
    df_cph1_hh.drop(df_cph1_hh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
    df_cph1_vh = pd.read_csv(files[1])
    df_cph1_vh.drop(df_cph1_vh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
    df_cph6_hh = pd.read_csv(files[2])
    df_cph6_hh.drop(df_cph6_hh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
    df_cph6_vh = pd.read_csv(files[3])
    df_cph6_vh.drop(df_cph6_vh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
    iter = 0
    segments = {}
    aran_segment_details = {}
    for j in tqdm(range(len(routes))):
        synth = synth_acc[j]
        synth = synth[synth['synth_acc'].notna()]
        synth = synth[synth['gm_speed'] >= 20]
        synth = synth.reset_index(drop=True)
        route = routes[j][:7]

        if route == 'CPH1_HH':
            p79_gps = df_cph1_hh
            hdf5_route = ('aligned_data/'+route+'.hdf5')
            hdf5file = h5py.File(hdf5_route, 'r')
            aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
            aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
            aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
            aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])
        elif route == 'CPH1_VH':
            p79_gps = df_cph1_vh
            hdf5_route = ('aligned_data/'+route+'.hdf5')
            hdf5file = h5py.File(hdf5_route, 'r')
            aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
            aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
            aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
            aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])
        elif route == 'CPH6_HH':
            p79_gps = df_cph6_hh
            hdf5_route = ('aligned_data/'+route+'.hdf5')
            hdf5file = h5py.File(hdf5_route, 'r')
            aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
            aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
            aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
            aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])
        elif route == 'CPH6_VH':
            p79_gps = df_cph6_vh
            hdf5_route = ('aligned_data/'+route+'.hdf5')
            hdf5file = h5py.File(hdf5_route, 'r')
            aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
            aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
            aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
            aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])
    
        p79_start = p79_gps[['Latitude','Longitude']].iloc[0].values
        aran_start_idx, _ = find_min_gps_vector(p79_start,aran_location[['LatitudeFrom','LongitudeFrom']].iloc[:100].values)
        p79_end = p79_gps[['Latitude','Longitude']].iloc[-1].values
        aran_end_idx, _ = find_min_gps_vector(p79_end,aran_location[['LatitudeTo','LongitudeTo']].iloc[-100:].values)
        aran_end_idx = (len(aran_location)-100)+aran_end_idx
        
        i = aran_start_idx
        # Get 50m from ARAN -> Find gps signal from p79 -> Get measurements from synthetic data
        while (i < (aran_end_idx-segment_size) ):
            aran_start = [aran_location['LatitudeFrom'][i],aran_location['LongitudeFrom'][i]]
            aran_end = [aran_location['LatitudeTo'][i+segment_size-1],aran_location['LongitudeTo'][i+segment_size-1]]
            p79_start_idx, start_dist = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
            p79_end_idx, end_dist = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)
            if start_dist < 15 and end_dist < 15:
                dfdf = p79_gps['Distance'][p79_start_idx:p79_end_idx]
                dfdf = dfdf.reset_index(drop=True)   

                synth_seg = synth[((synth['Distance'] >= np.min(dfdf)) & (synth['Distance'] <= np.max(dfdf)))]
                synth_seg = synth_seg.reset_index(drop=True)

                stat1 = synth_seg['Distance'].empty
                lag = []
                for h in range(len(synth_seg)-1):
                    lag.append(synth_seg['Distance'][h+1]-synth_seg['Distance'][h])        
                large = [y for y in lag if y > 5]
                
                if stat1:
                    stat2 = True
                    stat3 = True
                    stat4 = True
                else:
                    stat2 = not 40 <= (synth_seg['Distance'][len(synth_seg['Distance'])-1]-synth_seg['Distance'][0]) <= 60
                    stat3 = (len(synth_seg['synth_acc'])) > 5000
                    stat4 = False if bool(large) == False else (np.max(large) > 5)
                    
                if stat1 | stat2 | stat3 | stat4:
                    i += 1
                else:
                    i += segment_size
                    segments[iter] = synth_seg['synth_acc']
                    aran_concat = pd.concat([aran_location[i:i+segment_size],aran_alligator[i:i+segment_size],aran_cracks[i:i+segment_size],aran_potholes[i:i+segment_size]],axis=1)
                    aran_segment_details[iter] = aran_concat
                    iter += 1
            else:
                i +=1

    synth_segments = pd.DataFrame.from_dict(segments,orient='index').transpose()
    synth_segments.to_csv("synth_data/"+"synthetic_segments"+".csv",index=False)
    aran_segment_details = pd.concat(aran_segment_details)
    aran_segment_details.to_csv("synth_data/"+"aran_segments"+".csv",index=False)
        
    return synth_segments, aran_segment_details














def feature_extraction(data,ids):

    Time_domain_names = ['Max','Min','Mean','Median','RMS','Peak_to_peak','Ten_point_average']
    Frequency_domain_names = ['Average_band_power','RMS_band','Max_band']
    Wavelet_domain_names = ['RMS_Mort4','Ten_point_Mort4','RMS_Mort5','Ten_point_Mort5','RMS_6Daub4','Ten_point_6Daub4',
                            'RMS_6Daub5','Ten_point_6Daub5','RMS_10Daub4','Ten_point_10Daub4','RMS_10Daub5','Ten_point_10Daub5']

    Features = {}

    series = np.random.randn(10)

    # Time domain features 
    Features[Time_domain_names[0]] = [np.max(series)]
    Features[Time_domain_names[1]] = [np.min(series)]
    Features[Time_domain_names[2]] = [np.mean(series)]
    Features[Time_domain_names[3]] = [np.median(series)]
    Features[Time_domain_names[4]] = [np.square(series).mean()]
    Features[Time_domain_names[5]] = [np.ptp(series)]
    Features[Time_domain_names[6]] = [np.sum(-series[np.argpartition(series,5)[:5]]+series[np.argpartition(series,-5)[-5:]])/5]
    
    extracted_features_2 = pd.DataFrame(Features).T
    extracted_features_2


    
    # Frequency domain features    
    from scipy import signal
    from scipy.integrate import simps
    freqs, psd = signal.welch(series,250,nperseg=4*250)
    freq_res = freqs[1]-freqs[0]
    
    low, high = 0.5, 4

    # Find intersecting values in frequency vector
    idx_delta = np.logical_and(freqs >= low, freqs <= high)
    
    delta_power = simps(psd[idx_delta],dx=freq_res)
    
    
    
    
    
    # Wavelet domain features
    












    #extracted_features = extract_features(out.iloc[:,0:2],column_id="id")
    if os.path.isfile("synth_data/extracted_features.csv"):
        feature_names = extract_features(pd.concat([ids,data.iloc[:,1]],axis=1),column_id="id",disable_progressbar=True)
        feature_names = np.transpose(feature_names)
        data = pd.read_csv("synth_data/extracted_features.csv")
    else:
        extracted_features = []
        for i in tqdm(range(np.shape(data)[1])):
            #print("current iteration:",i,"out of:",np.shape(data)[1])
            if(i==0):   
                feature_names = extract_features(pd.concat([ids,data.iloc[:,i]],axis=1),column_id="id",disable_progressbar=True)
                extracted_features.append(np.transpose(feature_names))
                feature_names = np.transpose(feature_names)
            else:
                extracted_features.append(np.transpose(extract_features(pd.concat([ids,data.iloc[:,i]],axis=1),column_id="id",disable_progressbar=True)))
        data = pd.DataFrame(np.concatenate(extracted_features,axis=1))
        #data = data.fillna(0)
        impute(data)
        data.to_csv("synth_data/extracted_features.csv",index=False)

    return data,feature_names