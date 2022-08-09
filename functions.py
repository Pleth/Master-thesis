from faulthandler import disable
import os
import pandas as pd
import math
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tqdm import tqdm
from haversine import haversine, Unit
import glob
import h5py
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

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

def rm_aligned(gps,gt):

    dist = []
    for i in range(len(gps.index)-1):
        dist.append(math.dist([gps['lat'][i],gps['lon'][i]],[gps['lat'][i+1],gps['lon'][i+1]]))

    return dist


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


def synthetic_segmentation(synth_acc,routes):

    synth_acc = synthetic_data()
    routes = []
    for i in range(len(synth_acc)): 
        routes.append(synth_acc[i].axes[0].name)


    segments = {}

    files = glob.glob("p79/*.csv")
        
    df_cph1_hh = pd.read_csv(files[0])
    df_cph1_hh.drop(df_cph1_hh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
    df_cph1_vh = pd.read_csv(files[1])
    df_cph1_vh.drop(df_cph1_vh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
    df_cph6_hh = pd.read_csv(files[2])
    df_cph6_hh.drop(df_cph6_hh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
    df_cph6_vh = pd.read_csv(files[3])
    df_cph6_vh.drop(df_cph6_vh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)

    hdf5file_cph1_hh = h5py.File('aligned_data/CPH1_HH.hdf5', 'r')
    hdf5file_cph1_vh = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
    hdf5file_cph6_hh = h5py.File('aligned_data/CPH6_HH.hdf5', 'r')
    hdf5file_cph6_vh = h5py.File('aligned_data/CPH6_VH.hdf5', 'r')    
    
    
    iter = 0
    segments = {}
    for j in tqdm(range(len(routes))):
        synth = synth_acc[j]
        synth = synth[synth['synth_acc'].notna()]
        synth = synth[synth['gm_speed'] >= 20]
        synth = synth.reset_index(drop=True)
        route = routes[j][:7]
        if route == 'CPH1_HH':
            p79_gps = df_cph1_hh
            aran_location = pd.DataFrame(hdf5file_cph1_hh['aran/trip_1/pass_1']['Location'], columns = hdf5file_cph1_hh['aran/trip_1/pass_1']['Location'].attrs['chNames'])
        elif route == 'CPH1_VH':
            p79_gps = df_cph1_vh
            aran_location = pd.DataFrame(hdf5file_cph1_vh['aran/trip_1/pass_1']['Location'], columns = hdf5file_cph1_vh['aran/trip_1/pass_1']['Location'].attrs['chNames'])
        elif route == 'CPH6_HH':
            p79_gps = df_cph6_hh
            aran_location = pd.DataFrame(hdf5file_cph6_hh['aran/trip_1/pass_1']['Location'], columns = hdf5file_cph6_hh['aran/trip_1/pass_1']['Location'].attrs['chNames'])
        elif route == 'CPH6_VH':
            p79_gps = df_cph6_vh
            aran_location = pd.DataFrame(hdf5file_cph6_vh['aran/trip_1/pass_1']['Location'], columns = hdf5file_cph6_vh['aran/trip_1/pass_1']['Location'].attrs['chNames'])

        i,k = 0, 0
        # Get 50m from ARAN -> Find gps signal from p79 -> Get measurements from synthetic data
        while (i < (len(aran_location['LatitudeFrom'])-6) ):
            aran_start = [aran_location['LatitudeFrom'][i],aran_location['LongitudeFrom'][i]]
            aran_end = [aran_location['LatitudeTo'][i+5],aran_location['LongitudeTo'][i+5]]
            _, p79_start_idx, start_dist = find_min_gps(aran_start[0], aran_start[1], p79_gps['Latitude'].values, p79_gps['Longitude'].values)
            _, p79_end_idx, end_dist = find_min_gps(aran_end[0], aran_end[1], p79_gps['Latitude'].values, p79_gps['Longitude'].values)

            print('Iteration:',i)
            print('start_dist:',start_dist)
            print('end_dist:',end_dist)

            if start_dist < 15 and end_dist < 15:
                dfdf = p79_gps['Distance'][p79_start_idx:p79_end_idx]
                dfdf = dfdf.reset_index(drop=True)   

                synth_seg = synth[((synth['Distance'] >= np.min(dfdf)) & (synth['Distance'] <= np.max(dfdf)))]
                synth_seg = synth_seg.reset_index(drop=True)

                stat1 = synth_seg['Distance'].empty
                if stat1:
                    stat2 = True
                    stat3 = True
                else:
                    stat2 = (synth_seg['Distance'][len(synth_seg['Distance'])-1]-synth_seg['Distance'][0]) <= 40
                    stat3 = (len(synth_seg['synth_acc'])) > 5000
                if stat1 | stat2 | stat3:
                    i += 1
                else:
                    k += 1
                    i += 5
                    segments[iter] = synth_seg['synth_acc']
                    Lat_cord = [p79_gps['Latitude'][p79_start_idx],p79_gps['Latitude'][p79_end_idx]]
                    Lon_cord = [p79_gps['Longitude'][p79_start_idx],p79_gps['Longitude'][p79_end_idx]]
                    _ = plt.gca().add_patch(Rectangle((Lon_cord[0],Lat_cord[0]),Lon_cord[1]-Lon_cord[0],Lat_cord[1]-Lat_cord[0],edgecolor='green',facecolor='none',lw=1))
                
                    iter += 1
            else:
                i +=1

        
        plt.scatter(x=aran_location['LongitudeFrom'][0:100], y=aran_location['LatitudeFrom'][0:100],s=1,c="red")
        plt.scatter(x=aran_location['LongitudeTo'][0:100], y=aran_location['LatitudeTo'][0:100],s=1,c="black")
        plt.scatter(x=p79_gps[p79_gps["Longitude"] != 0]['Longitude'], y=p79_gps[p79_gps["Latitude"] != 0]['Latitude'],s=1,c="blue")
        plt.show()

    synth_segments = pd.DataFrame.from_dict(segments,orient='index').transpose()
    return synth_segments

















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