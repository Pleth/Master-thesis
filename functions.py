from faulthandler import disable
import os
import pandas as pd
import math
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tqdm import tqdm
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

def rm_aligned(gps,gt):

    dist = []
    for i in range(len(gps.index)-1):
        dist.append(math.dist([gps['lat'][i],gps['lon'][i]],[gps['lat'][i+1],gps['lon'][i+1]]))

    return dist



def data_window(gm_data):

    data = np.asarray(gm_data["synth_acc"])
    shape = np.shape(data)[0]
    n = shape - int(shape/125)*125
    data = data[:-n]
    data = data.reshape((-1,125))
    data = np.transpose(data)

    return data

def synthetic_data():

    df_dict = {}

    counter = len(glob.glob1("p79/","*.txt"))
    files = glob.glob("p79/*.txt")
    k=0
    for i in range(0,counter,2):
        #p79
        p79_gps = files[i]
        p79_laser = files[i+1]

        df_p79 = pd.read_csv(p79_laser,sep=" ")
        df_p79["Distance[m]|"] = df_p79["Distance[m]|"].str[:-1].astype(float)
        df_p79["Laser5[mm]|"] = df_p79["Laser5[mm]|"].str[:-1].astype(float)
        df_p79["Laser21[mm]|"] = df_p79["Laser21[mm]|"].str[:-1].astype(float)
        df_p79.drop(df_p79.columns.difference(['Distance[m]|','Laser5[mm]|','Laser21[mm]|']),axis=1,inplace=True)

        df_p79_gps = pd.read_csv(p79_gps,sep=" ")
        df_p79_gps["Distance[m]|"] = df_p79_gps["Distance[m]|"].str[:-1].astype(float)
        df_p79_gps["Latitude[dd.dddd]|"] = df_p79_gps["Latitude[dd.dddd]|"].str[:-1].astype(float)
        df_p79_gps["Longitude[dd.dddd]|"] = df_p79_gps["Longitude[dd.dddd]|"].str[:-1].astype(float)
        df_p79_gps.drop(df_p79_gps.columns.difference(['Distance[m]|','Latitude[dd.dddd]|','Longitude[dd.dddd]|']),axis=1,inplace=True)
            
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
                print("Loaded Synthetic Profile for trip:",int(i/2)+1,"/",int(counter/2),"- passage:",j+1,"/",len(passage))                  
                k += 1
            else:
                passagefile = hdf5file[passage[j]]
                gm_speed = pd.DataFrame(passagefile['obd.spd_veh'], columns = passagefile['obd.spd_veh'].attrs['chNames'])
                
                print("Generating Synthetic Profile for trip:",int(i/2)+1,"/",int(counter/2),"- passage:",j+1,"/",len(passage))
                synth_data = synthetic_data = create_synthetic_signal(
                                        p79_distances=np.array(df_p79["Distance[m]|"]),
                                        p79_laser5=np.array(df_p79["Laser5[mm]|"]),
                                        p79_laser21=np.array(df_p79["Laser21[mm]|"]),
                                        gm_times=np.array(gm_speed["TS_or_Distance"]),
                                        gm_speed=np.array(gm_speed["spd_veh"]))
                
                df_dict[k] = pd.DataFrame({'time':synth_data["times"].reshape(np.shape(synth_data["times"])[0]),'synth_acc':synth_data["synth_acc"]})
                df_dict[k].rename_axis(name,inplace=True)
                df_dict[k].to_csv("synth_data/"+name+".csv",index=False)
                k += 1

    return df_dict

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