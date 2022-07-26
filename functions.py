from faulthandler import disable
import os
import pandas as pd
import math
import numpy as np
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tqdm import tqdm

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

    if os.path.isfile("synth_data/synth.csv"):
        data = pd.read_csv("synth_data/synth.csv")
    else:
        file_p79 = "tosend/data/p79_data.csv"
        file_gm = "tosend/data/green_mob_data.csv"

        df_gm = pd.read_csv(file_gm) # speed_gm,times
        df_p79 = pd.read_csv(file_p79) #distances,laser5,laser21

        synth_data = synthetic_data = create_synthetic_signal(
                                p79_distances=np.array(df_p79["distances"]),
                                p79_laser5=np.array(df_p79["laser5"]),
                                p79_laser21=np.array(df_p79["laser21"]),
                                gm_times=np.array(df_gm["times"]),
                                gm_speed=np.array(df_gm["speed_gm"]))
        
        #data = pd.DataFrame(synth_data["synth_acc"],columns = ['synth_acc'])
        data = pd.DataFrame({'time':synth_data["times"].reshape(np.shape(synth_data["times"])[0]),'synth_acc':synth_data["synth_acc"]})
        data.to_csv("synth_data/synth.csv",index=False)

    return data


def feature_extraction(data,ids):
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