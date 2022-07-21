import os
import pandas as pd
import math
import numpy as np

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