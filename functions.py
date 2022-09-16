import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from haversine import haversine, Unit
import glob
import h5py
import tsfel
import time
import joblib
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyRegressor

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
    return DI, alligsum, cracksum, potholesum

def calc_DI_v2(aran):
    aran = aran.fillna(0)
    
    alligsum = (3*aran['AlligCracksSmall'] + 4*aran['AlligCracksMed'] + 5*aran['AlligCracksLarge'])**0.3
    cracksum = (aran['CracksLongitudinalSmall']**2 + aran['CracksLongitudinalMed']**3 + aran['CracksLongitudinalLarge']**4 + \
                aran['CracksLongitudinalSealed']**2 + 3*aran['CracksTransverseSmall'] + 4*aran['CracksTransverseMed'] + \
                5*aran['CracksTransverseLarge'] + 2*aran['CracksTransverseSealed'])**0.1
    potholesum = (5*aran['PotholeAreaAffectedLow'] + 7*aran['PotholeAreaAffectedMed'] + 10*aran['PotholeAreaAffectedHigh'] + \
                  5*aran['PotholeAreaAffectedDelam'])**0.1
    DI = alligsum + cracksum + potholesum
    return DI, alligsum, cracksum, potholesum


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
    if os.path.isfile("synth_data/"+"aran_segments"+".csv"):
        synth_segments = pd.read_csv("synth_data/"+"synthetic_segments"+".csv")
        aran_segment_details = pd.read_csv("synth_data/"+"aran_segments"+".csv")
        route_details = eval(open("synth_data/routes_details.txt", 'r').read())
        print("Loaded already segmented data")              
        
    else:    
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
        route_details = {}
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

                if start_dist < 5 and end_dist < 5:
                    dfdf = p79_gps['Distance'][p79_start_idx:p79_end_idx+1]
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
                        route_details[iter] = routes[j]
                        iter += 1
                else:
                    i +=1

        synth_segments = pd.DataFrame.from_dict(segments,orient='index').transpose()
        synth_segments.to_csv("synth_data/"+"synthetic_segments"+".csv",index=False)
        aran_segment_details = pd.concat(aran_segment_details)
        aran_segment_details.to_csv("synth_data/"+"aran_segments"+".csv",index=False)
        myfile = open("synth_data/routes_details.txt","w")
        myfile.write(str(route_details))
        myfile.close()
        aran_segment_details = pd.read_csv("synth_data/"+"aran_segments"+".csv")
        
    return synth_segments, aran_segment_details, route_details

def feature_extraction(data,id,fs=250,min_idx=0):
    cfg_file = tsfel.get_features_by_domain()
    if os.path.isfile(id+'.csv'):
        feature_names = np.transpose(tsfel.time_series_features_extractor(cfg_file,data[str(min_idx)].dropna(),fs=fs,verbose=0))
        data = pd.read_csv(id+'.csv')
    else:
        extracted_features = []
        feature_names = np.transpose(tsfel.time_series_features_extractor(cfg_file,data[str(min_idx)].dropna(),fs=fs,verbose=0))
        for i in tqdm(range(np.shape(data)[1])):
            temp = np.transpose(tsfel.time_series_features_extractor(cfg_file,data[str(i)].dropna(),fs=fs,verbose=0))
            diff = list(set(temp.index) - set(feature_names.index))
            extracted_features.append(temp.drop(diff))
        data = pd.DataFrame(np.concatenate(extracted_features,axis=1))
        data.to_csv(id+'.csv',index=False)

    return data,feature_names


def feature_extraction2(data,id,fs=250,min_idx=0):
    cfg_file = tsfel.get_features_by_domain()
    if os.path.isfile(id+'.csv'):
        feature_names = np.transpose(tsfel.time_series_features_extractor(cfg_file,data[min_idx].dropna(),fs=fs,verbose=0))
        data = pd.read_csv(id+'.csv')
    else:
        extracted_features = []
        feature_names = np.transpose(tsfel.time_series_features_extractor(cfg_file,data[min_idx].dropna(),fs=fs,verbose=0))
        for i in tqdm(range(np.shape(data)[1])):
            temp = np.transpose(tsfel.time_series_features_extractor(cfg_file,data[i].dropna(),fs=fs,verbose=0))
            diff = list(set(temp.index) - set(feature_names.index))
            extracted_features.append(temp.drop(diff))
        data = pd.DataFrame(np.concatenate(extracted_features,axis=1))
        data.to_csv(id+'.csv',index=False)

    return data,feature_names

def method_RandomForest(features_train, features_test, y_train, y_test, id, model=False, gridsearch=0, cv_in=5, verbose=3,n_jobs=None):
    cv = cv_in[0]

    X_train, X_test = features_train.T, features_test.T
    y_train, y_test = y_train.values.reshape(-1,), y_test.values.reshape(-1,)

    if model != False:
        rf_train = joblib.load('models/RandomForest_best_model_'+id+'.sav')
        # loaded_model.fit(X_train,y_train)
        y_pred = rf_train.predict(X_test)

        dummy_regr = DummyRegressor(strategy="mean")
        dummy_regr.fit(X_train,y_train)
        dummy_pred = dummy_regr.predict(X_test)
        r2_dummy = r2_score(y_test,dummy_pred)
        print('Dummy R2 value '+id+':', r2_dummy)
        
        r2 = r2_score(y_test,y_pred)
        MSE = mean_squared_error(y_test,y_pred, squared=True)
        RMSE = mean_squared_error(y_test,y_pred, squared=False)
        MAE = mean_absolute_error(y_test,y_pred)

        train_pred = rf_train.predict(X_train)
        train_y = y_train
        r2_train = r2_score(train_y,train_pred)
        MSE_train = mean_squared_error(train_y,train_pred, squared=True)
        RMSE_train = mean_squared_error(train_y,train_pred, squared=False)
        MAE_train = mean_absolute_error(train_y,train_pred)
    else:        
        parameters={'criterion': ['squared_error'],
                    'bootstrap': [True],
                    'max_depth': [2, 3, 4, 5, 6, 7],
                    'max_features': [2, 6, 10, 14, 18, 'log2', 'sqrt'],
                    'min_samples_leaf': [2, 4, 6, 8, 10],
                    'min_samples_split': [2, 4, 6, 8, 10, 12, 15, 20, 25],
                    'n_estimators': [250]}
        parameters={'criterion': ['squared_error'],
                    'bootstrap': [True],
                    'max_depth': [5],
                    'max_features': [18],
                    'min_samples_leaf': [2],
                    'min_samples_split': [4],
                    'n_estimators': [250]}
        
        start_time = time.time()
        if gridsearch == 1:
            rf_train = GridSearchCV(RandomForestRegressor(), parameters, cv = cv, scoring='r2',verbose=verbose,n_jobs=n_jobs,return_train_score=True) # scoring='neg_mean_squared_error'
            rf_train.fit(X_train,y_train)
            joblib.dump(rf_train,'models/RandomForest_best_model_'+id+'.sav')
        else:
            rf_train = RandomForestRegressor(n_estimators=250,max_depth=7,max_features=18,bootstrap=True,criterion='squared_error',min_samples_leaf=2,min_samples_split=4)
            #rf_train.fit(X_train,y_train)
            rf_train.fit(X_train,y_train)
        end_time = time.time()
        run_time = end_time - start_time
        print('Run time:',round(run_time/60,2),'mins')
        
        y_pred = rf_train.predict(X_test)

        r2 = r2_score(y_test,y_pred)
        MSE = mean_squared_error(y_test,y_pred, squared=True)
        RMSE = mean_squared_error(y_test,y_pred, squared=False)
        MAE = mean_absolute_error(y_test,y_pred)

        train_pred = rf_train.predict(X_train)
        train_y = y_train
        r2_train = r2_score(train_y,train_pred)
        MSE_train = mean_squared_error(train_y,train_pred, squared=True)
        RMSE_train = mean_squared_error(train_y,train_pred, squared=False)
        MAE_train = mean_absolute_error(train_y,train_pred)
    return {"R2":[r2 ,r2_train], "MSE": [MSE, MSE_train], "RMSE": [RMSE, RMSE_train], "MAE": [MAE, MAE_train],"Gridsearchcv_obj": rf_train}


def custom_splits(aran_segments,route_details,save=False):
    if save & os.path.isfile("synth_data/split1"):
        with open("synth_data/split1",'rb') as fp:
            split1 = pickle.load(fp)
        with open("synth_data/split2",'rb') as fp:
            split2 = pickle.load(fp)
        with open("synth_data/split3",'rb') as fp:
            split3 = pickle.load(fp)
        with open("synth_data/split4",'rb') as fp:
            split4 = pickle.load(fp)
        with open("synth_data/split5",'rb') as fp:
            split5 = pickle.load(fp)
        print("Loaded cross validation splits")
    else:
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

        cph1_len = (len(cph1_hh) + len(cph1_vh))

        counter = 0
        chain_start = 645
        for i in range(int(cph1_len)):
            counter1 = np.sum(np.sum(aran_segments['EndChainage'].iloc[cph1_hh] < chain_start))
            counter2 = np.sum(np.sum(aran_segments['BeginChainage'].iloc[cph1_vh] < chain_start))
            counter = (counter1 + counter2)/5
            if (counter >= (cph1_len/5)/3):
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
        print('split 1',chain_start)
        split1 = list(set(list((np.array(temp_cph1_hh)/5).astype(int)))) + list(set(list((np.array(temp_cph1_vh)/5).astype(int))))

        counter = 0
        chain_end = chain_start
        chain_start = chain_start + 50
        for i in range(int(cph1_len)):
            counter1 = np.sum(np.sum((chain_end < aran_segments['BeginChainage'].iloc[cph1_hh]) & (aran_segments['EndChainage'].iloc[cph1_hh] < chain_start)))
            counter2 = np.sum(np.sum((chain_end < aran_segments['EndChainage'].iloc[cph1_vh]) & (aran_segments['BeginChainage'].iloc[cph1_vh] < chain_start)))
            counter = (counter1 + counter2)/5
            if (counter >= (cph1_len/5)/3):
                break
            chain_start += 10

        dd = (chain_end < aran_segments['BeginChainage'][cph1_hh]) & (aran_segments['EndChainage'][cph1_hh] < chain_start)
        temp_cph1_hh = []
        for i in range(len(cph1_hh)):
            if dd.iloc[i] == True:
                temp_cph1_hh.append(cph1_hh[i])

        dd = (chain_end < aran_segments['EndChainage'][cph1_vh]) & (aran_segments['BeginChainage'][cph1_vh] < chain_start)
        temp_cph1_vh = []
        for i in range(len(cph1_vh)):
            if dd.iloc[i] == True:
                temp_cph1_vh.append(cph1_vh[i])
        print('split 2',chain_start)
        split2 = list(set(list((np.array(temp_cph1_hh)/5).astype(int)))) + list(set(list((np.array(temp_cph1_vh)/5).astype(int))))

        counter = 0
        chain_end = chain_start
        chain_start = chain_start + 50
        for i in range(int(cph1_len)):
            counter1 = np.sum(np.sum(chain_end < aran_segments['BeginChainage'].iloc[cph1_hh]))
            counter2 = np.sum(np.sum(chain_end < aran_segments['EndChainage'].iloc[cph1_vh]))
            counter = (counter1 + counter2)/5
            if (counter >= (cph1_len/5)/3):
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
        print('split 3',chain_start)
        split3 = list(set(list((np.array(temp_cph1_hh)/5).astype(int)))) + list(set(list((np.array(temp_cph1_vh)/5).astype(int))))


        cph6_len = (len(cph6_hh) + len(cph6_vh))

        counter = 0
        chain_start = 0
        for i in range(int(cph6_len)):
            counter1 = np.sum(np.sum(aran_segments['EndChainage'].iloc[cph6_hh] < chain_start))
            counter2 = np.sum(np.sum(aran_segments['BeginChainage'].iloc[cph6_vh] < chain_start))
            counter = (counter1 + counter2)/5
            if (counter >= (cph6_len/5)/2):
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
        print('split 4',chain_start)
        split4 = list(set(list((np.array(temp_cph6_hh)/5).astype(int)))) + list(set(list((np.array(temp_cph6_vh)/5).astype(int))))
        
        counter = 0
        chain_end = chain_start
        for i in range(int(cph6_len)):
            counter1 = np.sum(np.sum(chain_end < aran_segments['BeginChainage'].iloc[cph6_hh]))
            counter2 = np.sum(np.sum(chain_end < aran_segments['EndChainage'].iloc[cph6_vh]))
            counter = (counter1 + counter2)/5
            if (counter >= (cph6_len/5)/2):
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
        print('split 5',chain_start)
        split5 = list(set(list((np.array(temp_cph6_hh)/5).astype(int)))) + list(set(list((np.array(temp_cph6_vh)/5).astype(int)))) 

        s1 = set(split1)
        s2 = set(split2)
        s3 = set(split3)
        sp1 = s1.difference(s2)
        sp1 = sp1.difference(s3)
        sp2 = s2.difference(sp1)
        split2 = list(sp2.difference(s3))
        split1 = list(sp1)

        s4 = set(split4)
        s5 = set(split5)
        split4 = list(s4.difference(s5))
        if save:
            with open("synth_data/split1","wb") as fp:
                pickle.dump(split1,fp)
            with open("synth_data/split2","wb") as fp:
                pickle.dump(split2,fp)
            with open("synth_data/split3","wb") as fp:
                pickle.dump(split3,fp)
            with open("synth_data/split4","wb") as fp:
                pickle.dump(split4,fp)
            with open("synth_data/split5","wb") as fp:
                pickle.dump(split5,fp)

    cv = [(split2+split3+split4,split5),
          (split5+split3+split4,split2),
          (split5+split2+split4,split3),
          (split5+split2+split3,split4)]

    cv2 = [(split2+split3+split4+split5,split1),
           (split1+split3+split4+split5,split2),
           (split1+split2+split4+split5,split3),
           (split1+split2+split3+split5,split4),
           (split1+split2+split3+split4,split5)]

    splits = {'1': split1, '2': split2, '3': split3, '4': split4, '5': split5}

    return cv, split1, cv2, splits

def rearange_splits(splits,features):
    cv_train = []
    split_test = splits['1']
    features=features.T
    train = []
    train = features.iloc[splits['2']].reset_index(drop=True)
    train = pd.concat([train,features.iloc[splits['3']].reset_index(drop=True)],ignore_index=True)
    train = pd.concat([train,features.iloc[splits['4']].reset_index(drop=True)],ignore_index=True)
    train = pd.concat([train,features.iloc[splits['5']].reset_index(drop=True)],ignore_index=True)
    train = train.T
    test = features.iloc[splits['1']].reset_index(drop=True).T

    split2 = list(range(0,len(splits['2'])))
    split3 = list(range(len(split2),len(split2)+len(splits['3'])))
    split4 = list(range(len(split2+split3),len(split2+split3)+len(splits['4'])))
    split5 = list(range(len(split2+split3+split4),len(split2+split3+split4)+len(splits['5'])))

    cv_train = [(split2+split3+split4,split5),
                (split5+split3+split4,split2),
                (split5+split2+split4,split3),
                (split5+split2+split3,split4)]

    return cv_train, split_test, train, test


def real_splits(features,aran_segments,route_details,cut,split_nr):

    cph1 = []
    cph6 = []
    for i in range(len(route_details)):
        if (route_details[i][:7] == 'CPH1_VH') | (route_details[i][:7] == 'CPH1_HH'):
            cph1.append(i)
        elif  (route_details[i][:7] == 'CPH6_VH') | (route_details[i][:7] == 'CPH6_HH'):
            cph6.append(i)

    cph1_aran = {}
    for i in range(len(cph1)):
        cph1_aran[i] = aran_segments.loc[i]
    cph1_aran = pd.concat(cph1_aran)

    cph6_aran = {}
    for i in range(len(cph6)):
        cph6_aran[i] = aran_segments.loc[i]
    cph6_aran = pd.concat(cph6_aran)
    
    cph1_aran.index = cph1_aran.index.get_level_values(0)
    cph6_aran.index = cph6_aran.index.get_level_values(0)

    split1 = []
    for i in list(cph1_aran[cph1_aran['BeginChainage'] < cut[0]].index):
       if i not in split1:
          split1.append(i)
    split2 = []
    for i in list(cph1_aran[ (cph1_aran['BeginChainage'] >= cut[0]) & (cph1_aran['BeginChainage'] <= cut[1]) ].index):
       if i not in split2:
          split2.append(i)
    split3 = []
    for i in list(cph1_aran[cph1_aran['BeginChainage'] > cut[1]].index):
       if i not in split3:
          split3.append(i)
    
    split4 = []
    for i in list(cph6_aran[cph6_aran['BeginChainage'] < cut[2]].index):
       if i not in split4:
          split4.append(i)
    split5 = []
    for i in list(cph6_aran[cph6_aran['BeginChainage'] >= cut[2]].index):
       if i not in split5:
          split5.append(i)
    
    splits = {'1': split1, '2': split2, '3': split3, '4': split4, '5': split5}

    features=features.T
    if split_nr == 'split1':
        train = []
        train = features.iloc[split2].reset_index(drop=True)
        train = pd.concat([train,features.iloc[split3].reset_index(drop=True)],ignore_index=True)
        train = pd.concat([train,features.iloc[split4].reset_index(drop=True)],ignore_index=True)
        train = pd.concat([train,features.iloc[split5].reset_index(drop=True)],ignore_index=True)
        train = train.T
        test = features.iloc[split1].reset_index(drop=True).T

        split2 = list(range(0,len(splits['2'])))
        split3 = list(range(len(split2),len(split2)+len(splits['3'])))
        split4 = list(range(len(split2+split3),len(split2+split3)+len(splits['4'])))
        split5 = list(range(len(split2+split3+split4),len(split2+split3+split4)+len(splits['5'])))

        split_test = list(range(0,len(splits['1'])))
        cv_train = [(split2+split3+split4,split5),
                    (split5+split3+split4,split2),
                    (split5+split2+split4,split3),
                    (split5+split2+split3,split4)]

    if split_nr == 'split2':
        train = []
        train = features.iloc[split1].reset_index(drop=True)
        train = pd.concat([train,features.iloc[split3].reset_index(drop=True)],ignore_index=True)
        train = pd.concat([train,features.iloc[split4].reset_index(drop=True)],ignore_index=True)
        train = pd.concat([train,features.iloc[split5].reset_index(drop=True)],ignore_index=True)
        train = train.T
        test = features.iloc[split2].reset_index(drop=True).T

        split1 = list(range(0,len(splits['1'])))
        split3 = list(range(len(split1),len(split1)+len(splits['3'])))
        split4 = list(range(len(split1+split3),len(split1+split3)+len(splits['4'])))
        split5 = list(range(len(split1+split3+split4),len(split1+split3+split4)+len(splits['5'])))

        split_test = list(range(0,len(splits['2'])))
        cv_train = [(split1+split3+split4,split5),
                    (split5+split3+split4,split1),
                    (split5+split1+split4,split3),
                    (split5+split1+split3,split4)]
        

    if split_nr == 'split3':
        train = []
        train = features.iloc[split1].reset_index(drop=True)
        train = pd.concat([train,features.iloc[split2].reset_index(drop=True)],ignore_index=True)
        train = pd.concat([train,features.iloc[split4].reset_index(drop=True)],ignore_index=True)
        train = pd.concat([train,features.iloc[split5].reset_index(drop=True)],ignore_index=True)
        train = train.T
        test = features.iloc[split3].reset_index(drop=True).T

        split1 = list(range(0,len(splits['1'])))
        split2 = list(range(len(split1),len(split1)+len(splits['2'])))
        split4 = list(range(len(split1+split2),len(split1+split2)+len(splits['4'])))
        split5 = list(range(len(split1+split2+split4),len(split1+split2+split4)+len(splits['5'])))

        split_test = list(range(0,len(splits['3'])))
        cv_train = [(split2+split1+split4,split5),
                    (split5+split1+split4,split2),
                    (split5+split2+split4,split1),
                    (split5+split2+split1,split4)]
    if split_nr == 'split4':
        train = []
        train = features.iloc[split1].reset_index(drop=True)
        train = pd.concat([train,features.iloc[split2].reset_index(drop=True)],ignore_index=True)
        train = pd.concat([train,features.iloc[split3].reset_index(drop=True)],ignore_index=True)
        train = pd.concat([train,features.iloc[split5].reset_index(drop=True)],ignore_index=True)
        train = train.T
        test = features.iloc[split4].reset_index(drop=True).T

        split1 = list(range(0,len(splits['1'])))
        split2 = list(range(len(split1),len(split1)+len(splits['2'])))
        split3 = list(range(len(split1+split2),len(split1+split2)+len(splits['3'])))
        split5 = list(range(len(split1+split2+split3),len(split1+split2+split3)+len(splits['5'])))

        split_test = list(range(0,len(splits['4'])))
        cv_train = [(split2+split3+split1,split5),
                    (split5+split3+split1,split2),
                    (split5+split2+split1,split3),
                    (split5+split2+split3,split1)]

    if split_nr == 'split5':
        train = []
        train = features.iloc[split1].reset_index(drop=True)
        train = pd.concat([train,features.iloc[split2].reset_index(drop=True)],ignore_index=True)
        train = pd.concat([train,features.iloc[split3].reset_index(drop=True)],ignore_index=True)
        train = pd.concat([train,features.iloc[split4].reset_index(drop=True)],ignore_index=True)
        train = train.T
        test = features.iloc[split5].reset_index(drop=True).T

        split1 = list(range(0,len(splits['1'])))
        split2 = list(range(len(split1),len(split1)+len(splits['2'])))
        split3 = list(range(len(split1+split2),len(split1+split2)+len(splits['3'])))
        split4 = list(range(len(split1+split2+split3),len(split1+split2+split3)+len(splits['4'])))

        split_test = list(range(0,len(splits['5'])))
        cv_train = [(split2+split3+split4,split1),
                    (split1+split3+split4,split2),
                    (split1+split2+split4,split3),
                    (split1+split2+split3,split4)]

    return cv_train, split_test, train, test, splits

def test_synthetic_data():

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
        j = 0
        name = (file).replace('/','_')
        if os.path.isfile("synth_data/tests/"+name+".csv"): # Load synthetic profile if already calculated
            df_dict[k] = pd.read_csv("synth_data/tests/"+name+".csv")
            df_dict[k].rename_axis(name,inplace=True)
            print("Loaded Synthetic Profile for trip:",i+1,"/",counter)                  
            k += 1
        else:
            passagefile = hdf5file[passage[j]]
            gm_speed = pd.DataFrame(passagefile['obd.spd_veh'], columns = passagefile['obd.spd_veh'].attrs['chNames'])
            gm_speed['spd_veh'] = 45 # choose constant speed
            
            print("Generating Synthetic Profile for trip:",i+1,"/",counter)
            synth_data = create_synthetic_signal(
                                    p79_distances=np.array(df_p79["Distance"]),
                                    p79_laser5=np.array(df_p79["Laser5"]),
                                    p79_laser21=np.array(df_p79["Laser21"]),
                                    gm_times=np.array(gm_speed["TS_or_Distance"]),
                                    gm_speed=np.array(gm_speed["spd_veh"]))
            
            df_dict[k] = pd.DataFrame({'time':synth_data["times"].reshape(np.shape(synth_data["times"])[0]),'synth_acc':synth_data["synth_acc"],'Distance':synth_data["p79_distances"].reshape(np.shape(synth_data["p79_distances"])[0]),'gm_speed':synth_data["gm_speed"].reshape(np.shape(synth_data["gm_speed"])[0]),'laser':synth_data["profile"].reshape(np.shape(synth_data["profile"])[0])})
            df_dict[k].rename_axis(name,inplace=True)
            df_dict[k].to_csv("synth_data/tests/"+name+".csv",index=False)
            k += 1

    return df_dict

def test_segmentation(synth_acc,routes,segment_size=5,overlap=0):
    if os.path.isfile("synth_data/tests/"+"aran_segments"+".csv"):
        synth_segments = pd.read_csv("synth_data/tests/"+"synthetic_segments"+".csv")
        aran_segment_details = pd.read_csv("synth_data/tests/"+"aran_segments"+".csv")
        route_details = eval(open("synth_data/tests/routes_details.txt", 'r').read())
        laser_segments = pd.read_csv("synth_data/tests/"+"laser_segments"+".csv")
        print("Loaded already segmented data")              
        
    else:    
        files = glob.glob("p79/*.csv")
        df_cph1_hh = pd.read_csv(files[0])
        df_cph1_vh = pd.read_csv(files[1])
        df_cph6_hh = pd.read_csv(files[2])
        df_cph6_vh = pd.read_csv(files[3])
        iter = 0
        segments = {}
        lasers = {}
        aran_segment_details = {}
        route_details = {}
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

                if start_dist < 5 and end_dist < 5:
                    dfdf = p79_gps['Distance'][p79_start_idx:p79_end_idx+1]
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
                        lasers[iter] = synth_seg['laser']
                        aran_concat = pd.concat([aran_location[i:i+segment_size],aran_alligator[i:i+segment_size],aran_cracks[i:i+segment_size],aran_potholes[i:i+segment_size]],axis=1)
                        aran_segment_details[iter] = aran_concat
                        route_details[iter] = routes[j]
                        iter += 1
                else:
                    i +=1

        synth_segments = pd.DataFrame.from_dict(segments,orient='index').transpose()
        synth_segments.to_csv("synth_data/tests/"+"synthetic_segments"+".csv",index=False)
        aran_segment_details = pd.concat(aran_segment_details)
        aran_segment_details.to_csv("synth_data/tests/"+"aran_segments"+".csv",index=False)
        laser_segments = pd.DataFrame.from_dict(lasers,orient='index').transpose()
        laser_segments.to_csv("synth_data/tests/"+"laser_segments"+".csv",index=False)
        
        myfile = open("synth_data/tests/routes_details.txt","w")
        myfile.write(str(route_details))
        myfile.close()
        aran_segment_details = pd.read_csv("synth_data/tests/"+"aran_segments"+".csv")
        
    return synth_segments, aran_segment_details, route_details, laser_segments



def GM_segmentation(segment_size=5,overlap=0):
    if os.path.isfile("aligned_data/"+"aran_segments"+".csv"):
        synth_segments = pd.read_csv("aligned_data/"+"synthetic_segments"+".csv")
        aran_segment_details = pd.read_csv("aligned_data/"+"aran_segments"+".csv")
        route_details = eval(open("aligned_data/routes_details.txt", 'r').read())
        print("Loaded already segmented data")              
        
    else:
        files = glob.glob("aligned_data/*.hdf5")
        iter = 0
        segments = {}
        aran_segment_details = {}
        route_details = {}
        for j in tqdm(range(len(files))):
            route = files[j][13:]
            hdf5_route = ('aligned_data/'+route)
            hdf5file = h5py.File(hdf5_route, 'r')
            aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
            aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
            aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
            aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

            # if (route[:7] == 'CPH6_VH') | (route[:7] == 'CPH6_HH'):
            #     idx = aran_location[aran_location['BeginChainage'] < 9200].index
            #     aran_location.drop(idx, inplace=True)
            #     aran_location = aran_location.reset_index(drop=True)
            #     aran_alligator.drop(idx, inplace=True)
            #     aran_alligator = aran_alligator.reset_index(drop=True)
            #     aran_cracks.drop(idx, inplace=True)
            #     aran_cracks = aran_cracks.reset_index(drop=True)
            #     aran_potholes.drop(idx, inplace=True)
            #     aran_potholes = aran_potholes.reset_index(drop=True)
                

            aligned_passes = hdf5file.attrs['aligned_passes']
            for k in range(len(aligned_passes)):
                passagefile = hdf5file[aligned_passes[k]]
                aligned_gps = pd.DataFrame(passagefile['aligned_gps'], columns = passagefile['aligned_gps'].attrs['chNames'])
                acc_fs_50 = pd.DataFrame(passagefile['acc_fs_50'], columns = passagefile['acc_fs_50'].attrs['chNames'])
                f_dist = pd.DataFrame(passagefile['f_dist'], columns = passagefile['f_dist'].attrs['chNames'])
                spd_veh = pd.DataFrame(passagefile['obd.spd_veh'], columns = passagefile['obd.spd_veh'].attrs['chNames'])

                # if (route[:7] == 'CPH6_VH') | (route[:7] == 'CPH6_HH'):
                #     aligned_gps.drop(idx, inplace=True)
                #     aligned_gps = aligned_gps.reset_index(drop=True)
                #     acc_fs_50.drop(idx, inplace=True)
                #     acc_fs_50 = acc_fs_50.reset_index(drop=True)
                #     f_dist.drop(idx, inplace=True)
                #     f_dist = f_dist.reset_index(drop=True)

                # plt.scatter(x=aran_location['LongitudeFrom'], y=aran_location['LatitudeFrom'],s=1,c="red")
                # plt.scatter(x=aligned_gps[aligned_gps['lon'] > 12.45]['lon'], y=aligned_gps[aligned_gps['lat'] > 55.40]['lat'],s=1,c="blue")
                # plt.show()
        
                GM_start = aligned_gps[['lat','lon']].iloc[0].values
                aran_start_idx, _ = find_min_gps_vector(GM_start,aran_location[['LatitudeFrom','LongitudeFrom']].iloc[:200].values)
                GM_end = aligned_gps[['lat','lon']].iloc[-1].values
                aran_end_idx, _ = find_min_gps_vector(GM_end,aran_location[['LatitudeTo','LongitudeTo']].iloc[-200:].values)
                aran_end_idx = (len(aran_location)-200)+aran_end_idx
                
                i = aran_start_idx
                # Get 50m from ARAN -> Find gps signal from p79 -> Get measurements from synthetic data
                while (i < (aran_end_idx-segment_size) ):
                    aran_start = [aran_location['LatitudeFrom'][i],aran_location['LongitudeFrom'][i]]
                    aran_end = [aran_location['LatitudeTo'][i+segment_size-1],aran_location['LongitudeTo'][i+segment_size-1]]
                    GM_start_idx, start_dist = find_min_gps_vector(aran_start,aligned_gps[['lat','lon']].values)
                    GM_end_idx, end_dist = find_min_gps_vector(aran_end,aligned_gps[['lat','lon']].values)

                    if start_dist < 5 and end_dist < 5:
                        dfdf = aligned_gps['TS_or_Distance'][GM_start_idx:GM_end_idx+1]
                        dfdf = dfdf.reset_index(drop=True)   

                        dist_seg = aligned_gps['p79_dist'][GM_start_idx:GM_end_idx+1]
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
                            stat2 = True
                            stat3 = True
                            stat4 = True
                        else:
                            stat2 = not 40 <= (dist_seg[len(dist_seg)-1]-dist_seg[0]) <= 60
                            stat3 = (len(acc_seg['acc_z'])) > 5000
                            stat4 = False if bool(large) == False else (np.max(large) > 5)
                            stat5 = False if (len(spd_seg[spd_seg['spd_veh'] >= 20])) > len(acc_seg)*0.8 else True
                            
                        if stat1 | stat2 | stat3 | stat4 | stat5:
                            i += 1
                        else:
                            i += segment_size
                            segments[iter] = acc_seg['acc_z']
                            aran_concat = pd.concat([aran_location[i:i+segment_size],aran_alligator[i:i+segment_size],aran_cracks[i:i+segment_size],aran_potholes[i:i+segment_size]],axis=1)
                            aran_segment_details[iter] = aran_concat
                            route_details[iter] = route[:7]+aligned_passes[k]
                            iter += 1
                    else:
                        i +=1

        synth_segments = pd.DataFrame.from_dict(segments,orient='index').transpose()
        synth_segments.to_csv("aligned_data/"+"synthetic_segments"+".csv",index=False)
        aran_segment_details = pd.concat(aran_segment_details)
        aran_segment_details.to_csv("aligned_data/"+"aran_segments"+".csv",index=False)
        myfile = open("aligned_data/routes_details.txt","w")
        myfile.write(str(route_details))
        myfile.close()
        # aran_segment_details = pd.read_csv("aligned_data/"+"aran_segments"+".csv")
        
    return synth_segments, aran_segment_details, route_details

def GM_sample_segmentation(segment_size=150, overlap=0):
    if 1 == 2:
        d = 1
    # if os.path.isfile("aligned_data/"+"aran_segments"+".csv"):
    #     synth_segments = pd.read_csv("aligned_data/"+"synthetic_segments"+".csv")
    #     aran_segment_details = pd.read_csv("aligned_data/"+"aran_segments"+".csv")
    #     route_details = eval(open("aligned_data/routes_details.txt", 'r').read())
    #     print("Loaded already segmented data")              
        
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
                aran_end_idx = (len(aran_location)-200)+aran_end_idx
                
                aran_start = [aran_location['LatitudeFrom'][aran_start_idx],aran_location['LongitudeFrom'][aran_start_idx]]
                aran_end = [aran_location['LatitudeFrom'][aran_end_idx],aran_location['LongitudeTo'][aran_end_idx]]
                GM_start_idx, _ = find_min_gps_vector(aran_start,aligned_gps[['lat','lon']].values)
                GM_end_idx, _ = find_min_gps_vector(aran_end,aligned_gps[['lat','lon']].values)
                
                i = GM_start_idx
                # Get 50m from ARAN -> Find gps signal from p79 -> Get measurements from synthetic data
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
                            i += int(segment_size/3)
                        else:
                            aran_new_start = aran_location[['LatitudeFrom','LongitudeFrom']].iloc[aran_end_idx+1].values
                            GM_start_idx, _ = find_min_gps_vector(aran_new_start,aligned_gps[['lat','lon']].values)
                            i = GM_start_idx
                            # print(i)
                            segments[iter] = acc_seg['acc_z']
                            aran_concat = pd.concat([aran_location[aran_start_idx:aran_end_idx+1],aran_alligator[aran_start_idx:aran_end_idx+1],aran_cracks[aran_start_idx:aran_end_idx+1],aran_potholes[aran_start_idx:aran_end_idx+1]],axis=1)
                            aran_segment_details[iter] = aran_concat
                            route_details[iter] = route[:7]+aligned_passes[k]
                            dists.append(dist_seg.iloc[-1] - dist_seg.iloc[0])
                            iter += 1
                    else:
                        i += int(segment_size/3)

        synth_segments = pd.DataFrame.from_dict(segments,orient='index').transpose()
        # synth_segments.to_csv("aligned_data/"+"synthetic_segments"+".csv",index=False)
        # aran_segment_details = pd.concat(aran_segment_details)
        # aran_segment_details.to_csv("aligned_data/"+"aran_segments"+".csv",index=False)
        # myfile = open("aligned_data/routes_details.txt","w")
        # myfile.write(str(route_details))
        # myfile.close()
        # aran_segment_details = pd.read_csv("aligned_data/"+"aran_segments"+".csv")

    return synth_segments, aran_segment_details, route_details, dists




def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()

# Calling Method 
# plot_grid_search(knn_train.cv_results_, [10,20,40], ['uniform','distance'], 'n-neighbors', 'weights')


