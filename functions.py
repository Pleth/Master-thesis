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

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV



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

def feature_extraction(data):
    cfg_file = tsfel.get_features_by_domain()
    if os.path.isfile("synth_data/extracted_features.csv"):
        feature_names = np.transpose(tsfel.time_series_features_extractor(cfg_file,data[str(0)].dropna(),fs=250,verbose=0))
        data = pd.read_csv("synth_data/extracted_features.csv")
    else:
        extracted_features = []
        feature_names = np.transpose(tsfel.time_series_features_extractor(cfg_file,data[str(0)].dropna(),fs=250,verbose=0))
        for i in tqdm(range(np.shape(data)[1])):
            extracted_features.append(np.transpose(tsfel.time_series_features_extractor(cfg_file,data[str(i)].dropna(),fs=250,verbose=0)))
        data = pd.DataFrame(np.concatenate(extracted_features,axis=1))
        data.to_csv("synth_data/extracted_features.csv",index=False)

    return data,feature_names


# def method_SVR(features, y, id, model=False, gridsearch=0, verbose=3,n_jobs=None):
#     X = features.T
#     sc_X = StandardScaler()
#     X = sc_X.fit_transform(X)
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)
#     X_train, X_test, y_train, y_test = X, X, y, y

#     if model != False:
#         loaded_model = joblib.load('models/SVR_best_model'+id+'.sav')
#         # loaded_model.fit(X_train,y_train)
#         y_pred = loaded_model.predict(X_test)

#         r2 = r2_score(y_test,y_pred)
#         MSE = mean_squared_error(y_test,y_pred, squared=True)
#         RMSE = mean_squared_error(y_test,y_pred, squared=False)
#         MAE = mean_absolute_error(y_test,y_pred)
#     else:
#         parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 0.01, 0.1],'C': [1, 10, 20]}]
#                     #{'kernel': ['sigmoid'], 'gamma': [1e-5, 1e-4, 1e-3],'C': [1, 10, 100]},
#                     #{'kernel': ['poly'], 'gamma': [1e-3, 0.01, 0.1],'C': [1, 10, 20]},
#                     #{'kernel': ['linear'], 'gamma': [1e-3, 0.01, 0.1],'C': [1, 10, 20]}]

#         start_time = time.time()
#         if gridsearch == 1:
#             svr_train = GridSearchCV(SVR(epsilon = 0.01), parameters, cv = 5,scoring='neg_mean_squared_error',verbose=verbose,n_jobs=n_jobs)
#             svr_train.fit(X_train,y_train)
#             joblib.dump(svr_train,'models/SVR_best_model'+id+'.sav')
#         else:
#             svr_train = SVR(kernel='rbf',C=1,gamma=0.1,epsilon=0.01)
#             svr_train.fit(X_train,y_train)
#         end_time = time.time()
#         run_time = end_time - start_time
#         print('Run time:',round(run_time/60,2),'mins')
        
#         y_pred = svr_train.predict(X_test)

#         r2 = r2_score(y_test,y_pred)
#         MSE = mean_squared_error(y_test,y_pred, squared=True)
#         RMSE = mean_squared_error(y_test,y_pred, squared=False)
#         MAE = mean_absolute_error(y_test,y_pred)

#     return {"R2":r2, "MSE": MSE, "RMSE": RMSE, "MAE": MAE,"Gridsearchcv_obj": svr_train}

    
# def method_KNN(features, y, id, model=False, gridsearch=0, verbose=3,n_jobs=None):
#     X = features.T
#     sc_X = StandardScaler()
#     X = sc_X.fit_transform(X)
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)
#     X_train, X_test, y_train, y_test = X, X, y, y

#     if model != False:
#         loaded_model = joblib.load('models/KNN_best_model'+id+'.sav')
#         # loaded_model.fit(X_train,y_train)
#         y_pred = loaded_model.predict(X_test)

#         r2 = r2_score(y_test,y_pred)
#         MSE = mean_squared_error(y_test,y_pred, squared=True)
#         RMSE = mean_squared_error(y_test,y_pred, squared=False)
#         MAE = mean_absolute_error(y_test,y_pred)
#     else:
#         parameters = [{'weights':['uniform','distance'],'algorithm': ['ball_tree','kd_tree','brute'],
#                        'n_neighbors': [1,2,5,10,20,40,50,60,100]}]
#         parameters = [{'weights':['uniform','distance'],'algorithm': ['ball_tree'],
#                        'n_neighbors': [10,20,40]}]
                    
#         start_time = time.time()
#         if gridsearch == 1:
#             knn_train = GridSearchCV(KNeighborsRegressor(), parameters, cv = 5,scoring='neg_mean_squared_error',verbose=verbose,n_jobs=n_jobs)
#             knn_train.fit(X_train,y_train)
#             joblib.dump(knn_train,'models/KNN_best_model'+id+'.sav')
#         else:
#             knn_train = KNeighborsRegressor(algorithm='ball_tree', n_neighbors=20, weights='distance')
#             knn_train.fit(X_train,y_train)
#         end_time = time.time()
#         run_time = end_time - start_time
#         print('Run time:',round(run_time/60,2),'mins')
        
#         y_pred = knn_train.predict(X_test)

#         r2 = r2_score(y_test,y_pred)
#         MSE = mean_squared_error(y_test,y_pred, squared=True)
#         RMSE = mean_squared_error(y_test,y_pred, squared=False)
#         MAE = mean_absolute_error(y_test,y_pred)

#     return {"R2":r2, "MSE": MSE, "RMSE": RMSE, "MAE": MAE,"Gridsearchcv_obj": knn_train}


def method_DT(features, y, id, model=False, gridsearch=0, verbose=3,n_jobs=None):
    X = features.T
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)
    # X_train, X_test, y_train, y_test = X, X, y, y

    if model != False:
        loaded_model = joblib.load('models/DT_best_model'+id+'.sav')
        # loaded_model.fit(X_train,y_train)
        y_pred = loaded_model.predict(X_test)

        r2 = r2_score(y_test,y_pred)
        MSE = mean_squared_error(y_test,y_pred, squared=True)
        RMSE = mean_squared_error(y_test,y_pred, squared=False)
        MAE = mean_absolute_error(y_test,y_pred)
    else:
        parameters={"criterion": ["squared_error"],
                    "splitter":["best","random"],
                    "max_depth" : [1,3,5,7,9,11,12,None],
                    "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
                    "min_weight_fraction_leaf":[0.0,0.1,0.2,0.3,0.4,0.5],
                    "max_features":["log2","sqrt",None],
                    "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] }

        parameters={"criterion": ["friedman_mse","squared_error","absolute_error"],
                    "splitter":["random","best"],
                    "max_depth" : [1,3,5,8,12,20,50,100],
                    "min_samples_leaf":[2,4,6,8,10],
                    "min_weight_fraction_leaf":[0.0,0.2,0.4],
                    "max_features":['sqrt','log2'],
                    "max_leaf_nodes":[10,20,30,40,50,60,70,80,90] }
        
        start_time = time.time()
        if gridsearch == 1:
            dt_train = GridSearchCV(DecisionTreeRegressor(), parameters, cv = 5,scoring='neg_mean_squared_error',verbose=verbose,n_jobs=n_jobs)
            dt_train.fit(X_train,y_train)
            joblib.dump(dt_train,'models/DT_best_model'+id+'.sav')
        else:
            dt_train = DecisionTreeRegressor(criterion="friedman_mse",max_depth=None,min_weight_fraction_leaf=0.0,max_features=None,splitter='random',min_samples_leaf=5)
            dt_train.fit(X_train,y_train)
        end_time = time.time()
        run_time = end_time - start_time
        print('Run time:',round(run_time/60,2),'mins')
        
        y_pred = dt_train.predict(X_test)

        r2 = r2_score(y_test,y_pred)
        MSE = mean_squared_error(y_test,y_pred, squared=True)
        RMSE = mean_squared_error(y_test,y_pred, squared=False)
        MAE = mean_absolute_error(y_test,y_pred)

    return {"R2":r2, "MSE": MSE, "RMSE": RMSE, "MAE": MAE,"Gridsearchcv_obj": dt_train}







def custom_splits(aran_segments,route_details):
    splits = []
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

    cph1_len = (len(cph1_hh) + len(cph1_vh))/5

    counter = 0
    chain_start = 645
    for i in range(int(cph1_len)):
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

    split1 = list(set(list((np.array(temp_cph1_hh)/5).astype(int)))) + list(set(list((np.array(temp_cph1_vh)/5).astype(int))))

    counter = 0
    chain_end = chain_start
    chain_start = chain_start + 50
    for i in range(int(cph1_len)):
        counter1 = np.sum(np.sum((chain_end < aran_segments['BeginChainage'].iloc[cph1_hh]) & (aran_segments['EndChainage'].iloc[cph1_hh] < chain_start)))
        counter2 = np.sum(np.sum((chain_end < aran_segments['EndChainage'].iloc[cph1_vh]) & (aran_segments['BeginChainage'].iloc[cph1_vh] < chain_start)))
        counter = (counter1 + counter2)/5
        if (counter >= cph1_len/3):
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

    split2 = list(set(list((np.array(temp_cph1_hh)/5).astype(int)))) + list(set(list((np.array(temp_cph1_vh)/5).astype(int))))

    counter = 0
    chain_end = chain_start
    chain_start = chain_start + 50
    for i in range(int(cph1_len)):
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

    split3 = list(set(list((np.array(temp_cph1_hh)/5).astype(int)))) + list(set(list((np.array(temp_cph1_vh)/5).astype(int))))


    cph6_len = (len(cph6_hh) + len(cph6_vh))/5

    counter = 0
    chain_start = 0
    for i in range(int(cph6_len)):
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

    split4 = list(set(list((np.array(temp_cph6_hh)/5).astype(int)))) + list(set(list((np.array(temp_cph6_vh)/5).astype(int))))
    
    counter = 0
    chain_end = chain_start
    for i in range(int(cph6_len)):
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

    split5 = list(set(list((np.array(temp_cph6_hh)/5).astype(int)))) + list(set(list((np.array(temp_cph6_vh)/5).astype(int))))
    
    splits = set().union(split1+split2+split3+split4+split5)

    return splits








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