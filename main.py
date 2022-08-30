import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from functions import *
from LiRA_functions import *

from tabulate import tabulate


if __name__ == '__main__':
    
    if sys.argv[1] == 'synth_test':
        synth_acc = synthetic_data()
        routes = []
        for i in range(len(synth_acc)): 
            routes.append(synth_acc[i].axes[0].name)

        synth_segments, aran_segments, route_details = synthetic_segmentation(synth_acc,routes,segment_size=5,overlap=0)


        hdf5file = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
        aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
        aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
        aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
        aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

        DI, allig, cracks, potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
        

        data = synth_segments#.iloc[:,0:100]
        features,feature_names = feature_extraction(data,'synth_data/extracted_features')

        seg_len = 5 
        seg_cap = 4
        DI = []
        alligator = []
        cracks = []
        potholes = []
        for i in tqdm(range(int(np.shape(aran_segments)[0]/seg_len))):
            aran_details = aran_segments.iloc[i*seg_len:i*seg_len+seg_cap+1]
            aran_alligator = aran_details[aran_alligator.columns]
            aran_cracks = aran_details[aran_cracks.columns]
            aran_potholes = aran_details[aran_potholes.columns]
            temp_DI, temp_alligator, temp_cracks, temp_potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
            DI.append(np.max(temp_DI))
            alligator.append(np.max(temp_alligator))
            cracks.append(np.max(temp_cracks))
            potholes.append(np.max(temp_potholes))
        
        
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        model = False

        cv, test_split, cv2 = custom_splits(aran_segments,route_details,save=True)

        scores_RandomForest_DI        = method_RandomForest(features, DI, 'DI', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        scores_RandomForest_potholes  = method_RandomForest(features, potholes, 'potholes', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        scores_RandomForest_cracks    = method_RandomForest(features, cracks, 'cracks', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        scores_RandomForest_alligator = method_RandomForest(features, alligator, 'alligator', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)

        print("Splits - total segments: ",len(cv[0][1])+len(cv[1][1])+len(cv[2][1])+len(cv[3][1])+len(test_split))
        print("Split1: ",len(cv[0][0]),len(cv[0][1]))
        print("Split2: ",len(cv[1][0]),len(cv[1][1]))
        print("Split3: ",len(cv[2][0]),len(cv[2][1]))
        print("Split4: ",len(cv[3][0]),len(cv[3][1]))
        print("Test split (split5): ",len(test_split))

        print('Training metrics')
        print(tabulate([['R2',scores_RandomForest_DI['R2'][1],scores_RandomForest_potholes['R2'][1],scores_RandomForest_cracks['R2'][1],scores_RandomForest_alligator['R2'][1]],
                        ['MSE',scores_RandomForest_DI['MSE'][1],scores_RandomForest_potholes['MSE'][1],scores_RandomForest_cracks['MSE'][1],scores_RandomForest_alligator['MSE'][1]],
                        ['RMSE',scores_RandomForest_DI['RMSE'][1],scores_RandomForest_potholes['RMSE'][1],scores_RandomForest_cracks['RMSE'][1],scores_RandomForest_alligator['RMSE'][1]],
                        ['MAE',scores_RandomForest_DI['MAE'][1],scores_RandomForest_potholes['MAE'][1],scores_RandomForest_cracks['MAE'][1],scores_RandomForest_alligator['MAE'][1]]], 
                headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))
        
        print('Test metrics')
        print(tabulate([['R2',scores_RandomForest_DI['R2'][0],scores_RandomForest_potholes['R2'][0],scores_RandomForest_cracks['R2'][0],scores_RandomForest_alligator['R2'][0]],
                        ['MSE',scores_RandomForest_DI['MSE'][0],scores_RandomForest_potholes['MSE'][0],scores_RandomForest_cracks['MSE'][0],scores_RandomForest_alligator['MSE'][0]],
                        ['RMSE',scores_RandomForest_DI['RMSE'][0],scores_RandomForest_potholes['RMSE'][0],scores_RandomForest_cracks['RMSE'][0],scores_RandomForest_alligator['RMSE'][0]],
                        ['MAE',scores_RandomForest_DI['MAE'][0],scores_RandomForest_potholes['MAE'][0],scores_RandomForest_cracks['MAE'][0],scores_RandomForest_alligator['MAE'][0]]], 
                headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))










    ################################################## TESTS - 45 km/h speed #################################################
    if sys.argv[1] == '45km_test':
        synth_acc = test_synthetic_data()
        routes = []
        for i in range(len(synth_acc)): 
            routes.append(synth_acc[i].axes[0].name)

        synth_segments, aran_segments, route_details, laser_segments = test_segmentation(synth_acc,routes,segment_size=5,overlap=0)

        hdf5file = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
        aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
        aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
        aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
        aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

        DI, allig, cracks, potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
        
        data = synth_segments#.iloc[:,0:100]
        features,feature_names = feature_extraction(data,'synth_data/tests/features_45km')

        seg_len = 5 
        seg_cap = 4
        DI = []
        alligator = []
        cracks = []
        potholes = []
        for i in tqdm(range(int(np.shape(aran_segments)[0]/seg_len))):
            aran_details = aran_segments.iloc[i*seg_len:i*seg_len+seg_cap+1]
            aran_alligator = aran_details[aran_alligator.columns]
            aran_cracks = aran_details[aran_cracks.columns]
            aran_potholes = aran_details[aran_potholes.columns]
            temp_DI, temp_alligator, temp_cracks, temp_potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
            DI.append(np.max(temp_DI))
            alligator.append(np.max(temp_alligator))
            cracks.append(np.max(temp_cracks))
            potholes.append(np.max(temp_potholes))
        
        
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        model = False

        cv, test_split, cv2 = custom_splits(aran_segments,route_details)

        scores_RandomForest_DI        = method_RandomForest(features, DI, 'DI_45km', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        scores_RandomForest_potholes  = method_RandomForest(features, potholes, 'potholes_45km', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        scores_RandomForest_cracks    = method_RandomForest(features, cracks, 'cracks_45km', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        scores_RandomForest_alligator = method_RandomForest(features, alligator, 'alligator_45km', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)

        print("Splits - total segments: ",len(cv[0][1])+len(cv[1][1])+len(cv[2][1])+len(cv[3][1])+len(test_split))
        print("Split1: ",len(cv[0][0]),len(cv[0][1]))
        print("Split2: ",len(cv[1][0]),len(cv[1][1]))
        print("Split3: ",len(cv[2][0]),len(cv[2][1]))
        print("Split4: ",len(cv[3][0]),len(cv[3][1]))
        print("Test split (split5): ",len(test_split))

        print('Training metrics')
        print(tabulate([['R2',scores_RandomForest_DI['R2'][1],scores_RandomForest_potholes['R2'][1],scores_RandomForest_cracks['R2'][1],scores_RandomForest_alligator['R2'][1]],
                        ['MSE',scores_RandomForest_DI['MSE'][1],scores_RandomForest_potholes['MSE'][1],scores_RandomForest_cracks['MSE'][1],scores_RandomForest_alligator['MSE'][1]],
                        ['RMSE',scores_RandomForest_DI['RMSE'][1],scores_RandomForest_potholes['RMSE'][1],scores_RandomForest_cracks['RMSE'][1],scores_RandomForest_alligator['RMSE'][1]],
                        ['MAE',scores_RandomForest_DI['MAE'][1],scores_RandomForest_potholes['MAE'][1],scores_RandomForest_cracks['MAE'][1],scores_RandomForest_alligator['MAE'][1]]], 
                headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))
        
        print('Test metrics')
        print(tabulate([['R2',scores_RandomForest_DI['R2'][0],scores_RandomForest_potholes['R2'][0],scores_RandomForest_cracks['R2'][0],scores_RandomForest_alligator['R2'][0]],
                        ['MSE',scores_RandomForest_DI['MSE'][0],scores_RandomForest_potholes['MSE'][0],scores_RandomForest_cracks['MSE'][0],scores_RandomForest_alligator['MSE'][0]],
                        ['RMSE',scores_RandomForest_DI['RMSE'][0],scores_RandomForest_potholes['RMSE'][0],scores_RandomForest_cracks['RMSE'][0],scores_RandomForest_alligator['RMSE'][0]],
                        ['MAE',scores_RandomForest_DI['MAE'][0],scores_RandomForest_potholes['MAE'][0],scores_RandomForest_cracks['MAE'][0],scores_RandomForest_alligator['MAE'][0]]], 
                headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))



    ################################################## TESTS - (laser5+laser21)/2/1e3 #################################################
    if sys.argv[1] == 'laser_test':
        synth_acc = test_synthetic_data()
        routes = []
        for i in range(len(synth_acc)): 
            routes.append(synth_acc[i].axes[0].name)

        synth_segments, aran_segments, route_details, laser_segments = test_segmentation(synth_acc,routes,segment_size=5,overlap=0)

        hdf5file = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
        aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
        aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
        aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
        aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

        DI, allig, cracks, potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
        
        data = laser_segments#.iloc[:,0:100]
        features,feature_names = feature_extraction(data,'synth_data/tests/features_laser')

        seg_len = 5 
        seg_cap = 4
        DI = []
        alligator = []
        cracks = []
        potholes = []
        for i in tqdm(range(int(np.shape(aran_segments)[0]/seg_len))):
            aran_details = aran_segments.iloc[i*seg_len:i*seg_len+seg_cap+1]
            aran_alligator = aran_details[aran_alligator.columns]
            aran_cracks = aran_details[aran_cracks.columns]
            aran_potholes = aran_details[aran_potholes.columns]
            temp_DI, temp_alligator, temp_cracks, temp_potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
            DI.append(np.max(temp_DI))
            alligator.append(np.max(temp_alligator))
            cracks.append(np.max(temp_cracks))
            potholes.append(np.max(temp_potholes))
        
        
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        model = False

        cv, test_split, cv2 = custom_splits(aran_segments,route_details)

        scores_RandomForest_DI        = method_RandomForest(features, DI, 'DI_laser', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        scores_RandomForest_potholes  = method_RandomForest(features, potholes, 'potholes_laser', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        scores_RandomForest_cracks    = method_RandomForest(features, cracks, 'cracks_laser', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        scores_RandomForest_alligator = method_RandomForest(features, alligator, 'alligator_laser', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)

        
        print("Splits - total segments: ",len(cv[0][1])+len(cv[1][1])+len(cv[2][1])+len(cv[3][1])+len(test_split))
        print("Split1: ",len(cv[0][0]),len(cv[0][1]))
        print("Split2: ",len(cv[1][0]),len(cv[1][1]))
        print("Split3: ",len(cv[2][0]),len(cv[2][1]))
        print("Split4: ",len(cv[3][0]),len(cv[3][1]))
        print("Test split (split5): ",len(test_split))

        print('Training metrics')
        print(tabulate([['R2',scores_RandomForest_DI['R2'][1],scores_RandomForest_potholes['R2'][1],scores_RandomForest_cracks['R2'][1],scores_RandomForest_alligator['R2'][1]],
                        ['MSE',scores_RandomForest_DI['MSE'][1],scores_RandomForest_potholes['MSE'][1],scores_RandomForest_cracks['MSE'][1],scores_RandomForest_alligator['MSE'][1]],
                        ['RMSE',scores_RandomForest_DI['RMSE'][1],scores_RandomForest_potholes['RMSE'][1],scores_RandomForest_cracks['RMSE'][1],scores_RandomForest_alligator['RMSE'][1]],
                        ['MAE',scores_RandomForest_DI['MAE'][1],scores_RandomForest_potholes['MAE'][1],scores_RandomForest_cracks['MAE'][1],scores_RandomForest_alligator['MAE'][1]]], 
                headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))
        
        print('Test metrics')
        print(tabulate([['R2',scores_RandomForest_DI['R2'][0],scores_RandomForest_potholes['R2'][0],scores_RandomForest_cracks['R2'][0],scores_RandomForest_alligator['R2'][0]],
                        ['MSE',scores_RandomForest_DI['MSE'][0],scores_RandomForest_potholes['MSE'][0],scores_RandomForest_cracks['MSE'][0],scores_RandomForest_alligator['MSE'][0]],
                        ['RMSE',scores_RandomForest_DI['RMSE'][0],scores_RandomForest_potholes['RMSE'][0],scores_RandomForest_cracks['RMSE'][0],scores_RandomForest_alligator['RMSE'][0]],
                        ['MAE',scores_RandomForest_DI['MAE'][0],scores_RandomForest_potholes['MAE'][0],scores_RandomForest_cracks['MAE'][0],scores_RandomForest_alligator['MAE'][0]]], 
                headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))

        
    ########################################################### GREEN MOBILITY TEST ##############################################################
    if sys.argv[1] == 'GM_test':
        GM_segments, aran_segments, route_details = GM_segmentation(segment_size=5,overlap=0)

        hdf5file = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
        aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
        aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
        aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
        aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

        DI, allig, cracks, potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
        
        data = GM_segments#.iloc[:,0:100]
        temp = 1000
        maxi = 0
        for i in range(np.shape(data)[1]):
            min = len(data[str(i)].dropna())
            if min < temp:
                temp = min
                min_idx = i
            if min > maxi:
                maxi = min
                max_idx = i
        features,feature_names = feature_extraction(data,'aligned_data/extracted_features',fs=50,min_idx=min_idx)

        seg_len = 5 
        seg_cap = 4
        DI = []
        alligator = []
        cracks = []
        potholes = []
        for i in tqdm(range(int(np.shape(aran_segments)[0]/seg_len))):
            aran_details = aran_segments.iloc[i*seg_len:i*seg_len+seg_cap+1]
            aran_alligator = aran_details[aran_alligator.columns]
            aran_cracks = aran_details[aran_cracks.columns]
            aran_potholes = aran_details[aran_potholes.columns]
            temp_DI, temp_alligator, temp_cracks, temp_potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
            DI.append(np.max(temp_DI))
            alligator.append(np.max(temp_alligator))
            cracks.append(np.max(temp_cracks))
            potholes.append(np.max(temp_potholes))
        
        
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        model = False

        cv, test_split, cv2 = custom_splits(aran_segments,route_details)

        scores_RandomForest_DI        = method_RandomForest(features, DI, 'DI_GM', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        scores_RandomForest_potholes  = method_RandomForest(features, potholes, 'potholes_GM', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        scores_RandomForest_cracks    = method_RandomForest(features, cracks, 'cracks_GM', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        scores_RandomForest_alligator = method_RandomForest(features, alligator, 'alligator_GM', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)

        print("Splits - total segments: ",len(cv[0][1])+len(cv[1][1])+len(cv[2][1])+len(cv[3][1])+len(test_split))
        print("Split1: ",len(cv[0][0]),len(cv[0][1]))
        print("Split2: ",len(cv[1][0]),len(cv[1][1]))
        print("Split3: ",len(cv[2][0]),len(cv[2][1]))
        print("Split4: ",len(cv[3][0]),len(cv[3][1]))
        print("Test split (split5): ",len(test_split))

        print('Training metrics')
        print(tabulate([['R2',scores_RandomForest_DI['R2'][1],scores_RandomForest_potholes['R2'][1],scores_RandomForest_cracks['R2'][1],scores_RandomForest_alligator['R2'][1]],
                        ['MSE',scores_RandomForest_DI['MSE'][1],scores_RandomForest_potholes['MSE'][1],scores_RandomForest_cracks['MSE'][1],scores_RandomForest_alligator['MSE'][1]],
                        ['RMSE',scores_RandomForest_DI['RMSE'][1],scores_RandomForest_potholes['RMSE'][1],scores_RandomForest_cracks['RMSE'][1],scores_RandomForest_alligator['RMSE'][1]],
                        ['MAE',scores_RandomForest_DI['MAE'][1],scores_RandomForest_potholes['MAE'][1],scores_RandomForest_cracks['MAE'][1],scores_RandomForest_alligator['MAE'][1]]], 
                headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))
        
        print('Test metrics')
        print(tabulate([['R2',scores_RandomForest_DI['R2'][0],scores_RandomForest_potholes['R2'][0],scores_RandomForest_cracks['R2'][0],scores_RandomForest_alligator['R2'][0]],
                        ['MSE',scores_RandomForest_DI['MSE'][0],scores_RandomForest_potholes['MSE'][0],scores_RandomForest_cracks['MSE'][0],scores_RandomForest_alligator['MSE'][0]],
                        ['RMSE',scores_RandomForest_DI['RMSE'][0],scores_RandomForest_potholes['RMSE'][0],scores_RandomForest_cracks['RMSE'][0],scores_RandomForest_alligator['RMSE'][0]],
                        ['MAE',scores_RandomForest_DI['MAE'][0],scores_RandomForest_potholes['MAE'][0],scores_RandomForest_cracks['MAE'][0],scores_RandomForest_alligator['MAE'][0]]], 
                headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))
        obj = scores_RandomForest_DI['Gridsearchcv_obj']
        k = 0
        temp = 0
        for train, test in cv:
            for i in range(len(train)):
                temp += np.sum(train[i] == test_split)
            for j in range(len(test)):
                temp += np.sum(test[j] == test_split)   
            train == obj.cv[k][0]
            test == obj.cv[k][1]
            k+=1
        temp