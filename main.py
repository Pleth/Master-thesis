import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
# from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  

from functions import *
from LiRA_functions import *

from tabulate import tabulate
from scipy import signal


if __name__ == '__main__':
    
    if sys.argv[1] == 'rm_aligned':
        os.remove('aligned_data/aran_segments.csv')
        os.remove('aligned_data/routes_details.txt')
        os.remove('aligned_data/synthetic_segments.csv')
        os.remove('aligned_data/features.csv')

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
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1

        # cv, test_split, cv2, splits = custom_splits(aran_segments,route_details,save=True)
        # cv_train, split_test, X_train, X_test = rearange_splits(splits,features)

        # DI = pd.DataFrame(DI)
        # DI_test = DI.iloc[splits['1']].reset_index(drop=True)
        # DI_train = DI.iloc[splits['2']+splits['3']+splits['4']+splits['5']].reset_index(drop=True)
        # cv_in = [cv_train,split_test]

        X_train = features.iloc[:,540:].reset_index(drop=True)
        DI_train = pd.DataFrame(DI[540:])
        X_test = features.iloc[:,:540]
        DI_test = pd.DataFrame(DI[:540])

        cv_in = [4,10]
        
        scores_RandomForest_DI        = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI', model=model, gridsearch=gridsearch, cv_in=cv_in, verbose=verbose,n_jobs=n_jobs)
        # scores_RandomForest_cracks    = method_RandomForest(features, cracks, 'cracks', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        # scores_RandomForest_potholes  = method_RandomForest(features, potholes, 'potholes', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        # scores_RandomForest_alligator = method_RandomForest(features, alligator, 'alligator', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)

        # print("Splits - total segments: ",len(cv[0][1])+len(cv[1][1])+len(cv[2][1])+len(cv[3][1])+len(test_split))
        # print("Split1: ",len(cv[0][0]),len(cv[0][1]))
        # print("Split2: ",len(cv[1][0]),len(cv[1][1]))
        # print("Split3: ",len(cv[2][0]),len(cv[2][1]))
        # print("Split4: ",len(cv[3][0]),len(cv[3][1]))
        # print("Test split (split5): ",len(test_split))

        # print('Training metrics')
        # print(tabulate([['R2',scores_RandomForest_DI['R2'][1],scores_RandomForest_potholes['R2'][1],scores_RandomForest_cracks['R2'][1],scores_RandomForest_alligator['R2'][1]],
        #                 ['MSE',scores_RandomForest_DI['MSE'][1],scores_RandomForest_potholes['MSE'][1],scores_RandomForest_cracks['MSE'][1],scores_RandomForest_alligator['MSE'][1]],
        #                 ['RMSE',scores_RandomForest_DI['RMSE'][1],scores_RandomForest_potholes['RMSE'][1],scores_RandomForest_cracks['RMSE'][1],scores_RandomForest_alligator['RMSE'][1]],
        #                 ['MAE',scores_RandomForest_DI['MAE'][1],scores_RandomForest_potholes['MAE'][1],scores_RandomForest_cracks['MAE'][1],scores_RandomForest_alligator['MAE'][1]]], 
        #         headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))
        
        # print('Test metrics')
        # print(tabulate([['R2',scores_RandomForest_DI['R2'][0],scores_RandomForest_potholes['R2'][0],scores_RandomForest_cracks['R2'][0],scores_RandomForest_alligator['R2'][0]],
        #                 ['MSE',scores_RandomForest_DI['MSE'][0],scores_RandomForest_potholes['MSE'][0],scores_RandomForest_cracks['MSE'][0],scores_RandomForest_alligator['MSE'][0]],
        #                 ['RMSE',scores_RandomForest_DI['RMSE'][0],scores_RandomForest_potholes['RMSE'][0],scores_RandomForest_cracks['RMSE'][0],scores_RandomForest_alligator['RMSE'][0]],
        #                 ['MAE',scores_RandomForest_DI['MAE'][0],scores_RandomForest_potholes['MAE'][0],scores_RandomForest_cracks['MAE'][0],scores_RandomForest_alligator['MAE'][0]]], 
        #         headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI.sav')
        print(rf_train.best_estimator_)



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
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1

        cv, test_split, cv2, splits = custom_splits(aran_segments,route_details)
        cv_train, split_test, X_train, X_test = rearange_splits(splits,features)

        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['1']].reset_index(drop=True)
        DI_train = DI.iloc[splits['2']+splits['3']+splits['4']+splits['5']].reset_index(drop=True)

        scores_RandomForest_DI        = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_45km', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        # scores_RandomForest_potholes  = method_RandomForest(features, potholes, 'potholes_45km', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        # scores_RandomForest_cracks    = method_RandomForest(features, cracks, 'cracks_45km', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        # scores_RandomForest_alligator = method_RandomForest(features, alligator, 'alligator_45km', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)

        # print("Splits - total segments: ",len(cv[0][1])+len(cv[1][1])+len(cv[2][1])+len(cv[3][1])+len(test_split))
        # print("Split1: ",len(cv[0][0]),len(cv[0][1]))
        # print("Split2: ",len(cv[1][0]),len(cv[1][1]))
        # print("Split3: ",len(cv[2][0]),len(cv[2][1]))
        # print("Split4: ",len(cv[3][0]),len(cv[3][1]))
        # print("Test split (split5): ",len(test_split))

        # print('Training metrics')
        # print(tabulate([['R2',scores_RandomForest_DI['R2'][1],scores_RandomForest_potholes['R2'][1],scores_RandomForest_cracks['R2'][1],scores_RandomForest_alligator['R2'][1]],
        #                 ['MSE',scores_RandomForest_DI['MSE'][1],scores_RandomForest_potholes['MSE'][1],scores_RandomForest_cracks['MSE'][1],scores_RandomForest_alligator['MSE'][1]],
        #                 ['RMSE',scores_RandomForest_DI['RMSE'][1],scores_RandomForest_potholes['RMSE'][1],scores_RandomForest_cracks['RMSE'][1],scores_RandomForest_alligator['RMSE'][1]],
        #                 ['MAE',scores_RandomForest_DI['MAE'][1],scores_RandomForest_potholes['MAE'][1],scores_RandomForest_cracks['MAE'][1],scores_RandomForest_alligator['MAE'][1]]], 
        #         headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))
        
        # print('Test metrics')
        # print(tabulate([['R2',scores_RandomForest_DI['R2'][0],scores_RandomForest_potholes['R2'][0],scores_RandomForest_cracks['R2'][0],scores_RandomForest_alligator['R2'][0]],
        #                 ['MSE',scores_RandomForest_DI['MSE'][0],scores_RandomForest_potholes['MSE'][0],scores_RandomForest_cracks['MSE'][0],scores_RandomForest_alligator['MSE'][0]],
        #                 ['RMSE',scores_RandomForest_DI['RMSE'][0],scores_RandomForest_potholes['RMSE'][0],scores_RandomForest_cracks['RMSE'][0],scores_RandomForest_alligator['RMSE'][0]],
        #                 ['MAE',scores_RandomForest_DI['MAE'][0],scores_RandomForest_potholes['MAE'][0],scores_RandomForest_cracks['MAE'][0],scores_RandomForest_alligator['MAE'][0]]], 
        #         headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_45km.sav')
        print(rf_train.best_estimator_)


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
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1

        cv, test_split, cv2, splits = custom_splits(aran_segments,route_details)
        cv_train, split_test, X_train, X_test = rearange_splits(splits,features)

        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['1']].reset_index(drop=True)
        DI_train = DI.iloc[splits['2']+splits['3']+splits['4']+splits['5']].reset_index(drop=True)

        scores_RandomForest_DI        = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_laser', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        # scores_RandomForest_potholes  = method_RandomForest(features, potholes, 'potholes_laser', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        # scores_RandomForest_cracks    = method_RandomForest(features, cracks, 'cracks_laser', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        # scores_RandomForest_alligator = method_RandomForest(features, alligator, 'alligator_laser', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)

        
        # print("Splits - total segments: ",len(cv[0][1])+len(cv[1][1])+len(cv[2][1])+len(cv[3][1])+len(test_split))
        # print("Split1: ",len(cv[0][0]),len(cv[0][1]))
        # print("Split2: ",len(cv[1][0]),len(cv[1][1]))
        # print("Split3: ",len(cv[2][0]),len(cv[2][1]))
        # print("Split4: ",len(cv[3][0]),len(cv[3][1]))
        # print("Test split (split5): ",len(test_split))

        # print('Training metrics')
        # print(tabulate([['R2',scores_RandomForest_DI['R2'][1],scores_RandomForest_potholes['R2'][1],scores_RandomForest_cracks['R2'][1],scores_RandomForest_alligator['R2'][1]],
        #                 ['MSE',scores_RandomForest_DI['MSE'][1],scores_RandomForest_potholes['MSE'][1],scores_RandomForest_cracks['MSE'][1],scores_RandomForest_alligator['MSE'][1]],
        #                 ['RMSE',scores_RandomForest_DI['RMSE'][1],scores_RandomForest_potholes['RMSE'][1],scores_RandomForest_cracks['RMSE'][1],scores_RandomForest_alligator['RMSE'][1]],
        #                 ['MAE',scores_RandomForest_DI['MAE'][1],scores_RandomForest_potholes['MAE'][1],scores_RandomForest_cracks['MAE'][1],scores_RandomForest_alligator['MAE'][1]]], 
        #         headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))
        
        # print('Test metrics')
        # print(tabulate([['R2',scores_RandomForest_DI['R2'][0],scores_RandomForest_potholes['R2'][0],scores_RandomForest_cracks['R2'][0],scores_RandomForest_alligator['R2'][0]],
        #                 ['MSE',scores_RandomForest_DI['MSE'][0],scores_RandomForest_potholes['MSE'][0],scores_RandomForest_cracks['MSE'][0],scores_RandomForest_alligator['MSE'][0]],
        #                 ['RMSE',scores_RandomForest_DI['RMSE'][0],scores_RandomForest_potholes['RMSE'][0],scores_RandomForest_cracks['RMSE'][0],scores_RandomForest_alligator['RMSE'][0]],
        #                 ['MAE',scores_RandomForest_DI['MAE'][0],scores_RandomForest_potholes['MAE'][0],scores_RandomForest_cracks['MAE'][0],scores_RandomForest_alligator['MAE'][0]]], 
        #         headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))

        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_laser.sav')
        print(rf_train.best_estimator_)
        

    ########################################################### GREEN MOBILITY TEST ##############################################################
    if sys.argv[1] == 'GM_test':
        GM_segments, aran_segments, route_details = GM_segmentation(segment_size=5,overlap=0)
        
        lens = []
        for i in range(np.shape(GM_segments)[1]):
            lens.append(len(GM_segments[str(i)].dropna()))
        
        new_GM_seg = {}
        for i in range(np.shape(GM_segments)[1]):
            new_GM_seg[i] = signal.resample(GM_segments[str(i)].dropna(),int(np.median(lens)))
        GM_segments = pd.DataFrame.from_dict(new_GM_seg,orient='index').transpose()
        
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
            min = len(data[i].dropna())
            if min < temp:
                temp = min
                min_idx = i
            if min > maxi:
                maxi = min
                max_idx = i
        features,feature_names = feature_extraction2(data,'aligned_data/test',fs=50,min_idx=min_idx)

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
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1

        cv, test_split, cv2, splits = custom_splits(aran_segments,route_details)
        cv_train, split_test, X_train, X_test = rearange_splits(splits,features)

        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['1']].reset_index(drop=True)
        DI_train = DI.iloc[splits['2']+splits['3']+splits['4']+splits['5']].reset_index(drop=True)

        scores_RandomForest_DI        = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_GM', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        # scores_RandomForest_potholes  = method_RandomForest(features, potholes, 'potholes_GM', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        # scores_RandomForest_cracks    = method_RandomForest(features, cracks, 'cracks_GM', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)
        # scores_RandomForest_alligator = method_RandomForest(features, alligator, 'alligator_GM', model=model, gridsearch=gridsearch, cv_in=[cv,test_split], verbose=verbose,n_jobs=n_jobs)

        # print("Splits - total segments: ",len(cv[0][1])+len(cv[1][1])+len(cv[2][1])+len(cv[3][1])+len(test_split))
        # print("Split1: ",len(cv[0][0]),len(cv[0][1]))
        # print("Split2: ",len(cv[1][0]),len(cv[1][1]))
        # print("Split3: ",len(cv[2][0]),len(cv[2][1]))
        # print("Split4: ",len(cv[3][0]),len(cv[3][1]))
        # print("Test split (split5): ",len(test_split))

        # print('Training metrics')
        # print(tabulate([['R2',scores_RandomForest_DI['R2'][1],scores_RandomForest_potholes['R2'][1],scores_RandomForest_cracks['R2'][1],scores_RandomForest_alligator['R2'][1]],
        #                 ['MSE',scores_RandomForest_DI['MSE'][1],scores_RandomForest_potholes['MSE'][1],scores_RandomForest_cracks['MSE'][1],scores_RandomForest_alligator['MSE'][1]],
        #                 ['RMSE',scores_RandomForest_DI['RMSE'][1],scores_RandomForest_potholes['RMSE'][1],scores_RandomForest_cracks['RMSE'][1],scores_RandomForest_alligator['RMSE'][1]],
        #                 ['MAE',scores_RandomForest_DI['MAE'][1],scores_RandomForest_potholes['MAE'][1],scores_RandomForest_cracks['MAE'][1],scores_RandomForest_alligator['MAE'][1]]], 
        #         headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))
        
        # print('Test metrics')
        # print(tabulate([['R2',scores_RandomForest_DI['R2'][0],scores_RandomForest_potholes['R2'][0],scores_RandomForest_cracks['R2'][0],scores_RandomForest_alligator['R2'][0]],
        #                 ['MSE',scores_RandomForest_DI['MSE'][0],scores_RandomForest_potholes['MSE'][0],scores_RandomForest_cracks['MSE'][0],scores_RandomForest_alligator['MSE'][0]],
        #                 ['RMSE',scores_RandomForest_DI['RMSE'][0],scores_RandomForest_potholes['RMSE'][0],scores_RandomForest_cracks['RMSE'][0],scores_RandomForest_alligator['RMSE'][0]],
        #                 ['MAE',scores_RandomForest_DI['MAE'][0],scores_RandomForest_potholes['MAE'][0],scores_RandomForest_cracks['MAE'][0],scores_RandomForest_alligator['MAE'][0]]], 
        #         headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM.sav')
        print(rf_train.best_estimator_)
        # max_dep = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # max_feat = ['log2', 'sqrt', None]
        # plot_grid_search(rf_train.cv_results_, max_dep, max_feat, 'Max_depth', 'Max_features')



    if sys.argv[1] == 'copeium':
        GM_segments, aran_segments, route_details = GM_segmentation(segment_size=5,overlap=0)

        # lens = []
        # for i in range(np.shape(GM_segments)[1]):
        #     lens.append(len(GM_segments[str(i)].dropna()))
        
        # new_GM_seg = {}
        # for i in range(np.shape(GM_segments)[1]):
        #     new_GM_seg[i] = signal.resample(GM_segments[str(i)].dropna(),int(np.median(lens)))
        # GM_segments = pd.DataFrame.from_dict(new_GM_seg,orient='index').transpose()

        # idxs = np.array(aran_segments[7000:][aran_segments[7000:]['BeginChainage'] < 9200].index/5).astype(int)
        
        # aran_segments.drop(aran_segments[7000:][aran_segments[7000:]['BeginChainage'] < 9200].index, inplace=True)
        # aran_segments = aran_segments.reset_index(drop=True)
        
        # GM_segments.drop(GM_segments.columns[idxs],axis=1,inplace=True)
        # GM_segments = GM_segments.reset_index(drop=True)

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

        
        features,feature_names = feature_extraction(data,'aligned_data/test2',fs=50,min_idx=min_idx)

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
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1

        # cv, test_split, cv2, splits = custom_splits(aran_segments,route_details)
        # cv_train, split_test, X_train, X_test = rearange_splits(splits,features)

        # DI = pd.DataFrame(DI)
        # DI_test = DI.iloc[splits['1']].reset_index(drop=True)
        # DI_train = DI.iloc[splits['2']+splits['3']+splits['4']+splits['5']].reset_index(drop=True)

        X_train = features.iloc[:,540:].reset_index(drop=True)
        DI_train = pd.DataFrame(DI[540:])
        X_test = features.iloc[:,:540]
        DI_test = pd.DataFrame(DI[:540])

        scores_RandomForest_DI        = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_GM_cope', model=model, gridsearch=gridsearch, cv_in=[4,10], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM.sav')
        print(rf_train.best_estimator_)
        # max_dep = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # max_feat = ['log2', 'sqrt', None]
        # plot_grid_search(rf_train.cv_results_, max_dep, max_feat, 'Max_depth', 'Max_features')


        synth_acc = test_synthetic_data()
        routes = []
        for i in range(len(synth_acc)): 
            routes.append(synth_acc[i].axes[0].name)

        synth_segments, aran_segments, route_details, laser_segments = test_segmentation(synth_acc,routes,segment_size=5,overlap=0)


        seg_len = 5 
        seg_cap = 4
        chain_max = []
        chain_min = []
        for i in tqdm(range(int(np.shape(aran_segments)[0]/seg_len))):
            aran_details = aran_segments.iloc[i*seg_len:i*seg_len+seg_cap+1]
            chain_max.append(np.max([np.max(aran_details['BeginChainage']),np.max(aran_details['EndChainage'])]))
            chain_min.append(np.min([np.min(aran_details['BeginChainage']),np.min(aran_details['EndChainage'])]))

        maxer = []
        miner = []
        for i in range(len(route_details)):
            if route_details[i][:7] == 'CPH6_VH':
                maxer.append(chain_max[i])
            if route_details[i][:7] == 'CPH6_HH':
                miner.append(chain_min[i])

        # heading min = 9200


    if sys.argv[1] == 'copeiumv2':
        GM_segments, aran_segment_details, route_details, dists = GM_sample_segmentation(segment_size=150)

        hdf5file = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
        aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
        aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
        aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
        aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

        DI, allig, cracks, potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
        
        data = GM_segments#.iloc[:,0:100]
        data.columns = data.columns.map(str)
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
        
        features,feature_names = feature_extraction(data,'aligned_data/extracted_features_sample',fs=50,min_idx=min_idx)

        DI = []
        alligator = []
        cracks = []
        potholes = []
        aran_dists = []
        for i in tqdm(range(int(len(aran_segment_details)))):
            aran_details = aran_segment_details[i]
            aran_alligator = aran_details[aran_alligator.columns]
            aran_cracks = aran_details[aran_cracks.columns]
            aran_potholes = aran_details[aran_potholes.columns]
            temp_DI, temp_alligator, temp_cracks, temp_potholes = calc_DI(aran_alligator,aran_cracks,aran_potholes)
            DI.append(np.max(temp_DI))
            alligator.append(np.max(temp_alligator))
            cracks.append(np.max(temp_cracks))
            potholes.append(np.max(temp_potholes))
            aran_dists.append(abs(aran_details['EndChainage'].iloc[-1]-aran_details['BeginChainage'].iloc[0]))
        
        
        gridsearch = 1
        verbose = 3
        n_jobs = -1 # 4
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1

        X_train = features.iloc[:,540:].reset_index(drop=True)
        DI_train = pd.DataFrame(DI[540:])
        X_test = features.iloc[:,:540]
        DI_test = pd.DataFrame(DI[:540])

        scores_RandomForest_DI        = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_GM_sample', model=model, gridsearch=gridsearch, cv_in=[4,10], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_sample.sav')
        print(rf_train.best_estimator_)

        plt.plot(rf_train.cv_results_['mean_train_score'])
        plt.plot(rf_train.cv_results_['mean_test_score'])
        plt.show()
        np.argpartition(rf_train.cv_results_['mean_test_score'],-5)[-5:]



if sys.argv[1] == 'GM_split_test':
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
            min = len(data[i].dropna())
            if min < temp:
                temp = min
                min_idx = i
            if min > maxi:
                maxi = min
                max_idx = i
        features,feature_names = feature_extraction2(data,'aligned_data/features',fs=50,min_idx=min_idx)

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
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1
        #### split 1
        # print('---------SPLIT 1--------')
        # cut = [10000,19000,9000]
        # cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split1')
        
        # DI = pd.DataFrame(DI)
        # DI_test = DI.iloc[splits['1']].reset_index(drop=True)
        # DI_train = DI.iloc[splits['2']+splits['3']+splits['4']+splits['5']].reset_index(drop=True)

        # scores_RandomForest_DI = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_GM_split1', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        # print(scores_RandomForest_DI['R2'][1])
        # print(scores_RandomForest_DI['R2'][0])

        # rf_train = joblib.load('models/RandomForest_best_model_DI_GM_split1.sav')
        # print(rf_train.best_estimator_)

        print('---------SPLIT 2--------')
        cut = [10000,19000,9000]
        cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split2')
        
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['2']].reset_index(drop=True)
        DI_train = DI.iloc[splits['1']+splits['3']+splits['4']+splits['5']].reset_index(drop=True)

        scores_RandomForest_DI = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_GM_split2', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_split2.sav')
        print(rf_train.best_estimator_)

        print('---------SPLIT 3--------')
        cut = [10000,19000,9000]
        cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split3')
        
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['3']].reset_index(drop=True)
        DI_train = DI.iloc[splits['1']+splits['2']+splits['4']+splits['5']].reset_index(drop=True)

        scores_RandomForest_DI = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_GM_split3', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_split3.sav')
        print(rf_train.best_estimator_)

        print('---------SPLIT 4--------')
        cut = [10000,19000,9000]
        cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split4')
        
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['4']].reset_index(drop=True)
        DI_train = DI.iloc[splits['1']+splits['2']+splits['3']+splits['5']].reset_index(drop=True)

        scores_RandomForest_DI = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_GM_split4', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_split4.sav')
        print(rf_train.best_estimator_)

        print('---------SPLIT 5--------')
        cut = [10000,19000,9000]
        cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split5')
        
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['5']].reset_index(drop=True)
        DI_train = DI.iloc[splits['1']+splits['2']+splits['3']+splits['4']].reset_index(drop=True)

        scores_RandomForest_DI = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_GM_split5', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_split5.sav')
        print(rf_train.best_estimator_)
