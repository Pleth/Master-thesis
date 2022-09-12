import sys
import numpy as np
import pandas as pd

from functions import *
from LiRA_functions import *

if __name__ == '__main__':
    
    if sys.argv[1] == 'synth_test':
        aran_segments = pd.read_csv('hpc/aran_segments.csv')
        features = pd.read_csv('hpc/extracted_features.csv')
        route_details = eval(open("hpc/routes_details.txt", 'r').read())

        seg_len = 5 
        seg_cap = 4
        DI = []
        alligator = []
        cracks = []
        potholes = []
        for i in tqdm(range(int(np.shape(aran_segments)[0]/seg_len))):
            aran_details = aran_segments.iloc[i*seg_len:i*seg_len+seg_cap+1]
            temp_DI, temp_alligator, temp_cracks, temp_potholes = calc_DI_v2(aran_details)
            DI.append(np.max(temp_DI))
            alligator.append(np.max(temp_alligator))
            cracks.append(np.max(temp_cracks))
            potholes.append(np.max(temp_potholes))
        
        
        gridsearch = 1
        verbose = 3
        n_jobs = -1 # 4
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1

        cv, test_split, cv2, splits = custom_splits(aran_segments,route_details,save=True)
        cv_train, split_test, X_train, X_test = rearange_splits(splits,features)

        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['1']].reset_index(drop=True)
        DI_train = DI.iloc[splits['2']+splits['3']+splits['4']+splits['5']].reset_index(drop=True)

        scores_RandomForest_DI        = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
    
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI.sav')
        print(rf_train.best_estimator_)



    ################################################## TESTS - 45 km/h speed #################################################
    if sys.argv[1] == '45km_test':
        aran_segments = pd.read_csv('hpc/synth/aran_segments.csv')
        features = pd.read_csv('hpc/synth/features_45km.csv')
        route_details = eval(open("hpc/synth/routes_details.txt", 'r').read())

        seg_len = 5 
        seg_cap = 4
        DI = []
        alligator = []
        cracks = []
        potholes = []
        for i in tqdm(range(int(np.shape(aran_segments)[0]/seg_len))):
            aran_details = aran_segments.iloc[i*seg_len:i*seg_len+seg_cap+1]
            temp_DI, temp_alligator, temp_cracks, temp_potholes = calc_DI_v2(aran_details)
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
    
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_45km.sav')
        print(rf_train.best_estimator_)


    ################################################## TESTS - (laser5+laser21)/2/1e3 #################################################
    if sys.argv[1] == 'laser_test':
        aran_segments = pd.read_csv('hpc/synth/aran_segments.csv')
        features = pd.read_csv('hpc/synth/features_laser.csv')
        route_details = eval(open("hpc/synth/routes_details.txt", 'r').read())

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
    
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_laser.sav')
        print(rf_train.best_estimator_)
        

    ########################################################### GREEN MOBILITY TEST ##############################################################
    if sys.argv[1] == 'GM_test':
        aran_segments = pd.read_csv('hpc/aligned/aran_segments.csv')
        features = pd.read_csv('hpc/aligned/features_45km.csv')
        route_details = eval(open("hpc/aligned/routes_details.txt", 'r').read())

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
    
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM.sav')
        print(rf_train.best_estimator_)