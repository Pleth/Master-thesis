import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from functions import *
from DL_functions import *
from LiRA_functions import *

# from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) 

from tabulate import tabulate

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

        features,feature_names = feature_extraction(synth_segments,'aligned_data/features',fs=250)

        DI, cracks, aliigator, potholes = calc_target(aran_segments)
        
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1

        X_train = features.iloc[:,540:].reset_index(drop=True)
        DI_train = pd.DataFrame(DI[540:])
        X_test = features.iloc[:,:540]
        DI_test = pd.DataFrame(DI[:540])

        cv_in = [4,10]
        
        scores_RandomForest_DI = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI', model=model, gridsearch=gridsearch, cv_in=cv_in, verbose=verbose,n_jobs=n_jobs)
        
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
        
        features,feature_names = feature_extraction(synth_segments,'aligned_data/features',fs=250)

        DI, cracks, aliigator, potholes = calc_target(aran_segments)
        
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1

        cut = [10500,18000,15000]

        #### split 1
        cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split2')
        
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['2']].reset_index(drop=True)
        DI_train = DI.iloc[splits['1']+splits['3']+splits['4']+splits['5']].reset_index(drop=True)

        cv_in = [4,split_test]
        scores_RandomForest_DI = method_RandomForest(X_train, X_test, DI_train, DI_test, 'DI_45km', model=model, gridsearch=gridsearch, cv_in=cv_in, verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_45km.sav')
        print(rf_train.best_estimator_)

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

        features,feature_names = feature_extraction(synth_segments,'aligned_data/features',fs=250)

        DI, cracks, aliigator, potholes = calc_target(aran_segments)
        
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
        GM_segments, aran_segments, route_details = GM_segmentation(segment_size=5,overlap=0)
        
        features,feature_names = feature_extraction(GM_segments,'aligned_data/features',fs=50)

        DI, cracks, aliigator, potholes = calc_target(aran_segments)
        
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1

        cut = [10000,19000,14000]

        cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split1')
        
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['1']].reset_index(drop=True)
        DI_train = DI.iloc[splits['2']+splits['3']+splits['4']+splits['5']].reset_index(drop=True)
        scores_RandomForest_DI        = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_GM', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM.sav')
        print(rf_train.best_estimator_)
       


    if sys.argv[1] == 'sample_test':
        GM_segments, aran_segments, route_details, dists = GM_sample_segmentation(segment_size=150)
        
        features,feature_names = feature_extraction(GM_segments,'aligned_data/features_sample',fs=50)

        DI, cracks, aliigator, potholes = calc_target(aran_segments)
       
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1
        
        cut = [10000,19000,12000]

        cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split1')
        
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['1']].reset_index(drop=True)
        DI_train = DI.iloc[splits['2']+splits['3']+splits['4']+splits['5']].reset_index(drop=True)
        
        cv_in = [cv_train,split_test]
        scores_RandomForest_DI        = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_GM_sample', model=model, gridsearch=gridsearch, cv_in=cv_in, verbose=verbose,n_jobs=n_jobs)
        
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

        features,feature_names = feature_extraction(GM_segments,'aligned_data/features',fs=50)

        DI, cracks, aliigator, potholes = calc_target(aran_segments)
        
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1
        
        cut = [10000,19000,14000]

        #### split 1
        print('---------SPLIT 1--------')
        cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split1')
        
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['1']].reset_index(drop=True)
        DI_train = DI.iloc[splits['2']+splits['3']+splits['4']+splits['5']].reset_index(drop=True)

        scores_RandomForest_DI = method_RandomForest(X_train, X_test, DI_train, DI_test, 'DI_GM_split1', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_split1.sav')
        print(rf_train.best_estimator_)

        print('---------SPLIT 2--------')
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
        cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split5')
        
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['5']].reset_index(drop=True)
        DI_train = DI.iloc[splits['1']+splits['2']+splits['3']+splits['4']].reset_index(drop=True)

        scores_RandomForest_DI = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_GM_split5', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_split5.sav')
        print(rf_train.best_estimator_)

    if sys.argv[1] == 'GM_route_split':
        GM_segments, aran_segments, route_details = GM_segmentation(segment_size=5,overlap=0)
        
        features,feature_names = feature_extraction(GM_segments,'aligned_data/features',fs=50)

        DI, cracks, aliigator, potholes = calc_target(aran_segments)
        
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1
        
        print('---------cph1_vh--------')
        cv_train, split_test, X_train, X_test, splits = route_splits(features,route_details,'cph1_vh')
        
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['cph1_vh']].reset_index(drop=True)
        DI_train = DI.iloc[splits['cph1_hh']+splits['cph6_vh']+splits['cph6_hh']].reset_index(drop=True)

        scores_RandomForest_DI = method_RandomForest(X_train, X_test, DI_train, DI_test, 'DI_GM_cph1_vh', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_cph1_vh.sav')
        print(rf_train.best_estimator_)

        print('---------cph1_hh--------')
        cv_train, split_test, X_train, X_test, splits = route_splits(features,route_details,'cph1_hh')
        
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['cph1_hh']].reset_index(drop=True)
        DI_train = DI.iloc[splits['cph1_vh']+splits['cph6_vh']+splits['cph6_hh']].reset_index(drop=True)

        scores_RandomForest_DI = method_RandomForest(X_train, X_test, DI_train, DI_test, 'DI_GM_cph1_hh', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_cph1_hh.sav')
        print(rf_train.best_estimator_)

        print('---------cph6_vh--------')
        cv_train, split_test, X_train, X_test, splits = route_splits(features,route_details,'cph6_vh')
        
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['cph6_vh']].reset_index(drop=True)
        DI_train = DI.iloc[splits['cph1_vh']+splits['cph1_hh']+splits['cph6_hh']].reset_index(drop=True)

        scores_RandomForest_DI = method_RandomForest(X_train, X_test, DI_train, DI_test, 'DI_GM_cph6_vh', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_cph6_vh.sav')
        print(rf_train.best_estimator_)

        print('---------cph6_hh--------')
        cv_train, split_test, X_train, X_test, splits = route_splits(features,route_details,'cph6_hh')
        
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['cph6_hh']].reset_index(drop=True)
        DI_train = DI.iloc[splits['cph1_vh']+splits['cph1_hh']+splits['cph6_vh']].reset_index(drop=True)

        scores_RandomForest_DI = method_RandomForest(X_train, X_test, DI_train, DI_test, 'DI_GM_cph6_hh', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_cph6_hh.sav')
        print(rf_train.best_estimator_)

    
    if sys.argv[1] == 'GM_shuffle_test':
        GM_segments, aran_segments, route_details = GM_segmentation(segment_size=5,overlap=0)

        features,feature_names = feature_extraction(GM_segments,'aligned_data/features',fs=50)

        DI, cracks, aliigator, potholes = calc_target(aran_segments)
        
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1
        
        features.columns = features.columns.astype(int)
        features_test = features.sample(frac=1,axis=1)
        idx_test = list(features_test.columns)

        X_train = features_test.iloc[:,540:].reset_index(drop=True)
        DI_train = pd.DataFrame(np.array(DI)[idx_test[540:]])
        X_test = features_test.iloc[:,:540]
        DI_test = pd.DataFrame(np.array(DI)[idx_test[:540]])

        scores_RandomForest_DI = method_RandomForest(X_train, X_test, DI_train, DI_test, 'DI_GM_shuffle', model=model, gridsearch=gridsearch, cv_in=[4,False], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_shuffle.sav')
        print(rf_train.best_estimator_)


    if sys.argv[1] == 'plot':
        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_split1.sav')
        plt.plot(rf_train.cv_results_['mean_train_score'],label='train')
        plt.plot(rf_train.cv_results_['mean_test_score'],label='test')
        plt.plot(rf_train.cv_results_['split0_test_score'],label='split0')
        plt.plot(rf_train.cv_results_['split1_test_score'],label='split1')
        plt.plot(rf_train.cv_results_['split2_test_score'],label='split2')
        plt.plot(rf_train.cv_results_['split3_test_score'],label='split3')
        plt.legend()
        plt.show()



    if sys.argv[1] == 'Deep':
        GM_segments, aran_segments, route_details, dists = GM_sample_segmentation(segment_size=150)

        aran_dists = []
        for i in range(aran_segments.index.max()[0]+1):
            aran_dists.append(abs(aran_segments.loc[i]['EndChainage'].iloc[-1] - aran_segments.loc[i]['BeginChainage'].iloc[0]))

        diff = np.array(dists)-np.array(aran_dists)

        fig, axs = plt.subplots(2,1)
        axs[0].hist(aran_dists,bins=20)
        axs[0].hist(dists,bins=20)
        axs[0].set_title('Median dist: ' + str(round(np.median(dists),2)) + ' Average dist: ' + str(round(np.mean(dists),2)))
        axs[1].plot(diff)
        axs[1].set_title('Distances: ')
        plt.show()