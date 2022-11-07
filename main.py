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


if __name__ == '__main__':

    if sys.argv[1] == 'rm_aligned':
        os.remove('aligned_data/aran_segments.csv')
        os.remove('aligned_data/routes_details.txt')
        os.remove('aligned_data/synthetic_segments.csv')
        os.remove('aligned_data/features.csv')

    if sys.argv[1] == 'synth_test':
        print('synth_test')
        synth_acc = synthetic_data()
        routes = []
        for i in range(len(synth_acc)): 
            routes.append(synth_acc[i].axes[0].name)

        synth_segments, aran_segments, route_details = synthetic_segmentation(synth_acc,routes,segment_size=5,overlap=0)

        features,feature_names = feature_extraction(synth_segments,'synth_data/extracted_features',fs=250)

        DI, cracks, aliigator, potholes = calc_target(aran_segments)
        
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1

        cut = [10500,18500,15000]

        cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split3')
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['3']].reset_index(drop=True)
        DI_train = DI.iloc[splits['1']+splits['2']+splits['4']+splits['5']].reset_index(drop=True)
        
        scores_RandomForest_DI = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_synth', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_synth.sav')
        print(rf_train.best_estimator_)



    ################################################## TESTS - 45 km/h speed #################################################
    if sys.argv[1] == '45km_test':
        print('45km_test')
        synth_acc = test_synthetic_data()
        routes = []
        for i in range(len(synth_acc)): 
            routes.append(synth_acc[i].axes[0].name)

        synth_segments, aran_segments, route_details, laser_segments = test_segmentation(synth_acc,routes,segment_size=5,overlap=0)
        
        features,feature_names = feature_extraction(synth_segments,'synth_data/tests/features_45km',fs=250)

        DI, cracks, aliigator, potholes = calc_target(aran_segments)
        
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1

        cut = [10500,18400,15000]

        cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split3')
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['3']].reset_index(drop=True)
        DI_train = DI.iloc[splits['1']+splits['2']+splits['4']+splits['5']].reset_index(drop=True)

        scores_RandomForest_DI = method_RandomForest(X_train, X_test, DI_train, DI_test, 'DI_45km', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_45km.sav')
        print(rf_train.best_estimator_)


    ################################################## TESTS - (laser5+laser21)/2/1e3 #################################################
    if sys.argv[1] == 'laser_test':
        print('laser_test')
        synth_acc = test_synthetic_data()
        routes = []
        for i in range(len(synth_acc)): 
            routes.append(synth_acc[i].axes[0].name)

        synth_segments, aran_segments, route_details, laser_segments = test_segmentation(synth_acc,routes,segment_size=5,overlap=0)

        features,feature_names = feature_extraction(laser_segments,'synth_data/tests/features_laser',fs=250)

        DI, cracks, aliigator, potholes = calc_target(aran_segments)
        
        gridsearch = 1
        verbose = 0
        n_jobs = -1 # 4
        if sys.argv[2] == 'train':
            model = False
        elif sys.argv[2] == 'test':
            model = 1

        cut = [10500,18400,15000]

        cv_train, split_test, X_train, X_test, splits = real_splits(features,aran_segments,route_details,cut,'split3')
        DI = pd.DataFrame(DI)
        DI_test = DI.iloc[splits['3']].reset_index(drop=True)
        DI_train = DI.iloc[splits['1']+splits['2']+splits['4']+splits['5']].reset_index(drop=True)

        scores_RandomForest_DI        = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_laser', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)

        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_laser.sav')
        print(rf_train.best_estimator_)
        

    ########################################################### GREEN MOBILITY TEST ##############################################################
    if sys.argv[1] == 'GM_test':
        print('GM_test')
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
        
        scores_RandomForest_DI        = method_RandomForest(X_train,X_test, DI_train, DI_test, 'DI_GM_sample', model=model, gridsearch=gridsearch, cv_in=[cv_train,split_test], verbose=verbose,n_jobs=n_jobs)
        
        print(scores_RandomForest_DI['R2'][1])
        print(scores_RandomForest_DI['R2'][0])

        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_sample.sav')
        print(rf_train.best_estimator_)

        plt.plot(rf_train.cv_results_['mean_train_score'])
        plt.plot(rf_train.cv_results_['mean_test_score'])
        plt.show()
        np.argpartition(rf_train.cv_results_['mean_test_score'],-5)[-5:]



    if sys.argv[1] == 'GM_split_test':
        print('GM_split_test')
        GM_segments, aran_segments, route_details = GM_segmentation(segment_size=5,overlap=0)

        features,feature_names = feature_extraction(GM_segments,'aligned_data/features_split',fs=50)

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

    if sys.argv[1] == 'GM_route_test':
        print('GM_route_test')
        GM_segments, aran_segments, route_details = GM_segmentation(segment_size=5,overlap=0)
        
        features,feature_names = feature_extraction(GM_segments,'aligned_data/features_route',fs=50)

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
        print('GM_shuffle_test')
        GM_segments, aran_segments, route_details = GM_segmentation(segment_size=5,overlap=0)

        features,feature_names = feature_extraction(GM_segments,'aligned_data/features_shuffle',fs=50)

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
        rf_train = joblib.load('models/RandomForest_best_model_DI_GM_cph6_vh.sav')
        plt.plot(rf_train.cv_results_['mean_train_score'],label='train')
        plt.plot(rf_train.cv_results_['mean_test_score'],label='test')
        plt.plot(rf_train.cv_results_['split0_test_score'],label='split0')
        plt.plot(rf_train.cv_results_['split1_test_score'],label='split1')
        plt.plot(rf_train.cv_results_['split2_test_score'],label='split2')
        plt.plot(rf_train.cv_results_['split3_test_score'],label='split3')
        plt.legend()
        plt.show()


    if sys.argv[1] == 'Dataset_create':
        GM_segments, aran_segments, route_details, dists = GM_sample_segmentation(segment_size=150)
        DI, cracks, alligator, potholes = calc_target(aran_segments)
        cut = [4700,9500,18500,15000]
        splits = DL_splits(aran_segments,route_details,cut)
        print(len(splits['1']), len(splits['2']), len(splits['3']), len(splits['4']))
        targets = {}
        targets[0] = DI
        targets[1] = cracks
        targets[2] = alligator
        targets[3] = potholes
        create_cwt_data(GM_segments,splits,targets,'DL_data','real')

    if sys.argv[1] == 'synth_dataset_create':
        synth_acc = synthetic_data()
        routes = []
        for i in range(len(synth_acc)): 
            routes.append(synth_acc[i].axes[0].name)
        GM_segments, aran_segments, route_details, dists = synthetic_sample_segmentation(synth_acc,routes,segment_size=896)
        DI, cracks, alligator, potholes = calc_target(aran_segments)
        cut = [8900,16300,20600,13000,17400]
        splits = DL_splits(aran_segments,route_details,cut)
        print(len(splits['1']),len(splits['2']),len(splits['3']),len(splits['4']),len(splits['5']),len(splits['6']),len(splits['7']))
        targets = {}
        targets[0] = DI
        targets[1] = cracks
        targets[2] = alligator
        targets[3] = potholes
        create_cwt_data(GM_segments,splits,targets,'DL_synth_data','synth')

    if sys.argv[1] == 'Deep':
        print('Deep')
        # prepare the data
        batch_size = 16
        path = 'DL_synth_data'
        # path = 'DL_data'
        labelsFile = 'DL_synth_data/labelsfile'
        # labelsFile = 'DL_data/labelsfile'
        train_dl, val_dl, test_dl = prepare_data(path,labelsFile,batch_size,nr_tar=1)
        print(len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset))

        train_features, train_labels = next(iter(train_dl))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        # img = train_features[0] #.squeeze()
        # img1 = img.permute(1,2,0)
        # label = train_labels[0]
        # plt.imshow(img1,cmap="gray")
        # plt.show()
        # print(f"Label: {label}")
        # model = CNN_simple(4)
        model = MyGoogleNet(in_fts=4,num_class=1)
        # print(model)
        train_model(train_dl, val_dl, model, 100, 0.00001)
        
        model_test = MyGoogleNet(in_fts=4,num_class=1)
        model_test.load_state_dict(torch.load("models/your_model_path.pt"))
        model_test.eval()
        acc = evaluate_model(test_dl, model_test)
        print('Test R2 - DI: %.3f' % acc)



    if sys.argv[1] == 'Linear':
        print('Linear')
        synth_acc = synthetic_data()
        routes = []
        for i in range(len(synth_acc)): 
            routes.append(synth_acc[i].axes[0].name)

        synth_segments, aran_segments, route_details = synthetic_segmentation(synth_acc,routes,segment_size=5,overlap=0)

        features,feature_names = feature_extraction(synth_segments,'synth_data/extracted_features',fs=250)

        DI, cracks, aliigator, potholes = calc_target(aran_segments)
        cut = [7000,13700,22000,13000]
        splits = DL_splits(aran_segments,route_details,cut)
        
        feats = features.T
        feats = (feats - feats.min())/(feats.max()-feats.min()+0.001)
        X_train = feats.iloc[splits['3']].reset_index(drop=True)
        X_train = pd.concat([X_train,feats.iloc[splits['4']]],ignore_index=True)
        X_train = pd.concat([X_train,feats.iloc[splits['5']]],ignore_index=True)
        X_train = pd.concat([X_train,feats.iloc[splits['6']]],ignore_index=True).reset_index(drop=True).T
        
        DI = pd.DataFrame(DI)
        y_train = DI.iloc[splits['3'] + splits['4'] + splits['5'] + splits['6']].reset_index(drop=True)

        train_dataset = FeatureDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True)
        print(len(train_loader.dataset))

        X_val = feats.iloc[splits['2']].reset_index(drop=True).T
        y_val = DI.iloc[splits['2']].reset_index(drop=True)
        val_dataset = FeatureDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1100, shuffle = True)
        print(len(val_loader.dataset))

        X_test = feats.iloc[splits['1']].reset_index(drop=True).T
        y_test = DI.iloc[splits['1']].reset_index(drop=True)
        test_dataset = FeatureDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1100, shuffle = True)





        model = LinearRegression_class(390,1)
        train_model(train_loader, val_loader, model, 100, 0.001,0.0,0.01,'NNLinReg')

        model_test = LinearRegression_class(390,1)
        model_test.load_state_dict(torch.load("models/model_NNLinReg.pt"))
        model_test.eval()
        acc = evaluate_model(test_loader, model_test)
        print('Test R2 - DI: %.3f' % acc)
        
        from sklearn.linear_model import Ridge, RidgeCV
        clf = Ridge(alpha=0.01).fit(X_train.T, y_train.values.reshape(-1,))
        y_pred = clf.predict(X_test.T)
        r2 = r2_score(y_test,y_pred)
        print(r2)

        batch_size = 32
        nr_tar=1
        path = 'DL_synth_data'
        labelsFile = 'DL_synth_data/labelsfile'
        sourceTransform = Compose([ToTensor()]) #, Resize((224,224))
        # load dataset
        train = CustomDataset(labelsFile+"_train.csv", path+'/train/', sourceTransform, nr_tar) 
        val = CustomDataset(labelsFile+"_val.csv", path+'/val/', sourceTransform, nr_tar)
        test = CustomDataset(labelsFile+"_test.csv", path+'/test/', sourceTransform, nr_tar)
        # prepare data loaders
        train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(test, batch_size=batch_size, shuffle=False)
        print(len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset))

        batch_max = []
        batch_min = []
        for i, (inputs, targets) in enumerate(train_dl):
            batch_max.append(np.max(inputs.numpy()))
            batch_min.append(np.min(inputs.numpy()))
        for i, (inputs, targets) in enumerate(val_dl):
            batch_max.append(np.max(inputs.numpy()))
            batch_min.append(np.min(inputs.numpy()))
        for i, (inputs, targets) in enumerate(test_dl):
            batch_max.append(np.max(inputs.numpy()))
            batch_min.append(np.min(inputs.numpy()))
        np.max(batch_max) # 0.9969462
        np.min(batch_min) # 1.18216e-08

        test_features, test_labels = next(iter(test_dl))
        print(f"Feature batch shape: {test_features.size()}")
        print(f"Labels batch shape: {test_labels.size()}")

        img = test_features[4] #.squeeze()
        img1 = img.permute(1,2,0)
        label = test_labels[4]
        plt.imshow(img1)
        plt.show()

        model = ImageRegression_class(224*224,1)
        train_model(train_dl, val_dl,test_dl, model, 100, 1e-5,0.0,0.001,'IMLinReg')
        
        model_test = ImageRegression_class(224*224,1)
        model_test.load_state_dict(torch.load("models/model_IMLinReg.pt"))
        model_test.eval()
        acc = evaluate_model(test_dl, model_test)
        print('Test R2 - DI: %.3f' % acc)

        segs = []
        tar = []
        for i, (inputs, targets) in enumerate(train_dl):
            inputs, targets = inputs, targets
            for j in range(inputs.size()[0]):
                arr = np.array(inputs[j].view(224*224,-1).squeeze())
                segs.append(arr)
                tar.append(np.array(targets[j]))
        trainer = pd.DataFrame(np.vstack(segs))        
        trainer = trainer.T
        train_tar = pd.DataFrame(tar)

        segs = []
        tar = []
        for i, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs, targets
            for j in range(inputs.size()[0]):
                arr = np.array(inputs[j].view(224*224,-1).squeeze())
                segs.append(arr)
                tar.append(np.array(targets[j]))
        tester = pd.DataFrame(np.vstack(segs))        
        tester = tester.T
        test_tar = pd.DataFrame(tar)

        clf = RidgeCV(alphas=[0.9],scoring='r2').fit(trainer.T, train_tar.values.reshape(-1,))
        y_pred = clf.predict(trainer.T)
        r2 = r2_score(train_tar,y_pred)
        print(r2)
        y_pred = clf.predict(tester.T)
        r2 = r2_score(test_tar,y_pred)
        print(r2)






        model = MyGoogleNet(in_fts=1,num_class=1)
        # print(model)
        train_model(train_dl, val_dl, model, 100, 1e-6,0.0,0.001,'GoogleNet')
        
        model_test = MyGoogleNet(in_fts=1,num_class=1)
        model_test.load_state_dict(torch.load("models/model_GoogleNet.pt"))
        model_test.eval()
        acc = evaluate_model(test_dl, model_test)
        print('Test R2 - DI: %.3f' % acc)



        model = LinearBaseline(390,1)
        train_model(train_loader, val_loader, model, 100, 0.001,0.0,0.01,'LinearBaseline')

        model_test = LinearBaseline(390,1)
        model_test.load_state_dict(torch.load("models/model_LinearBaseline.pt"))
        model_test.eval()
        acc = evaluate_model(test_loader, model_test)
        print('Test R2 - DI: %.3f' % acc)
        



    if sys.argv[1] == 'Deep_linear':
        print('Deep_linear')
        
        batch_size = 16
        nr_tar=1
        path = 'DL_synth_data'
        labelsFile = 'DL_synth_data/labelsfile'
        train_dl, val_dl, test_dl = prepare_data(path,labelsFile,batch_size,nr_tar=1)
        print(len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset))

        model = ImageRegression_class(224*224,1)
        train_model(train_dl, val_dl,test_dl, model, 100, 1e-5,0.0,0.001,'IMLinReg')
        
        model_test = ImageRegression_class(224*224,1)
        model_test.load_state_dict(torch.load("models/model_IMLinReg.pt"))
        model_test.eval()
        acc = evaluate_model(test_dl, model_test)
        print('Test R2 - DI: %.3f' % acc)
    
    if sys.argv[1] == 'Deep_google':
        print('Deep_google')
        
        batch_size = int(sys.argv[2])
        lr = float(sys.argv[3])
        wd = float(sys.argv[4])
        epoch_nr = int(sys.argv[5])

        print('batch_size = ',batch_size,'lr = ',lr,'wd = ',wd)

        nr_tar=1
        test_nr = float(sys.argv[7])
        path = 'DL_synth_data'
        labelsFile = 'DL_synth_data/labelsfile'
        train_dl, val_dl, test_dl = prepare_data(path,labelsFile,batch_size,nr_tar=1,test_nr=test_nr)
        print(len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset))

        id = 'GoogleNet_'+sys.argv[6]

        model = MyGoogleNet(in_fts=1,num_class=1)
        train_model(train_dl, val_dl, test_dl, model, epoch_nr, lr,0.0,wd,id)

        model.eval()
        acc = evaluate_model(test_dl, model)
        print('Test R2 - DI: %.3f' % acc)

        model_test = MyGoogleNet(in_fts=1,num_class=1)
        model_test.load_state_dict(torch.load("models/model_GoogleNet_"+id+".pt")) # ,map_location=torch.device('cpu')
        model_test.eval()
        acc = evaluate_model(test_dl, model_test)
        print('Test R2 - DI: %.3f' % acc)
