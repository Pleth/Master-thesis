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
    
    # if len(sys.argv) != 4:
    #     print("3 arguments needed")
    #     sys.exit(1)
    # arg1 = sys.argv[1]
    # arg2 = sys.argv[2]
    # arg3 = sys.argv[3]
    
    
    # # arg1 - Train or eval
    # if arg1 == "SVR":
    #     print(arg1)
    # elif arg1 == "KNN":
    #     print(arg1)
    # elif arg1 == "DT":
    #     print(arg1)
    # else:
    #     print("Choose argument 1 (Method): SVR, KNN or DT")
    #     sys.exit(1)
    
    # # arg2 - Train or eval
    # if arg2 == "train":
    #     print(arg2)
    # elif arg2 == "eval":
    #     print(arg2)
    # else:
    #     print("Choose argument 2 (Training or evaluation): train or eval") 
    #     sys.exit(1)   

    # # arg3 - Data
    # if arg3 == "real":
    #     print(arg3)
    # elif arg3 == "sim":
    #     print(arg3)
    # else:
    #     print("Choose argument 3 (Real data or simulated data): real or sim")
    #     sys.exit(1)
    
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
    features,feature_names = feature_extraction(data)

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
    n_jobs = 4
    model = False

    cv = custom_splits(aran_segments,route_details)

    scores_RandomForest_DI        = method_RandomForest(features, DI, 'DI', model=model, gridsearch=gridsearch, cv=cv, verbose=verbose,n_jobs=n_jobs)
    scores_RandomForest_potholes  = method_RandomForest(features, potholes, 'potholes', model=model, gridsearch=gridsearch, cv=cv, verbose=verbose,n_jobs=n_jobs)
    scores_RandomForest_cracks    = method_RandomForest(features, cracks, 'cracks', model=model, gridsearch=gridsearch, cv=cv, verbose=verbose,n_jobs=n_jobs)
    scores_RandomForest_alligator = method_RandomForest(features, alligator, 'alligator', model=model, gridsearch=gridsearch, cv=cv, verbose=verbose,n_jobs=n_jobs)

    

    print(tabulate([['R2',scores_RandomForest_DI['R2'],scores_RandomForest_potholes['R2'],scores_RandomForest_cracks['R2'],scores_RandomForest_alligator['R2']],
                    ['MSE',scores_RandomForest_DI['MSE'],scores_RandomForest_potholes['MSE'],scores_RandomForest_cracks['MSE'],scores_RandomForest_alligator['MSE']],
                    ['RMSE',scores_RandomForest_DI['RMSE'],scores_RandomForest_potholes['RMSE'],scores_RandomForest_cracks['RMSE'],scores_RandomForest_alligator['RMSE']],
                    ['MAE',scores_RandomForest_DI['MAE'],scores_RandomForest_potholes['MAE'],scores_RandomForest_cracks['MAE'],scores_RandomForest_alligator['MAE']]], 
              headers=['RandomForest', 'DI','Potholes','Cracks','Alligator']))

    
    obj = scores_RandomForest_DI['Gridsearchcv_obj']
    feature_importance = obj.best_estimator_.feature_importances_
    bis_features = np.argpartition(feature_importance,-10)[-10:]

    top_10 = feature_names.index[bis_features]

    plt.plot(feature_importance)
    plt.annotate(top_10[0][2:], xy =(bis_features[0],feature_importance[bis_features[0]]),xytext =(bis_features[0]+10,feature_importance[bis_features[0]]+0.001),arrowprops = dict(arrowstyle = "->",facecolor ='green'))
    plt.text(bis_features[1],feature_importance[bis_features[1]],top_10[1][2:])
    plt.annotate(top_10[2][2:], xy =(bis_features[2],feature_importance[bis_features[2]]),xytext =(bis_features[2]+10,feature_importance[bis_features[2]]+0.002),arrowprops = dict(arrowstyle = "->",facecolor ='green'))
    plt.annotate(top_10[3][2:], xy =(bis_features[3],feature_importance[bis_features[3]]),xytext =(bis_features[3]-20,feature_importance[bis_features[3]]+0.0001))
    plt.annotate(top_10[4][2:], xy =(bis_features[4],feature_importance[bis_features[4]]),xytext =(bis_features[4]+10,feature_importance[bis_features[4]]+0.001),arrowprops = dict(arrowstyle = "->",facecolor ='green'))
    plt.annotate(top_10[5][2:], xy =(bis_features[5],feature_importance[bis_features[5]]),xytext =(bis_features[5]-50,feature_importance[bis_features[5]]+0),arrowprops = dict(arrowstyle = "->",facecolor ='green'))
    plt.text(bis_features[6],feature_importance[bis_features[6]],top_10[0][2:])
    plt.annotate(top_10[7][2:], xy =(bis_features[7],feature_importance[bis_features[7]]),xytext =(bis_features[7]-50,feature_importance[bis_features[7]]+0.001),arrowprops = dict(arrowstyle = "->",facecolor ='green'))
    plt.annotate(top_10[8][2:], xy =(bis_features[8],feature_importance[bis_features[8]]),xytext =(bis_features[8]-0,feature_importance[bis_features[8]]+0.001),arrowprops = dict(arrowstyle = "->",facecolor ='green'))
    plt.annotate(top_10[9][2:], xy =(bis_features[9],feature_importance[bis_features[9]]),xytext =(bis_features[9]+10,feature_importance[bis_features[9]]+0.001),arrowprops = dict(arrowstyle = "->",facecolor ='green'))
    plt.show()