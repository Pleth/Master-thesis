import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from functions import *
from LiRA_functions import *

from functools import partialmethod


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

    print('SVR')
    print('DI')
    scores_SVR_DI        = method_SVR(features, DI, 'DI', model=model, gridsearch=gridsearch, verbose=verbose,n_jobs=n_jobs)
    print('Potholes')
    scores_SVR_potholes  = method_SVR(features, potholes, 'potholes', model=model, gridsearch=gridsearch, verbose=verbose,n_jobs=n_jobs)
    print('Cracks')
    scores_SVR_cracks    = method_SVR(features, cracks, 'cracks', model=model, gridsearch=gridsearch, verbose=verbose,n_jobs=n_jobs)
    print('Alligator')
    scores_SVR_alligator = method_SVR(features, alligator, 'alligator', model=model, gridsearch=gridsearch, verbose=verbose,n_jobs=n_jobs)
    
    print('KNN')
    print('DI')
    scores_KNN_DI        = method_KNN(features, DI, 'DI', model=model, gridsearch=gridsearch, verbose=verbose,n_jobs=n_jobs)
    print('Potholes')
    scores_KNN_potholes  = method_KNN(features, potholes, 'potholes', model=model, gridsearch=gridsearch, verbose=verbose,n_jobs=n_jobs)
    print('Cracks')
    scores_KNN_cracks    = method_KNN(features, cracks, 'cracks', model=model, gridsearch=gridsearch, verbose=verbose,n_jobs=n_jobs)
    print('Alligator')
    scores_KNN_alligator = method_KNN(features, alligator, 'alligator', model=model, gridsearch=gridsearch, verbose=verbose,n_jobs=n_jobs)
    
    print('DT')
    print('DI')
    scores_DT_DI        = method_DT(features, DI, 'DI', model=model, gridsearch=gridsearch, verbose=verbose,n_jobs=n_jobs)
    print('Potholes')
    scores_DT_potholes  = method_DT(features, potholes, 'potholes', model=model, gridsearch=gridsearch, verbose=verbose,n_jobs=n_jobs)
    print('Cracks')
    scores_DT_cracks    = method_DT(features, cracks, 'cracks', model=model, gridsearch=gridsearch, verbose=verbose,n_jobs=n_jobs)
    print('Alligator')
    scores_DT_alligator = method_DT(features, alligator, 'alligator', model=model, gridsearch=gridsearch, verbose=verbose,n_jobs=n_jobs)



    scores_KNN_alligator = method_KNN(features, alligator, 'alligator', model=False, gridsearch=1, verbose=3,n_jobs=n_jobs)

    scores_DT_cracksv1    = method_DT(features, cracks, 'cracks', model=model, gridsearch=gridsearch, verbose=3,n_jobs=n_jobs)
    scores_DT_cracksv2    = method_DT(features, cracks, 'cracks', model=model, gridsearch=0, verbose=3,n_jobs=n_jobs)

    
    scores_DT_cracksv3    = method_DT(features, cracks, 'cracks', model=model, gridsearch=gridsearch, verbose=3,n_jobs=n_jobs)
    


    from tabulate import tabulate
    id = 'R2'
    r2 = tabulate([['SVR', scores_SVR_DI[id],scores_SVR_potholes[id],scores_SVR_cracks[id],scores_SVR_alligator[id]], 
              ['KNN', scores_KNN_DI[id],scores_KNN_potholes[id],scores_KNN_cracks[id],scores_KNN_alligator[id]], 
              ['DT',  scores_DT_DI[id],scores_DT_potholes[id],scores_DT_cracks[id],scores_DT_alligator[id]]], 
              headers=['Method', 'DI','Potholes','Cracks','Alligator'])
    id = 'MSE'
    mse = tabulate([['SVR', scores_SVR_DI[id],scores_SVR_potholes[id],scores_SVR_cracks[id],scores_SVR_alligator[id]], 
              ['KNN', scores_KNN_DI[id],scores_KNN_potholes[id],scores_KNN_cracks[id],scores_KNN_alligator[id]], 
              ['DT',  scores_DT_DI[id],scores_DT_potholes[id],scores_DT_cracks[id],scores_DT_alligator[id]]], 
              headers=['Method', 'DI','Potholes','Cracks','Alligator'])
    id = 'RMSE'
    rmse = tabulate([['SVR', scores_SVR_DI[id],scores_SVR_potholes[id],scores_SVR_cracks[id],scores_SVR_alligator[id]], 
              ['KNN', scores_KNN_DI[id],scores_KNN_potholes[id],scores_KNN_cracks[id],scores_KNN_alligator[id]], 
              ['DT',  scores_DT_DI[id],scores_DT_potholes[id],scores_DT_cracks[id],scores_DT_alligator[id]]], 
              headers=['Method', 'DI','Potholes','Cracks','Alligator'])
    id = 'MAE'
    mae = tabulate([['SVR', scores_SVR_DI[id],scores_SVR_potholes[id],scores_SVR_cracks[id],scores_SVR_alligator[id]], 
              ['KNN', scores_KNN_DI[id],scores_KNN_potholes[id],scores_KNN_cracks[id],scores_KNN_alligator[id]], 
              ['DT',  scores_DT_DI[id],scores_DT_potholes[id],scores_DT_cracks[id],scores_DT_alligator[id]]], 
              headers=['Method', 'DI','Potholes','Cracks','Alligator'])

    print('==========================R2===========================')
    print(r2)
    print('==========================MSE==========================')
    print(mse)
    print('==========================RMSE=========================')
    print(rmse)
    print('==========================MAE==========================')
    print(mae)

