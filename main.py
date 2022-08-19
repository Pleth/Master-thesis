import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import time

from functions import *
from LiRA_functions import *

import tsfel


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
    

    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split, GridSearchCV

    X = features.T
    y = DI
    
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    regressor = SVR(kernel='rbf',verbose=True)
    regressor.fit(X_train,y_train)
    
    y_pred = regressor.predict(X_test)

    r2 = r2_score(y_test,y_pred)
    MSE = mean_squared_error(y_test,y_pred, squared=True)
    RMSE = mean_squared_error(y_test,y_pred, squared=False)
    MAE = mean_absolute_error(y_test,y_pred)


    plt.plot(range(len(y_test)),y_test,label='True values')
    plt.plot(range(len(y_pred)),y_pred,label='Predicted values')
    plt.legend()
    plt.show()


    parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],'C': [1, 10, 100, 1000, 10000]}]

    parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 0.01, 0.1],'C': [1, 10, 20]},
                  #{'kernel': ['sigmoid'], 'gamma': [1e-5, 1e-4, 1e-3],'C': [1, 10, 100]},
                  #{'kernel': ['poly'], 'gamma': [1e-3, 0.01, 0.1],'C': [1, 10, 20]},
                  {'kernel': ['linear'], 'gamma': [1e-3, 0.01, 0.1],'C': [1, 10, 20]}]


    start_time = time.time()
    svr_train = GridSearchCV(SVR(epsilon = 0.01), parameters, cv = 5,verbose=3,n_jobs=4)
    svr_train.fit(X,y)
    end_time = time.time()
    run_time = end_time - start_time
    print('Run time:',round(run_time/60,2),'mins')
    

    regressor = svr_train.best_estimator_
    regressor.fit(X_train,y_train)
    
    y_pred = regressor.predict(X_test)

    r2 = r2_score(y_test,y_pred)
    MSE = mean_squared_error(y_test,y_pred, squared=True)
    RMSE = mean_squared_error(y_test,y_pred, squared=False)
    MAE = mean_absolute_error(y_test,y_pred)
