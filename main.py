import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

from functions import *
from LiRA_functions import *


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

    hdf5file = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
    
    aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
    aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
    aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
    aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

    DI = pd.DataFrame(calc_DI(aran_alligator,aran_cracks,aran_potholes))
    DI.columns=['DI']
    gt = pd.concat([aran_location,DI],axis=1)


    test = synth_acc[13]

    test2 = test[test['Distance'] < 50+np.min(test['Distance'])]

    upper = round(test2['Distance'][test2.index[-1]]/10)*10
    lower = round(test2['Distance'][0]/10)*10

    files = glob.glob("p79/*.csv")
    p79_gps = files[1]
    
    df_p79_gps = pd.read_csv(p79_gps)
    df_p79_gps.drop(df_p79_gps.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
            
    test3 = df_p79_gps[df_p79_gps['Distance'] <= upper]

    start = [test3['Latitude'][0],test3['Longitude'][0]]
    end = [test3['Latitude'][5],test3['Longitude'][5]]

    test4 = aran_location[(aran_location['LatitudeFrom'] >= start[0]) & (aran_location['LongitudeFrom'] >= start[1])]


    plt.scatter(x=aran_location['LongitudeFrom'], y=aran_location['LatitudeFrom'],s=1,c="red")#c=gt['DI'],cmap='gray')
    plt.scatter(x=df_p79_gps["Longitude"], y=df_p79_gps["Latitude"],s=1,c="blue")
    plt.show()








    #out = pd.DataFrame(data_window(synth_acc))
    #ids = pd.DataFrame(np.ones(125,),columns=["id"])
    
    # data, feature_names = feature_extraction(out,ids)

    # from sklearn.svm import SVR
    # y = np.random.randn(np.shape(data)[1])
    # regr = SVR(kernel='rbf')
    # regr.fit(np.transpose(data.values),y)

