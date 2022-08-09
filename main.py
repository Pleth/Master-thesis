import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from matplotlib.patches import Rectangle

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

    hdf5file = h5py.File('aligned_data/CPH1_HH.hdf5', 'r')
    
    aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
    aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
    aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
    aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

    DI = pd.DataFrame(calc_DI(aran_alligator,aran_cracks,aran_potholes))
    DI.columns=['DI']
    gt = pd.concat([aran_location,DI],axis=1)


    test = synth_acc[0]
    test = test[test['gm_speed'] >= 20]
    test = test.reset_index(drop=True)

    Lat_cord = [aran_location['LatitudeFrom'][1000],aran_location['LatitudeTo'][1005]]
    Lon_cord = [aran_location['LongitudeFrom'][1000],aran_location['LongitudeTo'][1005]]

    
    
    files = glob.glob("p79/*.csv")
    p79_gps = files[0]
    
    df_p79_gps = pd.read_csv(p79_gps)
    df_p79_gps.drop(df_p79_gps.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)

    dfdf = df_p79_gps[(((np.min(Lat_cord) <= df_p79_gps['Latitude']) & (df_p79_gps['Latitude'] <= np.max(Lat_cord))) & ((np.min(Lon_cord) <= df_p79_gps['Longitude']) & (df_p79_gps['Longitude'] <= np.max(Lon_cord))))]
    dfdf = dfdf.reset_index(drop=True)   

    test2 = test[((test['Distance'] >= np.min(dfdf['Distance'])) & (test['Distance'] <= np.max(dfdf['Distance'])))]
    test2 = test2.reset_index(drop=True)


    files = glob.glob("p79/*.csv")
        
    df_cph1_hh = pd.read_csv(files[0])
    df_cph1_hh.drop(df_cph1_hh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
    df_cph1_vh = pd.read_csv(files[1])
    df_cph1_vh.drop(df_cph1_vh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
    df_cph6_hh = pd.read_csv(files[2])
    df_cph6_hh.drop(df_cph6_hh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)
    df_cph6_vh = pd.read_csv(files[3])
    df_cph6_vh.drop(df_cph6_vh.columns.difference(['Distance','Latitude','Longitude']),axis=1,inplace=True)

    hdf5file_cph1_hh = h5py.File('aligned_data/CPH1_HH.hdf5', 'r')
    hdf5file_cph1_vh = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
    hdf5file_cph6_hh = h5py.File('aligned_data/CPH6_HH.hdf5', 'r')
    hdf5file_cph6_vh = h5py.File('aligned_data/CPH6_VH.hdf5', 'r')    
    
    
    iter = 0
    segments = {}
    for j in tqdm(range(len(routes))):
        synth = synth_acc[j]
        synth = synth[synth['synth_acc'].notna()]
        synth = synth[synth['gm_speed'] >= 20]
        synth = synth.reset_index(drop=True)
        route = routes[j][:7]
        if route == 'CPH1_HH':
            p79_gps = df_cph1_hh
            aran_location = pd.DataFrame(hdf5file_cph1_hh['aran/trip_1/pass_1']['Location'], columns = hdf5file_cph1_hh['aran/trip_1/pass_1']['Location'].attrs['chNames'])
        elif route == 'CPH1_VH':
            p79_gps = df_cph1_vh
            aran_location = pd.DataFrame(hdf5file_cph1_vh['aran/trip_1/pass_1']['Location'], columns = hdf5file_cph1_vh['aran/trip_1/pass_1']['Location'].attrs['chNames'])
        elif route == 'CPH6_HH':
            p79_gps = df_cph6_hh
            aran_location = pd.DataFrame(hdf5file_cph6_hh['aran/trip_1/pass_1']['Location'], columns = hdf5file_cph6_hh['aran/trip_1/pass_1']['Location'].attrs['chNames'])
        elif route == 'CPH6_VH':
            p79_gps = df_cph6_vh
            aran_location = pd.DataFrame(hdf5file_cph6_vh['aran/trip_1/pass_1']['Location'], columns = hdf5file_cph6_vh['aran/trip_1/pass_1']['Location'].attrs['chNames'])

        i,k = 0, 0
        # Get 50m from ARAN -> Find gps signal from p79 -> Get measurements from synthetic data
        while (i < (len(aran_location['LatitudeFrom'])-6) ):
            Lat_cord = [aran_location['LatitudeFrom'][i],aran_location['LatitudeTo'][i+4]]
            Lon_cord = [aran_location['LongitudeFrom'][i],aran_location['LongitudeTo'][i+4]]
            # Lat_cord = [np.min([aran_location['LatitudeFrom'][i:i+5],aran_location['LatitudeTo'][i:i+5]]),np.max([aran_location['LatitudeFrom'][i:i+5],aran_location['LatitudeTo'][i:i+5]])]
            # Lon_cord = [np.min([aran_location['LongitudeFrom'][i:i+5],aran_location['LongitudeTo'][i:i+5]]),np.max([aran_location['LongitudeFrom'][i:i+5],aran_location['LongitudeTo'][i:i+5]])]
            

            dfdf = p79_gps[(((np.min(Lat_cord) <= p79_gps['Latitude']) & (p79_gps['Latitude'] <= np.max(Lat_cord))) & ((np.min(Lon_cord) <= p79_gps['Longitude']) & (p79_gps['Longitude'] <= np.max(Lon_cord))))]
            dfdf = dfdf.reset_index(drop=True)   

            synth_seg = synth[((synth['Distance'] >= np.min(dfdf['Distance'])) & (synth['Distance'] <= np.max(dfdf['Distance'])))]
            synth_seg = synth_seg.reset_index(drop=True)

            stat1 = synth_seg['Distance'].empty
            if stat1:
                stat2 = True
                stat3 = True
            else:
                stat2 = (synth_seg['Distance'][len(synth_seg['Distance'])-1]-synth_seg['Distance'][0]) <= 40
                stat3 = (len(synth_seg['synth_acc'])) > 5000
            if stat1 | stat2 | stat3:
                i += 1
            else:
                k += 1
                i += 5
                segments[iter] = synth_seg['synth_acc']
                if len(segments[iter]) > 5000:
                    print('route:',route)
                    print('length:',len(synth_seg['synth_acc']))
                    print('length:',len(segments[iter]))
                iter += 1
    
    synth_segments = pd.DataFrame.from_dict(segments,orient='index').transpose()
    



    
    iter = 0
    i = 0
    dists = []
    while (i < (len(aran_location['LatitudeFrom'])-6) ):
            Lat_cord = [np.min([aran_location['LatitudeFrom'][i:i+5],aran_location['LatitudeTo'][i:i+5]]),np.max([aran_location['LatitudeFrom'][i:i+5],aran_location['LatitudeTo'][i:i+5]])]
            Lon_cord = [np.min([aran_location['LongitudeFrom'][i:i+5],aran_location['LongitudeTo'][i:i+5]]),np.max([aran_location['LongitudeFrom'][i:i+5],aran_location['LongitudeTo'][i:i+5]])]
            dfdf = p79_gps[(((np.min(Lat_cord) <= p79_gps['Latitude']) & (p79_gps['Latitude'] <= np.max(Lat_cord))) & ((np.min(Lon_cord) <= p79_gps['Longitude']) & (p79_gps['Longitude'] <= np.max(Lon_cord))))]
            dfdf = dfdf.reset_index(drop=True)   

            synth_seg = synth[((synth['Distance'] >= np.min(dfdf['Distance'])) & (synth['Distance'] <= np.max(dfdf['Distance'])))]
            synth_seg = synth_seg.reset_index(drop=True)

            stat1 = synth_seg['Distance'].empty
            if stat1:
                stat2 = True
                stat3 = True
            else:
                stat2 = (synth_seg['Distance'][len(synth_seg['Distance'])-1]-synth_seg['Distance'][0]) <= 40
                stat3 = (len(synth_seg['synth_acc'])) > 5000
            if stat1 | stat2 | stat3:
                i += 1
            else:
                i += 5
                dists.append(synth_seg['Distance'][len(synth_seg['Distance'])-1]-synth_seg['Distance'][0])
                _ = plt.gca().add_patch(Rectangle((Lon_cord[0],Lat_cord[0]),Lon_cord[1]-Lon_cord[0],Lat_cord[1]-Lat_cord[0],edgecolor='green',facecolor='none',lw=1))
                iter += 1
            # Lat_cord = [aran_location['LatitudeFrom'][i],aran_location['LatitudeTo'][i+4]]
            # Lon_cord = [aran_location['LongitudeFrom'][i],aran_location['LongitudeTo'][i+4]]
            # _ = plt.gca().add_patch(Rectangle((Lon_cord[0],Lat_cord[0]),Lon_cord[1]-Lon_cord[0],Lat_cord[1]-Lat_cord[0],edgecolor='purple',facecolor='none',lw=1))
            # i += 5

    tester = synth_acc[0]
    tester = tester[tester['synth_acc'].notna()]
    tester = tester[tester['synth_acc'] != 0]
    p79_gps = p79_gps[p79_gps['Longitude'] != 0]
    p79_gps = p79_gps[p79_gps['Latitude'] != 0]
    p79_gps = p79_gps.reset_index(drop=True)
    plt.scatter(x=p79_gps[p79_gps["Longitude"] != 0]['Longitude'], y=p79_gps[p79_gps["Latitude"] != 0]['Latitude'],s=1,c="blue")
    for k in tqdm(range(len(p79_gps)-1)):
        x_val = [p79_gps['Longitude'][k],p79_gps['Longitude'][k+1]]
        y_val = [p79_gps['Latitude'][k],p79_gps['Latitude'][k+1]]

        if (np.min(x_val) < 10):
            print('x index:',k)
        if (np.min(y_val) < 10):
            print('y index:',k)
        

        speed = tester[(tester['Distance'] >= p79_gps['Distance'][k]) & (tester['Distance'] <= p79_gps['Distance'][k+1])]
        if (np.min(speed['gm_speed']) < 20):
            _ = plt.plot(x_val,y_val,'r')
        # else:
        #     _ = plt.plot(x_val,y_val,'b')


    plt.scatter(x=aran_location['LongitudeFrom'], y=aran_location['LatitudeFrom'],s=1,c="red")
    plt.scatter(x=aran_location['LongitudeTo'], y=aran_location['LatitudeTo'],s=1,c="black")
    # plt.scatter(x=p79_gps[p79_gps["Longitude"] != 0]['Longitude'], y=p79_gps[p79_gps["Latitude"] != 0]['Latitude'],s=1,c="blue")
    plt.show()

    #plt.savefig('plots/segmen_vali.pdf')

    plt.hist(np.array(dists),bins=100)
    plt.show()









    #out = pd.DataFrame(data_window(synth_acc))
    #ids = pd.DataFrame(np.ones(125,),columns=["id"])
    
    # data, feature_names = feature_extraction(out,ids)

    # from sklearn.svm import SVR
    # y = np.random.randn(np.shape(data)[1])
    # regr = SVR(kernel='rbf')
    # regr.fit(np.transpose(data.values),y)

