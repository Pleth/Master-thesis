import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import time

from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from haversine import haversine, inverse_haversine, Direction, Unit

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

    synth_segments, aran_segments = synthetic_segmentation(synth_acc,routes,segment_size=5,overlap=0)







    hdf5file = h5py.File('aligned_data/CPH6_VH.hdf5', 'r')
    
    aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
    aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
    aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
    aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

    DI = pd.DataFrame(calc_DI(aran_alligator,aran_cracks,aran_potholes))
    DI.columns=['DI']
    gt = pd.concat([aran_location,DI],axis=1)
    
    files = glob.glob("p79/*.csv")
    df_cph6_hh = pd.read_csv(files[2])
    df_cph6_vh = pd.read_csv(files[3])
    
    plt.scatter(x=df_cph6_hh['Longitude'], y=df_cph6_hh['Latitude'],s=1,c="blue")
    plt.scatter(x=df_cph6_vh['Longitude'], y=df_cph6_vh['Latitude'],s=1,c="green")
    hdf5file = h5py.File('aligned_data/CPH6_VH.hdf5', 'r')
    aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
    plt.scatter(x=aran_location['LongitudeFrom'], y=aran_location['LatitudeFrom'],s=1,c="red")
    plt.plot(aran_location['LongitudeFrom'][2291-770+4],aran_location['LatitudeFrom'][2291-770+4],'bo')

    aran_start_VH = [aran_location['LatitudeFrom'][2291-770+4],aran_location['LongitudeFrom'][2291-770+4]]

    hdf5file = h5py.File('aligned_data/CPH6_HH.hdf5', 'r')
    aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
    plt.scatter(x=aran_location['LongitudeFrom'], y=aran_location['LatitudeFrom'],s=1,c="black")
    plt.plot(aran_location['LongitudeFrom'][920],aran_location['LatitudeFrom'][920],'ro')
    
    aran_start_HH = [aran_location['LatitudeFrom'][920],aran_location['LongitudeFrom'][920]]

    p79_idx_vh, dist_vh = find_min_gps_vector(aran_start_VH,df_cph6_vh[['Latitude','Longitude']].values)
    
    p79_idx_hh, dist_hh = find_min_gps_vector(aran_start_HH,df_cph6_hh[['Latitude','Longitude']].values)

    plt.plot(df_cph6_vh['Longitude'][p79_idx_vh],df_cph6_vh['Latitude'][p79_idx_vh],'bo')
    plt.plot(df_cph6_hh['Longitude'][p79_idx_hh],df_cph6_hh['Latitude'][p79_idx_hh],'ro')

    new_cph6_hh = df_cph6_hh[p79_idx_hh:]
    new_cph6_hh = new_cph6_hh.reset_index(drop=True)
    new_cph6_vh = df_cph6_vh[:p79_idx_vh]
    
    new_cph6_hh.to_csv("p79/"+"new_CPH6_HH"+".csv",index=False)
    new_cph6_vh.to_csv("p79/"+"new_CPH6_VH"+".csv",index=False)

    p79_gps = df_cph6_hh
    p79_start = p79_gps[['Latitude','Longitude']].iloc[0].values
    aran_start_idx, dist1 = find_min_gps_vector(p79_start,aran_location[['LatitudeFrom','LongitudeFrom']].values)
    p79_end = p79_gps[['Latitude','Longitude']].iloc[-1].values
    aran_end_idx, dist2 = find_min_gps_vector(p79_end,aran_location[['LatitudeTo','LongitudeTo']].values)
    
    plt.plot(aran_location['LongitudeFrom'][aran_start_idx],aran_location['LatitudeFrom'][aran_start_idx],'go')
    plt.plot(aran_location['LongitudeTo'][aran_end_idx],aran_location['LatitudeTo'][aran_end_idx],'go')

    plt.show()






    synth_acc = synthetic_data()
    routes = []
    for i in range(len(synth_acc)): 
        routes.append(synth_acc[i].axes[0].name)

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
    for j in tqdm(range(1)):
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


        p79_start = p79_gps[['Latitude','Longitude']].iloc[0].values
        # _, aran_start_idx, _ = find_min_gps(p79_start[0], p79_start[1], aran_location['LatitudeFrom'].iloc[:100].values, aran_location['LongitudeFrom'].iloc[:100].values)
        aran_start_idx, _ = find_min_gps_vector(p79_start,aran_location[['LatitudeFrom','LongitudeFrom']].iloc[:100].values)
        aran_location = aran_location.iloc[aran_start_idx:].reset_index(drop=True)

        i = 0
        # Get 50m from ARAN -> Find gps signal from p79 -> Get measurements from synthetic data
        while (i < (len(aran_location['LatitudeFrom'])-6) ):
            aran_start = [aran_location['LatitudeFrom'][i],aran_location['LongitudeFrom'][i]]
            aran_end = [aran_location['LatitudeTo'][i+4],aran_location['LongitudeTo'][i+4]]

            # start = time.time()
            # _, p79_start_idx1, start_dist1 = find_min_gps(aran_start[0], aran_start[1], p79_gps['Latitude'].values, p79_gps['Longitude'].values)
            # _, p79_end_idx1, end_dist1 = find_min_gps(aran_end[0], aran_end[1], p79_gps['Latitude'].values, p79_gps['Longitude'].values)
            # end = time.time()
            # print('Iteration:',i)
            # print('start_dist:',start_dist)
            # print('end_dist:',end_dist)

            # p79_start1 = [p79_gps['Latitude'][p79_start_idx1],p79_gps['Longitude'][p79_start_idx1]]



            p79_start_idx, start_dist = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
            p79_end_idx, end_dist = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)

            # p79_start_idx2, start_dist2 = find_min_gps_cart(aran_start,p79_gps[['Latitude','Longitude']].values)
            # p79_end_idx2, end_dist2 = find_min_gps_cart(aran_end,p79_gps[['Latitude','Longitude']].values)

            p79_start_idx2, start_dist2 = tester_test(aran_start,p79_gps[['Latitude','Longitude']].values)
            p79_end_idx2, end_dist2 = tester_test(aran_end,p79_gps[['Latitude','Longitude']].values)

            # if p79_start_idx != p79_start_idx2:
            #     print('Haversine vs Euclidian:',p79_start_idx,p79_start_idx2)

            # p79_start = [p79_gps['Latitude'][p79_start_idx],p79_gps['Longitude'][p79_start_idx]]

            # start1 = time.time()
            # p79_start_idx, start_dist = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
            # end1 = time.time()
            # time1 = end1 - start1

            # print('Distance difference start:',abs(start_dist-start_dist1))
            # print('Distance difference end  :',abs(end_dist-end_dist1))

            if start_dist < 15 and end_dist < 15:
                dfdf = p79_gps['Distance'][p79_start_idx:p79_end_idx]
                dfdf = dfdf.reset_index(drop=True)   

                synth_seg = synth[((synth['Distance'] >= np.min(dfdf)) & (synth['Distance'] <= np.max(dfdf)))]
                synth_seg = synth_seg.reset_index(drop=True)

                stat1 = synth_seg['Distance'].empty
                lag = []
                for h in range(len(synth_seg)-1):
                    lag.append(synth_seg['Distance'][h+1]-synth_seg['Distance'][h])        
                large = [y for y in lag if y > 5]
                
                if stat1:
                    stat2 = True
                    stat3 = True
                    stat4 = True
                else:
                    stat2 = (synth_seg['Distance'][len(synth_seg['Distance'])-1]-synth_seg['Distance'][0]) <= 40
                    stat3 = (len(synth_seg['synth_acc'])) > 5000
                    stat4 = False if bool(large) == False else (np.max(large) > 5)
                if stat1 | stat2 | stat3 | stat4:
                    i += 1
                else:
                    i += 5
                    segments[iter] = synth_seg['synth_acc']
                    Lat_cord = [p79_gps['Latitude'][p79_start_idx],p79_gps['Latitude'][p79_end_idx]]
                    Lon_cord = [p79_gps['Longitude'][p79_start_idx],p79_gps['Longitude'][p79_end_idx]]
                    _ = plt.gca().add_patch(Rectangle((Lon_cord[0],Lat_cord[0]),Lon_cord[1]-Lon_cord[0],Lat_cord[1]-Lat_cord[0],edgecolor='green',facecolor='none',lw=1))
                    iter += 1

                    x_val = [aran_start[1],p79_gps['Longitude'][p79_start_idx]]
                    y_val = [aran_start[0],p79_gps['Latitude'][p79_start_idx]]
                    _ = plt.plot(x_val,y_val,'r')

                    x_val = [aran_end[1],p79_gps['Longitude'][p79_end_idx]]
                    y_val = [aran_end[0],p79_gps['Latitude'][p79_end_idx]]
                    _ = plt.plot(x_val,y_val,'k')

                    x_val = [aran_start[1],p79_gps['Longitude'][p79_start_idx2]]
                    y_val = [aran_start[0],p79_gps['Latitude'][p79_start_idx2]]
                    _ = plt.plot(x_val,y_val,'magenta')

                    x_val = [aran_end[1],p79_gps['Longitude'][p79_end_idx2]]
                    y_val = [aran_end[0],p79_gps['Latitude'][p79_end_idx2]]
                    _ = plt.plot(x_val,y_val,'yellow')

            else:
                i +=1

        # tester = synth_acc[j]
        # tester = tester[tester['synth_acc'].notna()]

        lat_len = 111332.67
        lon_len = 63195.85

        p79_gps = p79_gps[p79_gps['Longitude'] != 0]
        p79_gps = p79_gps[p79_gps['Latitude'] != 0]
        p79_gps = p79_gps.reset_index(drop=True)
        fig,ax = plt.subplots(figsize=(10,10))
        plt.scatter(x=p79_gps['Longitude']*lon_len, y=p79_gps['Latitude']*lat_len,s=1,c="blue",label='p79 route')
        # for k in tqdm(range(len(p79_gps)-1)):
        #     x_val = [p79_gps['Longitude'][k],p79_gps['Longitude'][k+1]]
        #     y_val = [p79_gps['Latitude'][k],p79_gps['Latitude'][k+1]]

        #     if (np.min(x_val) < 10):
        #         print('x index:',k)
        #     if (np.min(y_val) < 10):
        #         print('y index:',k)
            
        #     speed = tester[(tester['Distance'] >= p79_gps['Distance'][k]) & (tester['Distance'] <= p79_gps['Distance'][k+1])]
        #     if (np.min(speed['gm_speed']) < 20):
        #         _ = plt.plot(x_val,y_val,'grey')
        
        plt.scatter(x=aran_location['LongitudeFrom']*lon_len, y=aran_location['LatitudeFrom']*lat_len,s=1,c="red",label='AranFrom')
        plt.scatter(x=aran_location['LongitudeTo']*lon_len, y=aran_location['LatitudeTo']*lat_len,s=1,c="black",label='AranTo')
        # plt.scatter(x=p79_gps[p79_gps["Longitude"] != 0]['Longitude'], y=p79_gps[p79_gps["Latitude"] != 0]['Latitude'],s=1,c="blue")
           
        aran_start = [aran_location['LatitudeFrom'][2450],aran_location['LongitudeFrom'][2450]]
        aran_end = [aran_location['LatitudeTo'][2450+4],aran_location['LongitudeTo'][2450+4]]

        plt.plot(aran_start[1]*lon_len,aran_start[0]*lat_len,'ro')

        p79_start_idx, start_dist = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
        p79_end_idx, end_dist = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)

        p79_start_idx2, start_dist2 = tester_test(aran_start,p79_gps[['Latitude','Longitude']].values)
        p79_end_idx2, end_dist2 = tester_test(aran_end,p79_gps[['Latitude','Longitude']].values)

        haversine(aran_start,p79_gps[['Latitude','Longitude']].values[p79_start_idx],unit=Unit.METERS)
        haversine(aran_start,p79_gps[['Latitude','Longitude']].values[p79_start_idx-20],unit=Unit.METERS)
        haversine(aran_start,p79_gps[['Latitude','Longitude']].values[p79_start_idx2],unit=Unit.METERS)
        haversine(aran_start,p79_gps[['Latitude','Longitude']].values[p79_start_idx2-20],unit=Unit.METERS)

        x_val = [aran_start[1]*lon_len,p79_gps['Longitude'][p79_start_idx]*lon_len]
        y_val = [aran_start[0]*lat_len,p79_gps['Latitude'][p79_start_idx]*lat_len]
        _ = plt.plot(x_val,y_val,'r')

        x_val = [aran_end[1]*lon_len,p79_gps['Longitude'][p79_end_idx]*lon_len]
        y_val = [aran_end[0]*lat_len,p79_gps['Latitude'][p79_end_idx]*lat_len]
        _ = plt.plot(x_val,y_val,'k')

        x_val = [aran_start[1]*lon_len,p79_gps['Longitude'][p79_start_idx2]*lon_len]
        y_val = [aran_start[0]*lat_len,p79_gps['Latitude'][p79_start_idx2]*lat_len]
        _ = plt.plot(x_val,y_val,'magenta')

        x_val = [aran_end[1]*lon_len,p79_gps['Longitude'][p79_end_idx2]*lon_len]
        y_val = [aran_end[0]*lat_len,p79_gps['Latitude'][p79_end_idx2]*lat_len]
        _ = plt.plot(x_val,y_val,'yellow')

        custom_lines = [Line2D([0], [0], color='blue', lw=2),
                Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], color='black', lw=2),
                Line2D([0], [0], color='green', lw=2)]
        plt.legend(custom_lines,['p79 route','AranFrom','AranTo','Segment'])
        plt.title('Segmentation algorithm')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        plt.axis([(aran_start[1])*lon_len-10,(aran_start[1])*lon_len+10,(aran_start[0])*lat_len-10,(aran_start[0])*lat_len+10])
        # plt.axis([(aran_start[1]-0.0002),(aran_start[1]+0.0002),(aran_start[0]-0.0002),(aran_start[0]+0.0002)])

        plt.show()


    #out = pd.DataFrame(data_window(synth_acc))
    #ids = pd.DataFrame(np.ones(125,),columns=["id"])
    
    # data, feature_names = feature_extraction(out,ids)

    # from sklearn.svm import SVR
    # y = np.random.randn(np.shape(data)[1])
    # regr = SVR(kernel='rbf')
    # regr.fit(np.transpose(data.values),y)









        ################################################## Haversine scaled plot #################################################

        # lat_len = 111332.67
        # lon_len = 63195.85

        # p79_gps = p79_gps[p79_gps['Longitude'] != 0]
        # p79_gps = p79_gps[p79_gps['Latitude'] != 0]
        # p79_gps = p79_gps.reset_index(drop=True)
        # plt.scatter(x=p79_gps['Longitude']*lon_len, y=p79_gps['Latitude']*lat_len,s=1,c="blue",label='p79 route')
        
        
        # plt.scatter(x=aran_location['LongitudeFrom']*lon_len, y=aran_location['LatitudeFrom']*lat_len,s=1,c="red",label='AranFrom')
        # plt.scatter(x=aran_location['LongitudeTo']*lon_len, y=aran_location['LatitudeTo']*lat_len,s=1,c="black",label='AranTo')
           
        # aran_start = [aran_location['LatitudeFrom'][2450],aran_location['LongitudeFrom'][2450]]
        # aran_end = [aran_location['LatitudeTo'][2450+4],aran_location['LongitudeTo'][2450+4]]

        # plt.plot(aran_start[1]*lon_len,aran_start[0]*lat_len,'ro')

        # p79_start_idx, start_dist = find_min_gps_vector(aran_start,p79_gps[['Latitude','Longitude']].values)
        # p79_end_idx, end_dist = find_min_gps_vector(aran_end,p79_gps[['Latitude','Longitude']].values)

        # p79_start_idx2, start_dist2 = tester_test(aran_start,p79_gps[['Latitude','Longitude']].values)
        # p79_end_idx2, end_dist2 = tester_test(aran_end,p79_gps[['Latitude','Longitude']].values)

        # haversine(aran_start,p79_gps[['Latitude','Longitude']].values[p79_start_idx],unit=Unit.METERS)
        # haversine(aran_start,p79_gps[['Latitude','Longitude']].values[p79_start_idx-20],unit=Unit.METERS)
        # haversine(aran_start,p79_gps[['Latitude','Longitude']].values[p79_start_idx2],unit=Unit.METERS)
        # haversine(aran_start,p79_gps[['Latitude','Longitude']].values[p79_start_idx2-20],unit=Unit.METERS)

        # x_val = [aran_start[1]*lon_len,p79_gps['Longitude'][p79_start_idx]*lon_len]
        # y_val = [aran_start[0]*lat_len,p79_gps['Latitude'][p79_start_idx]*lat_len]
        # _ = plt.plot(x_val,y_val,'r')

        # x_val = [aran_end[1]*lon_len,p79_gps['Longitude'][p79_end_idx]*lon_len]
        # y_val = [aran_end[0]*lat_len,p79_gps['Latitude'][p79_end_idx]*lat_len]
        # _ = plt.plot(x_val,y_val,'k')

        # x_val = [aran_start[1]*lon_len,p79_gps['Longitude'][p79_start_idx2]*lon_len]
        # y_val = [aran_start[0]*lat_len,p79_gps['Latitude'][p79_start_idx2]*lat_len]
        # _ = plt.plot(x_val,y_val,'magenta')

        # x_val = [aran_end[1]*lon_len,p79_gps['Longitude'][p79_end_idx2]*lon_len]
        # y_val = [aran_end[0]*lat_len,p79_gps['Latitude'][p79_end_idx2]*lat_len]
        # _ = plt.plot(x_val,y_val,'yellow')

        # custom_lines = [Line2D([0], [0], color='blue', lw=2),
        #         Line2D([0], [0], color='red', lw=2),
        #         Line2D([0], [0], color='black', lw=2),
        #         Line2D([0], [0], color='green', lw=2)]
        # plt.legend(custom_lines,['p79 route','AranFrom','AranTo','Segment'])
        # plt.title('Segmentation algorithm')
        # plt.xlabel('Longitude')
        # plt.ylabel('Latitude')


        # plt.axis([(aran_start[1])*lon_len-10,(aran_start[1])*lon_len+10,(aran_start[0])*lat_len-10,(aran_start[0])*lat_len+10])
        # # plt.axis([(aran_start[1]-0.0002),(aran_start[1]+0.0002),(aran_start[0]-0.0002),(aran_start[0]+0.0002)])

        # plt.show()

        #################################################################################################################################