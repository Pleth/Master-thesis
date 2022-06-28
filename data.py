from gc import collect
import pandas as pd
import h5py
import os
from utils.helperfunctions import preprocess_gm, find_min_gps
import glob
import pickle
import numpy as np

def load_data(path, 
              routes_to_use, 
              window, 
              stride, 
              passages_to_use = None,
              gm_sensors = ['obd.spd', 'gps_mapmatch']):
    """Function for loading and windowing the data for
    feature extraction.

    Args:
        path (str): 
            path to location of hdf5 files
        routes_to_use (list or str): 
            list of the routes from which data should be loaded
        window (float or int):
            window size (in meters) with which to partition the
            data
        stride (float or int):
            stride (in meters) to use when segmenting the data
        gm_sensors (list):
            list of the gm_sensors to include in the dataset

    Returns:
        pd.DataFrame: pandas dataframe with the windowed data
    """
    if routes_to_use == 'all':
        routes_to_use = ['CPH1_VH', 'CPH1_HH', 'CPH6_VH', 'CPH6_HH']

    collect_data = pd.DataFrame()
    for route in routes_to_use:
        hdf5_path = '{}/{}.hdf5'.format(path,route)
        aran_data = get_DI_aran(hdf5_path, keep_cols=False)
        route_file = h5py.File(hdf5_path, 'r')
        aligned_passes = route_file.attrs['aligned_passes']
        if not passages_to_use is None:
            aligned_passes = [path for path in aligned_passes if path in passages_to_use]
        for gm_pass in aligned_passes:
            passagefile = route_file[gm_pass]
            gm_data = preprocess_gm(passagefile, acc_sig = 'acc_fs_50',  signals = gm_sensors)
            gm_gps = pd.DataFrame(passagefile['aligned_gps'], columns = passagefile['aligned_gps'].attrs['chNames'])
            dataset = match_DRD_segments(GM_gps = gm_gps, GM_data = gm_data, DRD_data = aran_data, window = window, max_distance = 5, stride = stride)
            dataset['route'] = route
            dataset['passage'] = gm_pass
            collect_data = pd.concat((collect_data, dataset), axis = 0)
    
    return collect_data

def load_data_from_pickles(dset_path,
                            hdf5_file_path,
                            window, 
                            stride, 
                            passages_to_use = 'all',
                            gm_sensors = ['obd.spd', 'gps_mapmatch']):
    """Function for loading and windowing the data for
    feature extraction.

    Args:
        path (str): 
            path to location of hdf5 files
        routes_to_use (list or str): 
            list of the routes from which data should be loaded
        window (float or int):
            window size (in meters) with which to partition the
            data
        stride (float or int):
            stride (in meters) to use when segmenting the data
        gm_sensors (list):
            list of the gm_sensors to include in the dataset

    Returns:
        pd.DataFrame: pandas dataframe with the windowed data
    """
    if passages_to_use == 'all':
        aligned_passages = os.listdir(dset_path)
        passages_to_use = [passage for passage in aligned_passages if not passage == '.DS_Store']

    collect_data = pd.DataFrame()
    for passage in passages_to_use:
        # load in data from passage
        route = '_'.join(passage.split('_')[:2])
        route_path = '{}/{}.hdf5'.format(hdf5_file_path, route)
        hdf5file = h5py.File(route_path, 'r+')
        trip = passage.split('_')[3]
        pass_number = '_'.join(passage.split('_')[4:])
        passagefile = hdf5file['GM'][trip][pass_number]
        GM_data = preprocess_gm(passagefile, acc_sig = 'acc_fs_50',  signals = gm_sensors)
        aran_data = get_DI_aran(route_path)

        # get all aligned gps segments
        files = glob.glob('{}/{}/segment_*.pickle'.format(dset_path, passage))
        for file in files:
            with open(file, 'rb') as fb:
                newsig = pickle.load(fb)
            dataset = match_DRD_segments(GM_gps = newsig, GM_data = GM_data, DRD_data = aran_data, window = window, max_distance = 5, stride = stride)
            dataset['route'] = route
            dataset['passage'] = '{}/{}'.format(trip, pass_number)
            collect_data = pd.concat((collect_data, dataset), axis = 0)

    return collect_data


def get_DI_aran(hdf5_path, keep_cols = False):
    """Return a dataframe with the aran measures needed for 
    damage index calculation on aran data

    Args:
        hdf5_path (str): path to hdf5 file that contains
        the route of interest
    """
    route_file = h5py.File(hdf5_path, 'r')
    aran = route_file['aran/trip_1/pass_1']

    di_variables = {
        'Allig': ['AlligCracksSmall', 'AlligCracksMed', 'AlligCracksLarge'],
        'Cracks': ['CracksLongitudinalSmall', 'CracksLongitudinalMed', 'CracksLongitudinalLarge',
                   'CracksLongitudinalSealed', 'CracksTransverseSmall', 'CracksTransverseMed', 
                   'CracksTransverseLarge', 'CracksTransverseSealed'],
        'Pothole': ['PotholeAreaAffectedLow', 'PotholeAreaAffectedMed', 'PotholeAreaAffectedHigh',
                    'PotholeAreaAffectedDelam'],
        'Location': ['BeginChainage', 'EndChainage', 'LatitudeFrom', 'LongitudeFrom', 'LatitudeTo', 'LongitudeTo'],
    }

    df = None
    for category in di_variables.keys():
        variables_full = pd.DataFrame(aran[category][:], columns = aran[category].attrs['chNames'])
        variables = variables_full[di_variables[category]].fillna(value = 0)
        if df is None:
            df = variables
        else:
            df = pd.concat((df,variables), axis = 1)    
    alligsum = (3*df['AlligCracksSmall'] + 4*df['AlligCracksMed'] + 5*df['AlligCracksLarge'])**0.3
    medalligsum = (4*df['AlligCracksMed'] + 5*df['AlligCracksLarge'])**0.3
    cracksum = (df['CracksLongitudinalSmall']**2 + df['CracksLongitudinalMed']**3 + df['CracksLongitudinalLarge']**4 + \
                df['CracksLongitudinalSealed']**2 + 3*df['CracksTransverseSmall'] + 4*df['CracksTransverseMed'] + \
                5*df['CracksTransverseLarge'] + 2*df['CracksTransverseSealed'])**0.1
    medcracksum = (df['CracksLongitudinalMed']**3 + df['CracksLongitudinalLarge']**4 + \
                df['CracksLongitudinalSealed']**2 + 4*df['CracksTransverseMed'] + \
                5*df['CracksTransverseLarge'] + 2*df['CracksTransverseSealed'])**0.1
    potholesum = (5*df['PotholeAreaAffectedLow'] + 7*df['PotholeAreaAffectedMed'] + 10*df['PotholeAreaAffectedHigh'] + \
                  5*df['PotholeAreaAffectedDelam'])**0.1
    medpotholesum = (7*df['PotholeAreaAffectedMed'] + 10*df['PotholeAreaAffectedHigh'] + \
                  5*df['PotholeAreaAffectedDelam'])**0.1
    df['DI'] = alligsum + cracksum + potholesum
    df['medDI'] = medalligsum + medcracksum + medpotholesum
    if not keep_cols:
        for category in di_variables.keys():
            if not category == 'Location':
                df.drop(columns=di_variables[category], inplace=True)
    return df 


def match_DRD_segments(GM_gps, GM_data, DRD_data, window, max_distance = 5, stride = None, fs=50):

    if stride is None:
        stride = window

    drd_gps_start = ['LatitudeFrom', 'LongitudeFrom']
    drd_gps_end = ['LatitudeTo', 'LongitudeTo']
    beginchainage = 'BeginChainage'

    # extract corresponding DRD gps segment
    GM_gps = GM_gps.sort_values(by = 'TS_or_Distance').reset_index()
    GM_gps['p79_dist'] = abs(GM_gps['p79_dist']- GM_gps['p79_dist'][0])
    start_gps = GM_gps[['lat', 'lon']].iloc[0].values
    end_gps = GM_gps[['lat', 'lon']].iloc[-1].values
    DRD_start_idx, _, _ = find_min_gps(DRD_data[drd_gps_start[0]].values, DRD_data[drd_gps_start[1]].values, start_gps[0], start_gps[1])
    DRD_end_idx, _, _ = find_min_gps(DRD_data[drd_gps_end[0]].values, DRD_data[drd_gps_end[1]].values, end_gps[0], end_gps[1])
    DRD_segment = DRD_data.iloc[DRD_start_idx:DRD_end_idx]

    drd_chain = abs(DRD_segment[beginchainage] - DRD_segment[beginchainage].iloc[0]).values
    n_segments = int((drd_chain[-1]-window)/stride + 1)
    
    columns = ['time seg']
    for col in GM_data.columns:
        if not col == 'TS_or_Distance':
            columns.append(col)
    for col in DRD_data.columns:
        columns.append(col)

    idx = 0
    collected_data = pd.DataFrame(columns = columns)
    start_idx = 0
    # approximate maximum number of GM samples in a window
    gm_samples_window = window*fs/(20/3.6)
    end_idx = int(1.5*gm_samples_window)
    for seg in range(n_segments):
        begin_window = seg*stride
        end_window = begin_window + window
        seg_idx = (drd_chain >= begin_window) & (drd_chain < end_window)
        DRD_seg = DRD_segment[seg_idx]
        GM_gps_seg = GM_gps.iloc[start_idx:end_idx]

        gps_window_start = DRD_seg[drd_gps_start].iloc[0]
        gps_window_end = DRD_seg[drd_gps_end].iloc[-1]

        _, GM_start_idx, start_dist = find_min_gps(gps_window_start[0], gps_window_start[1], GM_gps_seg['lat'].values, GM_gps_seg['lon'].values)
        _, GM_end_idx, end_dist = find_min_gps(gps_window_end[0], gps_window_end[1], GM_gps_seg['lat'].values, GM_gps_seg['lon'].values)

        if start_dist < max_distance and end_dist < max_distance:
            start_idx = int(np.max([start_idx + GM_start_idx - 0.5*gm_samples_window,0]))
            end_idx = int(start_idx + GM_end_idx + 0.5*gm_samples_window)
            GM_start_time = GM_gps_seg['TS_or_Distance'].iloc[GM_start_idx]
            GM_end_time = GM_gps_seg['TS_or_Distance'].iloc[GM_end_idx]
            collected_data.loc[idx, 'time seg'] = [GM_start_time, GM_end_time]
            GM_seg = GM_data[(GM_data['TS_or_Distance']>=GM_start_time) & (GM_data['TS_or_Distance']<=GM_end_time)]
            for col in GM_seg.columns:
                if not col == 'TS_or_Distance':
                    col_data = GM_seg[col].values
                    collected_data.loc[idx, col] = col_data
            for col in DRD_seg.columns:
                collected_data.loc[idx, col] = DRD_seg[col].values
            idx += 1
    return collected_data